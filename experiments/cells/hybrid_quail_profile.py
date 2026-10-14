"""Time Quail's Qwen3.5 forward pass by layer part on a realistic chunk.

Wraps the pipeline's mixers and the GDN kernels with synchronized
timers and runs one filter over documents the size of IMDB reviews.
The timers add launch gaps, so shares matter more than the totals.
The convolution and the recurrence are also timed per phase: the prefix
phase runs a document from zero state and saves its state; the suffix
phase runs the question from that saved state.

Prediction: the Gated DeltaNet layers (projection GEMMs, convolution,
recurrence, gated norm) take at least 60 percent of the forward time,
and the recurrence kernel plus the convolution take more than the
attention layers do. The suffix phase, with the copies around its
launches, takes at least 15 percent of the Gated DeltaNet core time.

    uv run modal run experiments/cells/hybrid_quail_profile.py \
        --model qwen3.5-4b-bf16 --backend flashinfer \
        2>&1 | tee /tmp/hybrid_quail_profile.log

The record is written to
/results/ablations/<model>-quail-profile-<backend>.json.
"""

import json
import os
import time
from collections import defaultdict

import modal

try:
    from quail.deploy.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}
DOCUMENTS = 600
DOCUMENT_TOKENS = 370
QUESTION = "Is the text above about a person?"


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def profile(model: str, backend: str = "triton") -> dict:
    """Return seconds by forward-pass part for one filter run."""
    os.environ["QUAIL_GDN_PREFILL"] = backend
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor import gdn
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import run_filter
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows, AsyncAnswers
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost import budgets
    from quail.logical import true_false_ids
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    true_ids, false_ids = true_false_ids(tokenizer)
    budget = budgets.chunk_budget(spec, H100_SXM)
    loaded = load_model(spec.hf_name, spec.revision or None,
                        answer_token_ids=true_ids | false_ids,
                        max_batched_tokens=budget,
                        text_only=spec.vision_tower)
    full_pages, sliding = budgets.arena_pages(
        spec, H100_SXM, budget, DOCUMENT_TOKENS)
    arena = KVArena(
        n_layers=spec.layers, n_pages=full_pages,
        page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv, d_head=spec.d_head,
        dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
        sliding_layers=spec.sliding_layer_set,
        sliding_window=spec.sliding_window, n_sliding_pages=sliding,
        **checkpoint_arena_args(spec, H100_SXM, budget, full_pages))
    pipeline = build_pipeline(spec, loaded, arena)
    rows = AnswerRows(torch, F, loaded, true_ids, false_ids)
    answers = AsyncAnswers(torch, rows)

    seconds = defaultdict(float)
    calls = defaultdict(int)

    def timed(name, function):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            seconds[name] += time.perf_counter() - started
            calls[name] += 1
            return result
        return wrapper

    pipeline._gated_delta_net = timed("gdn_layer", pipeline._gated_delta_net)
    import quail.backends.quail.executor.models.qwen3_5 as pipeline_module
    pipeline_module.gdn_core = timed("gdn_core", pipeline_module.gdn_core)
    pipeline._attention = timed("attention_layer", pipeline._attention)
    pipeline._mlp = timed("mlp", pipeline._mlp)
    # the prefix phase starts from zero state; a suffix phase reads a slot
    current = {"phase": ""}
    phase_tokens = defaultdict(int)
    phase_sequences = defaultdict(int)

    def phase_name(phase):
        return "prefix" if phase.read_slots is None else "suffix"

    real_conv = gdn.causal_conv_kernel

    def conv_by_phase(x, weight, phase, history, bias=None, activate=True):
        name = phase_name(phase)
        phase_tokens[name + "_conv"] += int(x.shape[0])
        return timed(f"gdn_convolution_{name}", real_conv)(
            x, weight, phase, history, bias, activate=activate)

    gdn.causal_conv_kernel = conv_by_phase
    real_batches = gdn._kernel_in_batches

    def batches_by_phase(spec, q, k, v, g, beta, load, store, phase, scale):
        name = phase_name(phase)
        current["phase"] = name
        phase_tokens[name + "_recurrence"] += int(q.shape[0])
        phase_sequences[name] += len(phase.lengths)
        return timed(f"gdn_recurrence_with_copies_{name}", real_batches)(
            spec, q, k, v, g, beta, load, store, phase, scale)

    gdn._kernel_in_batches = batches_by_phase
    gdn.causal_conv_reference = timed("gdn_convolution_reference",
                                      gdn.causal_conv_reference)
    mixer = pipeline.gdn_layer.linear_attn
    for name in ("in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
        for layer in pipeline.layers:
            if hasattr(layer, "linear_attn"):
                module = getattr(layer.linear_attn, name)
                module.forward = timed(f"gdn_{name}", module.forward)
    del mixer
    real_kernel = gdn.gated_delta_kernel

    def kernel_by_phase(*args, **kwargs):
        return timed(f"gdn_recurrence_kernel_{current['phase']}",
                     real_kernel)(*args, **kwargs)

    gdn.gated_delta_kernel = kernel_by_phase
    pipeline.embed = timed("embedding", pipeline.embed)

    word = tokenizer("the review describes a film and its cast in detail "
                     "including the plot", add_special_tokens=False)["input_ids"]
    documents = []
    for index in range(DOCUMENTS):
        body = (word * (DOCUMENT_TOKENS // len(word) + 2))[:DOCUMENT_TOKENS]
        documents.append([10 + (index % 40)] + body)
    question = tokenizer("\n" + QUESTION + spec.turn_suffix,
                         add_special_tokens=False)["input_ids"]
    with torch.inference_mode():
        run_filter(torch, arena, pipeline, answers, documents[:60],
                   [question], budget, arena_writes=True,
                   attention_mode="unified")
        for key in arena.resident_keys():
            arena.free_key(key)
        seconds.clear()
        calls.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        run_filter(torch, arena, pipeline, answers, documents, [question],
                   budget, arena_writes=True, attention_mode="unified")
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
    tokens = sum(len(d) + len(question) for d in documents)
    record = {"model": model, "backend": backend,
              "documents": DOCUMENTS, "tokens": tokens,
              "wall_s": wall, "tokens_per_second": tokens / wall,
              "chunk_budget": budget, "seconds": dict(seconds),
              "calls": dict(calls), "phase_tokens": dict(phase_tokens),
              "phase_sequences": dict(phase_sequences)}
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-quail-profile-{backend}.json", "w") as out:
        json.dump(record, out, indent=1)
    results.commit()
    return record


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16", backend: str = "triton"):
    call = profile.spawn(model, backend)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=1), flush=True)
