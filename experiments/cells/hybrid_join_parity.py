"""Compare Quail's join path with stock vLLM on the same pair prompts.

A join writes each anchor document and a frame into KV (and, on a hybrid
model, into a checkpoint that the frame extends), then streams every
partner after it. This cell sends a few anchors of different lengths
and a handful of partners through run_stages, and sends the same
concatenated prompt of every (anchor, partner) pair through stock vLLM.
The reference records each pair's TRUE minus FALSE logprob. Quail
records the same difference and its answer bit.

Two steps, each in its own container so the engines do not share a GPU.

Prediction: for each anchor, the per-pair differences agree with
stock vLLM within 0.25 nats at the median and 1.0 at the maximum, and
the answer bit (TRUE > FALSE) of every pair agrees on at least 95
percent of pairs.

    uv run modal run experiments/cells/hybrid_join_parity.py \
        --model qwen3.5-4b-bf16 2>&1 | tee /tmp/hybrid_join_parity.log

Records: /results/ablations/<model>-join-parity-{reference,quail}.json.
"""

import json
import os
import statistics

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
BUDGET = 8192
ANCHOR_WORDS = (30, 120, 400, 900)
PARTNERS = 8
FRAME = ("\nQuestion: does the text above describe the same topic as the "
         "passage below?\nPASSAGE:")
TAIL = "\nAnswer TRUE or FALSE."
TOPICS = ("a river", "an engineer", "a market", "a storm", "a violinist",
          "a bakery", "a mountain", "a teacher")


def _anchor_text(index):
    words = ANCHOR_WORDS[index % len(ANCHOR_WORDS)]
    topic = TOPICS[index % len(TOPICS)]
    sentence = (f"Entry {index} describes {topic} in detail, including the "
                f"place, the season, and what happened next. ")
    return (sentence * (words // 14 + 1))[: words * 6]


def _partner_text(index):
    topic = TOPICS[index % len(TOPICS)]
    return f" A short passage about {topic} number {index}."


def _pieces(spec, tokenizer):
    """Return anchors, frame ids, and per partner suffix ids, tokenized."""
    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    anchors = [ids(spec.turn_prefix + _anchor_text(i))
               for i in range(len(ANCHOR_WORDS))]
    frame = ids(FRAME)
    suffixes = [ids(_partner_text(j) + TAIL + spec.turn_suffix)
                for j in range(PARTNERS)]
    return anchors, frame, suffixes


def _run_join(torch, arena, pipeline, answers, anchors, suffixes, frame, budget):
    """Run one join stage of partner suffixes; return answer rows per anchor."""
    from quail.backends.quail.executor.stages import Stage, run_stages

    stages = [Stage(suffixes=suffixes, readout=answers, frame=frame)]
    stage_answers, _, _ = run_stages(
        torch, arena, pipeline, stages, anchors, budget,
        on_settled=lambda anchor, survived, row: None,
        attention_mode="unified", label="join (1 stage)")
    return stage_answers[0]


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def reference(model: str) -> dict:
    """Stock vLLM's TRUE minus FALSE logprob for every pair prompt."""
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"

    from transformers import AutoTokenizer

    from quail.baselines.vllm import VLLMEngine
    from quail.planner.prompt_lowering import true_false_ids
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    true_ids, false_ids = true_false_ids(tokenizer)
    allowed = sorted(true_ids | false_ids)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    class Engine(VLLMEngine):
        """The tuned engine with logprobs taken after the answer mask."""

        def llm_kwargs(self, spec):
            return {**super().llm_kwargs(spec),
                    "logprobs_mode": "processed_logprobs"}

    state, _ = Engine().boot(spec, allowed)
    anchors, frame, suffixes = _pieces(spec, tokenizer)
    prompts = [{"prompt_token_ids": anchor + frame + suffix}
               for anchor in anchors for suffix in suffixes]
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0,
                            allowed_token_ids=allowed, logprobs=len(allowed))
    outputs = state["client"].generate(prompts, params)
    differences = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs[0]
        best_true = max(logprobs[i].logprob for i in true_ids if i in logprobs)
        best_false = max(logprobs[i].logprob for i in false_ids
                         if i in logprobs)
        differences.append(best_true - best_false)
    record = {"model": model, "deep_gemm_warmup": "skip",
              "anchors": len(anchors), "partners": PARTNERS,
              "anchor_tokens": [len(a) for a in anchors],
              "difference": differences,
              "true_ids": sorted(true_ids), "false_ids": sorted(false_ids)}
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-join-parity-reference.json",
              "w") as out:
        json.dump(record, out)
    results.commit()
    return record


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def quail(model: str) -> dict:
    """Quail's TRUE minus FALSE logit for every pair, through run_stages."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AnswerRows
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.cost import budgets
    from quail.specs import H100_SXM, MODELS

    spec = MODELS[model]
    results.reload()
    with open(f"/results/ablations/{model}-join-parity-reference.json") as f:
        ref = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    anchors, frame, suffixes = _pieces(spec, tokenizer)
    allowed = sorted(set(ref["true_ids"]) | set(ref["false_ids"]))
    loaded = load_model(spec.hf_name, spec.revision or None,
                        answer_token_ids=allowed, max_batched_tokens=BUDGET,
                        moe_backend=spec.moe_backend,
                        text_only=spec.vision_tower)
    full_pages, sliding = budgets.arena_pages(spec, H100_SXM, BUDGET)
    arena = KVArena(
        n_layers=spec.layers, n_pages=full_pages,
        page_tokens=budgets.PAGE_TOKENS, n_kv=spec.n_kv, d_head=spec.d_head,
        dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
        sliding_layers=spec.sliding_layer_set,
        sliding_window=spec.sliding_window, n_sliding_pages=sliding,
        **checkpoint_arena_args(spec, H100_SXM, BUDGET, full_pages))
    pipeline = build_pipeline(spec, loaded, arena)
    rows = AnswerRows(torch, F, loaded, set(ref["true_ids"]),
                      set(ref["false_ids"]))

    class Differences:
        """Read TRUE minus FALSE logits as stage values."""

        dtype = np.float32

        def submit(self, normed):
            true, false = rows.logits(normed)
            return (true - false).float().cpu().numpy()

        @staticmethod
        def result(handle):
            return handle

    record = {"model": model, "slots": arena.n_state_slots,
              "alone": [], "alone_bits": [], "together": [],
              "together_bits": []}
    with torch.inference_mode():
        # each anchor alone, then every anchor in one run
        for anchor in anchors:
            answers = Differences()
            out = _run_join(torch, arena, pipeline, answers, [anchor],
                            suffixes, frame, BUDGET)
            record["alone"].append(out[0].tolist())
            record["alone_bits"].append(
                [int(b > 0) for b in out[0]])
            for key in arena.resident_keys():
                arena.free_key(key)
        answers = Differences()
        out = _run_join(torch, arena, pipeline, answers, anchors,
                        suffixes, frame, BUDGET)
        record["together"] = [float(v) for i in range(len(anchors)) for v in out[i]]
        record["together_bits"] = [
            [int(b > 0) for b in out[i]] for i in range(len(anchors))]
        for key in arena.resident_keys():
            arena.free_key(key)
    torch.cuda.synchronize()

    reference = ref["difference"]
    per_anchor = [reference[i * PARTNERS:(i + 1) * PARTNERS]
                  for i in range(len(anchors))]
    errors, agree = [], []
    for got, want, bits in zip(record["alone"], per_anchor,
                               record["alone_bits"]):
        errors.extend(abs(a - b) for a, b in zip(got, want))
        # bit i is partner i; the reference bit is the sign of its difference
        want_bits = [int(v > 0) for v in want]
        agree.extend(int(b == w) for b, w in zip(bits, want_bits))
    together_errors = [abs(a - b) for a, b in zip(
        record["together"], reference)]
    together_agree = [
        int(b == int(w > 0))
        for bits, want in zip(record["together_bits"], per_anchor)
        for b, w in zip(bits, want)]
    record["summary"] = {
        "pairs": len(reference),
        "alone_median_abs_error": statistics.median(errors),
        "alone_max_abs_error": max(errors),
        "alone_bit_agreement": sum(agree) / len(agree),
        "together_median_abs_error": statistics.median(together_errors),
        "together_max_abs_error": max(together_errors),
        "together_bit_agreement": sum(together_agree) / len(together_agree)}
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{model}-join-parity-quail.json",
              "w") as out_file:
        json.dump(record, out_file)
    results.commit()
    return record


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16", reuse_reference: bool = False):
    if not reuse_reference:
        first = reference.spawn(model)
        print(f"function call id: {first.object_id} (reference)", flush=True)
        first.get()
    second = quail.spawn(model)
    print(f"function call id: {second.object_id} (quail)", flush=True)
    print(json.dumps(second.get()["summary"], indent=1), flush=True)
