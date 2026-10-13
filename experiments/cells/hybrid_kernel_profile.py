"""Compare GPU kernel time of Quail and stock vLLM on the same documents.

Both engines run 600 distinct documents of 371 tokens, each followed by
one question, through a filter stage. Each engine runs the stage once
unprofiled for wall time, then once under torch.profiler for kernel time
by name. The two functions run in separate containers.

The gap between the engine's wall time and the sum of its kernel
durations is time the GPU spent idle, which the part timers of
hybrid_quail_profile.py cannot see.

Prediction: kernels that are neither GEMM, attention, GDN recurrence,
nor GDN convolution (norms, elementwise, copies) take 5 to 10 percent
of Quail's kernel time and under 3 percent of vLLM's; Quail's kernel
time is at least 90 percent of its wall time; vLLM's total kernel time
is at least 20 percent below Quail's on the same tokens.

    uv run modal run experiments/cells/hybrid_kernel_profile.py \
        --model qwen3.5-4b-bf16 --backend flashinfer \
        2>&1 | tee /tmp/hybrid_kernel_profile.log

Records: /results/ablations/<model>-kernel-profile-{quail,vllm}.json.
"""

import json
import os
import re
import time

import modal

try:
    from quail.bench.images import gpu_image
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
WARM_DOCUMENTS = 60
BODY_TOKENS = 369
QUESTION = "Is the text above about a person?"
# first match wins; the order puts specific names before general ones
CATEGORIES = (
    ("gemm", r"gemm|cutlass|nvjet|cublas|sm90_xmma|splitk|wgmma"),
    ("attention", r"flash|fmha|attention|attn|paged"),
    ("gdn_recurrence", r"chunk_gated|gated_delta|recurrent|delta_rule|"
                       r"gdn|fwd_kernel|wy_fast|chunk_local_cumsum|"
                       r"kkt|solve_tril|chunk_fwd"),
    ("gdn_convolution", r"conv"),
    ("norm", r"rms|norm|layer_norm"),
    ("memcpy", r"memcpy|memset|copy"),
    ("elementwise", r"elementwise|vectorized|silu|mul|add|sigmoid|cat|"
                    r"index|where|pow|mean|rsqrt|to_copy|fill|reduce"),
)


def category(name: str) -> str:
    """Return the first category whose pattern matches a kernel name."""
    lowered = name.lower()
    for label, pattern in CATEGORIES:
        if re.search(pattern, lowered):
            return label
    return "other"


def _kernel_table(profile) -> dict:
    """Return {kernel name: [microseconds, launches]} from a profiler."""
    from torch.autograd import DeviceType

    table = {}
    for event in profile.events():
        if event.device_type != DeviceType.CUDA:
            continue
        micros = getattr(event, "device_time", None)
        if micros is None:
            micros = event.cuda_time
        entry = table.setdefault(event.name, [0.0, 0])
        entry[0] += micros
        entry[1] += 1
    return table


def _documents(tokenizer):
    """Return distinct document token lists and the question tokens."""
    word = tokenizer("the review describes a film and its cast in detail "
                     "including the plot", add_special_tokens=False)["input_ids"]
    documents = []
    for index in range(DOCUMENTS):
        body = (word * (BODY_TOKENS // len(word) + 2))[:BODY_TOKENS]
        # two leading ids make every document unique, so no prefix repeats
        documents.append([10 + index % 40, 100 + index // 40] + body)
    return documents


# innermost operators only: contiguous, clone, and to contain copy_
COPY_OPERATORS = ("aten::copy_", "aten::index_select", "aten::index_copy_",
                  "aten::index_put_", "aten::cat", "aten::gather",
                  "aten::index", "aten::fill_", "aten::zero_",
                  "aten::pow", "aten::mul", "aten::add", "aten::mean",
                  "aten::div", "aten::rsqrt", "aten::sigmoid",
                  "aten::softplus", "aten::exp")


def _copy_sites(profile, keep: int = 60) -> list:
    """Return [operator, stack, microseconds, calls] by operator and site.

    The microseconds are the kernels each operator launched. The first
    row counts the events seen, with a stack, and with kernels.
    """
    from torch.autograd import DeviceType

    totals, seen, stacked, launched = {}, 0, 0, 0
    for event in profile.events():
        if event.device_type != DeviceType.CPU:
            continue
        if event.name not in COPY_OPERATORS:
            continue
        seen += 1
        stack = tuple(event.stack or ())
        micros = sum(kernel.duration for kernel in (event.kernels or ()))
        stacked += bool(stack)
        launched += micros > 0
        if micros <= 0:
            continue
        frames = [f for f in stack if "site-packages" not in f
                  or "quail" in f or "vllm" in f][:7]
        entry = totals.setdefault((event.name, tuple(frames)), [0.0, 0])
        entry[0] += micros
        entry[1] += 1
    sites = [[name, list(frames), micros, count]
             for (name, frames), (micros, count) in totals.items()]
    sites.sort(key=lambda item: -item[2])
    return [["counts", [], seen, [stacked, launched]]] + sites[:keep]


def _record(model, engine, wall, table, tokens, **extra) -> dict:
    busy = sum(micros for micros, _ in table.values()) / 1e6
    return {"model": model, "engine": engine, "documents": DOCUMENTS,
            "tokens": tokens, "wall_s": wall, "kernel_s": busy,
            "kernels": table, **extra}


def _save(record: dict, engine: str) -> None:
    os.makedirs("/results/ablations", exist_ok=True)
    path = f"/results/ablations/{record['model']}-kernel-profile-{engine}.json"
    with open(path, "w") as out:
        json.dump(record, out)
    results.commit()


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def quail(model: str, backend: str = "flashinfer",
          stacks: bool = False) -> dict:
    """Kernel time of one Quail filter stage.

    With stacks, the profiled pass also records the Python call stack of
    each copy-like operator and the GPU time of the kernels it launched.
    """
    os.environ["QUAIL_GDN_PREFILL"] = backend
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

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
    document_tokens = BODY_TOKENS + 2
    full_pages, sliding = budgets.arena_pages(
        spec, H100_SXM, budget, document_tokens)
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
    documents = _documents(tokenizer)
    question = tokenizer("\n" + QUESTION + spec.turn_suffix,
                         add_special_tokens=False)["input_ids"]

    def stage(batch):
        run_filter(torch, arena, pipeline, answers, batch, [question],
                   budget, arena_writes=True, attention_mode="unified")
        torch.cuda.synchronize()
        for key in arena.resident_keys():
            arena.free_key(key)

    with torch.inference_mode():
        stage(documents[:WARM_DOCUMENTS])
        started = time.perf_counter()
        stage(documents)
        wall = time.perf_counter() - started
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA,
                            torch.profiler.ProfilerActivity.CPU],
                with_stack=stacks) as profile:
            stage(documents)
    tokens = sum(len(d) + len(question) for d in documents)
    record = _record(model, "quail", wall, _kernel_table(profile), tokens,
                     backend=backend)
    if stacks:
        record["copy_sites"] = _copy_sites(profile)
    _save(record, "quail-stacks" if stacks else "quail")
    return record


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def vllm(model: str) -> dict:
    """Kernel time of the same documents through stock vLLM."""
    import torch
    from transformers import AutoTokenizer

    from quail.backends.vllm import VLLMEngine
    from quail.logical import true_false_ids
    from quail.specs import MODELS

    spec = MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_name, revision=spec.revision or None)
    true_ids, false_ids = true_false_ids(tokenizer)
    state, boot = VLLMEngine().boot(spec, sorted(true_ids | false_ids))
    client, params = state["client"], state["sampling_params"]
    documents = _documents(tokenizer)
    question = tokenizer("\n" + QUESTION + spec.turn_suffix,
                         add_special_tokens=False)["input_ids"]
    prompts = [{"prompt_token_ids": document + question}
               for document in documents]
    client.generate(prompts[:WARM_DOCUMENTS], params)
    client.reset_prefix_cache()
    started = time.perf_counter()
    client.generate(prompts, params)
    wall = time.perf_counter() - started
    client.reset_prefix_cache()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU]) as profile:
        client.generate(prompts, params)
        torch.cuda.synchronize()
    tokens = sum(len(p["prompt_token_ids"]) for p in prompts)
    record = _record(model, "vllm", wall, _kernel_table(profile), tokens,
                     prefix_caching=state["capacity"].get(
                         "enable_prefix_caching"),
                     boot_s=boot["boot_s"])
    _save(record, "vllm")
    return record


def _summary(record: dict) -> dict:
    """Return kernel seconds by category."""
    totals = {}
    for name, (micros, _) in record["kernels"].items():
        label = category(name)
        totals[label] = totals.get(label, 0.0) + micros / 1e6
    return totals


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16", backend: str = "flashinfer",
         stacks: bool = False):
    calls = {"quail": quail.spawn(model, backend, stacks)}
    if not stacks:
        calls["vllm"] = vllm.spawn(model)
    for engine, call in calls.items():
        print(f"function call id: {call.object_id} ({engine})", flush=True)
    records = {engine: call.get() for engine, call in calls.items()}
    for engine, record in records.items():
        print(f"\n{engine}: wall {record['wall_s']:.2f} s, kernels "
              f"{record['kernel_s']:.2f} s, tokens {record['tokens']}")
        for label, seconds in sorted(_summary(record).items(),
                                     key=lambda item: -item[1]):
            print(f"  {label:16s} {seconds:7.3f} s "
                  f"{100 * seconds / record['kernel_s']:5.1f}%")
        top = sorted(record["kernels"].items(), key=lambda i: -i[1][0])[:12]
        for name, (micros, launches) in top:
            print(f"    {micros / 1e6:7.3f} s {launches:6d}  {name[:90]}")
    for operator, frames, micros, count in records["quail"].get(
            "copy_sites", [])[:31]:
        print(f"\n{micros / 1e6:6.3f} s {count:5d}  {operator}")
        for frame in frames[:5]:
            print(f"        {frame[-110:]}")
