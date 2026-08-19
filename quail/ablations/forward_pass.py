"""The forward-pass ablation ladder: stock vLLM to the current packed
executor, one change per rung, all on the committed 10,000-document
five-filter workload (~3.83M fresh tokens).

  A0  stock vLLM, fp8 KV - the committed production setting:
      pipelined per-(document, stage) client, prefix caching on,
      25,305 step tokens, constrained YES/NO sampler
  A1  stock vLLM, bf16 KV - isolates the fp8 KV conversion tax
      (the ladder measured 96,946 -> 102,820 tok/s from dtype alone)
  A2  packed executor with vLLM's own between-GEMM kernels: "no
      engine" and the kept-KV machinery (arena with the host-index
      cache, paged cross-attention, LSE merge, 12-row YES/NO
      readout), 110,376-token chunks, pinned-memory staging - the
      current executor in every respect except the kernels
  A3  A2 + our three Triton kernels (norm+add+quantize,
      silu+mul+quantize, qk-norm+rope) - the current executor

Pinned staging is part of the packed base configuration, not a rung:
issue #12 already banked its worth (39.6 s pre-#12 to 34.6 s after,
split ~4 s arena host-index cache + ~1.4 s staging; see the 2026-08-19
report).

Predictions, stated before the run (house rule), from banked numbers:

  A0  42.8-43.2 s (the exploration's committed fp8 stock band).
      Config note: the admission budget stays at the bf16 value
      (374,891 tokens) so dtype is the only variable against A1; the
      plan-derived fp8 budget (749,782) oversubscribes the measured
      ~473k-token fp8 pool 1.6x and thrashes (47-79 s, banked in
      baseline_filter3.json).
  A1  38.7-39.0 s (banked: results/baseline_filter4.json)
  A2  ~42.9 s: 43.9 s measured before the direct KV write
      (kv_row_scatter); the write fix is worth ~1.0 s.
  A3  ~33.5 s: 34.5 s before the fix; the filter gate with the fix
      measured 33.5/33.6 s (results/m1_filter_kvscatter.json).

Container discipline: the packed rungs share one boot, so their
difference carries no container drift. Each stock rung has its own
container - the v1 engine core is a separate process that holds the
GPU until it exits, so two engine boots in one container fail
(measured: 1.34 GiB free at the second boot). Stock-to-packed
comparisons carry the +/-3% container band.

Gates: A3 must reproduce the banked current-executor counts exactly
(1,807 survivors, 6,294 wrong of 23,113 answered; banked in
results/m1_filter.json). A2 runs different quant kernels, so its
answers are reported against A3's, not gated to zero: this
checkpoint's YES/NO margins are thin, and the exploration measured
1,768 flipped answers per 10,000 from one silu-kernel swap, so
low-thousands disagreement at 10k documents is the expected band.
Every rung reports wrong answers against the planted flags.

Run from the quail/ directory (tee to a file per house rule):

    uv run modal run ablations/forward_pass.py::run_all 2>&1 | tee results/ablation_forward.log
"""

import json
import os

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

# Same image and pins as tests/gpu/milestone1.py: vllm==0.26.0 is
# measurement hygiene (same kernels, same loader), not an API
# dependency.
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0", "huggingface_hub", "pandas", "pyarrow",
                 "numpy", "datasets")
    .env({"VLLM_LOGGING_LEVEL": "WARNING",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
          "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
    .add_local_python_source("quail", "baselines")
    # corpus.py lives next to the milestone cells; mount it beside the
    # ablation cell rather than moving shared workload code
    .add_local_dir("tests/gpu", remote_path="/root/gpu_tests")
)

# House rule: never create new Modal app names - new GPU cells attach
# to an existing app.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

GPU_KW = dict(image=image, gpu="H100!", memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})

# the committed bf16-KV admission budget, shared by both stock rungs
# so KV dtype is the only variable between them
STOCK_BUDGET = 374_891
STOCK_STEP_TOKENS = 25_305
STOCK_MAX_SEQS = 2648


def _write(result, name):
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/ablations", exist_ok=True)
    with open(f"/results/ablations/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


def _wrong_count(answers_by_doc, flags):
    """answers_by_doc: doc -> list of stage bits. flags: doc -> planted
    0/1 per stage."""
    wrong = 0
    for d, row in answers_by_doc.items():
        for j, bit in enumerate(row):
            if bit != int(flags[d][j]):
                wrong += 1
    return wrong


# --------------------------------------------------------- stock rungs

STOCK_PREDICTIONS = {
    "A0": "42.8-43.2 s (exploration fp8 band)",
    "A1": "38.7-39.0 s (banked baseline_filter4.json)",
}


@app.function(timeout=5400, **GPU_KW)
def stock_rung(rung: str, kv_dtype: str, n_docs: int = 10000,
               reps: int = 2) -> str:
    """One stock rung in its own container: the pipelined stock client
    over the five-filter workload. kv_dtype is "fp8" for A0 and "auto"
    for A1 - "auto" follows the model dtype, which is bf16 KV for this
    checkpoint (vLLM has no literal "bf16" kv_cache_dtype value).
    Everything else is identical between the two."""
    import sys

    sys.path.insert(0, "/root/gpu_tests")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from baselines.stock import run_filter_chain
    from corpus import MODEL, build_corpus
    from quail.executor.loop import yes_no_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    yes, no = yes_no_ids(tokenizer)

    report = dict(
        cell="ablation_stock", rung=rung, kv=kv_dtype, n_docs=n_docs,
        workload="committed 10k five-filter, pipelined "
                 "per-(document, stage) client, prefix caching on",
        budget_tokens=STOCK_BUDGET, step_tokens=STOCK_STEP_TOKENS,
        max_num_seqs=STOCK_MAX_SEQS,
        prediction=STOCK_PREDICTIONS[rung],
        runs=[])
    print(f"[ablation_stock] {rung} prediction: "
          f"{STOCK_PREDICTIONS[rung]}", flush=True)

    llm = LLM(model=MODEL, kv_cache_dtype=kv_dtype,
              max_num_batched_tokens=STOCK_STEP_TOKENS,
              max_num_seqs=STOCK_MAX_SEQS,
              gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=1,
                              min_tokens=1,
                              allowed_token_ids=sorted(yes | no))
    engine = llm.llm_engine
    # warm the engine (kernel compile, allocator) outside the
    # measured reps
    run_filter_chain(engine, sampling, body_ids[:64], q_ids,
                     STOCK_BUDGET, tag="w", yes_ids=yes)
    for rep in range(reps):
        r = run_filter_chain(engine, sampling, body_ids, q_ids,
                             STOCK_BUDGET, tag=f"{rung}{rep}",
                             yes_ids=yes)
        by_doc = {}
        for (i, j), bit in r["answers"].items():
            by_doc.setdefault(i, {})[j] = bit
        by_doc = {i: [stages[j] for j in sorted(stages)]
                  for i, stages in by_doc.items()}
        row = dict(rung=rung, kv=kv_dtype, rep=rep,
                   wall=round(r["wall"], 2),
                   requests=r["requests"],
                   fresh_tokens=r["prompt_tokens"] - r["cached_tokens"],
                   cached_tokens=r["cached_tokens"],
                   survivors=len(r["survivors"]),
                   wrong=_wrong_count(by_doc, flags))
        report["runs"].append(row)
        print(f"[ablation_stock] {row}", flush=True)
    return _write(report, f"stock_{rung.lower()}")


# -------------------------------------------------------- packed rungs

# name, Pipeline kernels, pinned staging, the change the rung adds.
# Pinned staging is on in both rungs: it is part of the packed base
# configuration (issue #12 banked its worth), not a rung.
PACKED_RUNGS = (
    ("A2", "vllm", True,
     "packed executor, vLLM's between-GEMM kernels, kept KV, "
     "110,376-token chunks, pinned staging"),
    ("A3", "quail", True,
     "+ our three Triton kernels (the current executor)"),
)

PACKED_PREDICTIONS = {
    "A2": "~42.9 s: this rung measured 43.9 s before the direct KV "
          "write (kv_row_scatter); the write fix is worth ~1.0 s",
    "A3": "~33.5 s: 34.5 s before the KV write fix; the filter gate "
          "with the fix already measured 33.5/33.6 s "
          "(m1_filter_kvscatter.json)",
}

# banked current-executor counts (results/m1_filter.json); A3 must
# reproduce them exactly
BANKED_A3 = dict(survivors=1807, wrong=6294, answered=23113)


@app.function(timeout=5400, **GPU_KW)
def packed_rungs(n_docs: int = 10000, reps: int = 2) -> str:
    """A2 and A3 in one container, one boot: same loop, same arena,
    same attention path, same pinned staging; only the kernel set
    moves between rungs."""
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from corpus import MODEL, build_corpus
    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     warm_kernels)
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_model(MODEL)
    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    arena_tok = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    spec = QWEN3_4B_FP8
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)

    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    corpus_tokens = sum(len(b) for b in body_ids)

    report = dict(
        cell="ablation_packed", n_docs=n_docs,
        corpus_tokens=corpus_tokens, n_filters=len(q_ids),
        exec_budget=exec_budget, arena_tokens=arena_tok, kv="bf16",
        rungs={name: dict(change=change,
                          prediction=PACKED_PREDICTIONS[name])
               for name, _, _, change in PACKED_RUNGS},
        runs={})
    print(f"[ablation_packed] predictions: "
          f"{json.dumps(PACKED_PREDICTIONS)}", flush=True)

    # one warmup covers every rung: the DeepGEMM configuration sweep
    # is kernel-mode independent, and the budget-sized pass warms the
    # Triton kernels and the attention path
    t_warm = time.perf_counter()
    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, body_ids,
                     q_ids, exec_budget)
    torch.cuda.synchronize()
    kernel_cache.commit()
    report["warmup_s"] = round(time.perf_counter() - t_warm, 2)
    print(f"[ablation_packed] warmup {report['warmup_s']} s",
          flush=True)

    answers_by_rung = {}
    for name, kernels, pinned, _change in PACKED_RUNGS:
        pipeline.kernels = kernels
        with torch.inference_mode():
            # per-rung warm, unmeasured: first-call op init for the
            # vLLM CUDA ops stays out of the measured reps
            run_filter(torch, arena, pipeline, async_ans,
                       body_ids[:256], q_ids, exec_budget,
                       pinned=pinned)
            torch.cuda.synchronize()
            rows = []
            for rep in range(reps):
                torch.cuda.reset_peak_memory_stats()
                timers = {}
                t0 = time.perf_counter()
                answers, spans, tokens = run_filter(
                    torch, arena, pipeline, async_ans, body_ids,
                    q_ids, exec_budget, timing=timers, pinned=pinned)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                answered = sum(len(v) for v in answers.values())
                survivors = [d for d, row in answers.items()
                             if len(row) == len(q_ids) and all(row)]
                row = dict(
                    rung=name, kernels=kernels, pinned=pinned, rep=rep,
                    wall=round(wall, 2), fresh_tokens=tokens,
                    tok_s=round(tokens / wall, 1), answered=answered,
                    survivors=len(survivors),
                    wrong=_wrong_count(answers, flags),
                    chunks=len(spans),
                    gpu_s=round(sum(e0.elapsed_time(e1)
                                    for _, e0, e1 in spans) / 1e3, 2),
                    peak_gib=round(
                        torch.cuda.max_memory_allocated() / 2**30, 2),
                    cpu_phase_s={k: round(v, 3) for k, v
                                 in sorted(timers.items())})
                rows.append(row)
                print(f"[ablation_packed] {row}", flush=True)
        answers_by_rung[name] = answers
        report["runs"][name] = rows

    # gates: A3 is the current executor, so its counts must equal the
    # banked ones exactly. A2 runs different quant kernels, so its
    # answers are reported against A3, not gated to zero.
    def disagreements(x, y):
        return sum(bit_x != bit_y
                   for d in x for bit_x, bit_y in zip(x[d], y[d]))

    a2, a3 = answers_by_rung["A2"], answers_by_rung["A3"]
    last_a3 = report["runs"]["A3"][-1]
    report["gates"] = dict(
        a2_vs_a3_disagreements=disagreements(a2, a3),
        a3_vs_banked=dict(
            measured={k: last_a3[k] for k in BANKED_A3},
            banked=BANKED_A3),
    )
    report["pass"] = all(last_a3[k] == v for k, v in BANKED_A3.items())
    return _write(report, "packed")


# ---------------------------------------------------------- profiling

def _categorize(name):
    """Kernel name -> time bucket. Order matters: our Triton kernel
    names contain substrings ("add_rms", "quant") that also appear in
    vLLM's op names, so the Triton names are checked first."""
    low = name.lower()
    if "deep_gemm" in low or "sm90_fp8" in low or "gemm" in low:
        return "gemm"
    if "flash" in low or "attn" in low:
        return "attention"
    if any(k in low for k in ("silu_mul_quant", "add_rms_norm_quant",
                              "qk_norm_rope")):
        return "triton_fused"
    if any(k in low for k in ("rms_norm", "rotary", "silu_and_mul")):
        return "vllm_elementwise"
    if "quant" in low:
        return "quant"
    if any(k in low for k in ("memcpy", "copy", "index", "cat",
                              "gather", "scatter")):
        return "copies"
    return "other"


@app.function(timeout=3600, **GPU_KW)
def profile_packed(n_docs: int = 3000) -> str:
    """Per-kernel-category GPU time for A2 and A3 in one container,
    at the 110k-chunk geometry. Answers why A2 (vLLM's small kernels)
    is slower than A1 (stock engine): the stock kernel composition is
    banked (10.77 us/token at 25,305-token steps: 5.52 GEMM, 0.81
    attention, 0.43 KV write, 4.00 the small kernels); this profile
    measures the same buckets for our loop at 110,376-token chunks.

    Prediction: A2's small-kernel buckets (vllm_elementwise + quant)
    exceed the stock 4.00 us/token; GEMM and attention match A3.
    """
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    from corpus import MODEL, build_corpus
    from quail.executor.arena import KVArena
    from quail.executor.attention import Pipeline
    from quail.executor.loop import (Answerer, AsyncAnswers, run_filter,
                                     warm_kernels)
    from quail.executor.model import load_model
    from quail.planner import budgets
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = load_model(MODEL)
    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    arena_tok = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    spec = QWEN3_4B_FP8
    arena = KVArena(n_layers=spec.layers,
                    n_pages=arena_tok // budgets.PAGE_TOKENS,
                    page_tokens=budgets.PAGE_TOKENS,
                    n_kv=spec.n_kv, d_head=spec.d_head,
                    dtype=torch.bfloat16)
    pipeline = Pipeline(model, arena)
    answerer = Answerer(torch, F, model, tokenizer)
    async_ans = AsyncAnswers(torch, answerer)
    exec_budget = min(chunk, pipeline.max_chunk_tokens)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)

    with torch.inference_mode():
        warm_kernels(torch, arena, pipeline, async_ans, body_ids,
                     q_ids, exec_budget)
    torch.cuda.synchronize()
    kernel_cache.commit()

    result = {"n_docs": n_docs, "exec_budget": exec_budget,
          "stock_reference_us_per_token": dict(
              gemm=5.521, attention=0.813, kv_write_fp8=0.427,
              small_kernels=4.004, total=10.77,
              note="banked engine profile at 25,305-token steps, "
                   "fp8 KV (fusion_ab.json control cell)"),
          "rungs": {}}
    for name, kernels, pinned, _ in PACKED_RUNGS:
        pipeline.kernels = kernels
        with torch.inference_mode():
            # unprofiled reference: the true rate
            _, _, tokens = run_filter(torch, arena, pipeline,
                                      async_ans, body_ids, q_ids,
                                      exec_budget, pinned=pinned)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                run_filter(torch, arena, pipeline, async_ans, body_ids,
                           q_ids, exec_budget, pinned=pinned)
                torch.cuda.synchronize()

        cats, counts = {}, {}
        rows = []
        for ev in prof.key_averages():
            cuda_us = getattr(ev, "self_device_time_total", 0) or \
                getattr(ev, "self_cuda_time_total", 0)
            if not cuda_us:
                continue
            # skip op wrappers ("_C::...") and runtime events: they
            # carry the same device time as the kernel they launched
            # and would double-count it
            if ev.key.startswith("_C::") or "Command Buffer" in ev.key:
                continue
            cat = _categorize(ev.key)
            cats[cat] = cats.get(cat, 0.0) + cuda_us
            counts[cat] = counts.get(cat, 0) + ev.count
            rows.append((round(cuda_us / 1e6, 3), ev.count,
                         ev.key[:90]))
        rows.sort(reverse=True)
        busy = sum(cats.values())
        result["rungs"][name] = dict(
            fresh_tokens=tokens,
            cuda_busy_s=round(busy / 1e6, 2),
            us_per_token=round(busy / tokens, 2),
            category_s={k: round(v / 1e6, 2) for k, v
                        in sorted(cats.items())},
            category_us_per_token={k: round(v / tokens, 2)
                                   for k, v in sorted(cats.items())},
            category_launches={k: counts[k] for k in sorted(counts)},
            top_kernels=[dict(s=s, n=n, name=k) for s, n, k
                         in rows[:25]])
        prof.export_chrome_trace(
            f"/results/ablations/profile_{name.lower()}.json.gz")
        print(f"[profile_packed] {name}: "
              f"{json.dumps(result['rungs'][name]['category_us_per_token'])}",
              flush=True)
    return _write(result, "profile_packed")


# ---------------------------------------------------------- entrypoint

def _save(payload, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(json.loads(payload), f, indent=2)
    print(f"saved {out}")


@app.local_entrypoint()
def run_profile(out: str = "results/ablation_profile.json"):
    _save(profile_packed.remote(), out)


@app.local_entrypoint()
def run_all(n_docs: int = 10000, reps: int = 2,
            out: str = "results/ablation_forward.json"):
    a0_h = stock_rung.spawn("A0", "fp8", n_docs, reps)
    a1_h = stock_rung.spawn("A1", "auto", n_docs, reps)
    packed_h = packed_rungs.spawn(n_docs, reps)
    merged = dict(stock={"A0": json.loads(a0_h.get()),
                         "A1": json.loads(a1_h.get())},
                  packed=json.loads(packed_h.get()))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"saved {out}")
