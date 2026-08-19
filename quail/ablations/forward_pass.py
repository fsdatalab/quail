"""The forward-pass ablation ladder: stock vLLM to the current packed
executor, one change per rung, all on the committed 10,000-document
five-filter workload (~3.83M fresh tokens).

  A0  stock vLLM, fp8 KV - the committed production setting:
      pipelined per-(document, stage) client, prefix caching on,
      25,305 step tokens, constrained YES/NO sampler
  A1  stock vLLM, bf16 KV - isolates the fp8 KV conversion tax
      (the ladder measured 96,946 -> 102,820 tok/s from dtype alone)
  A2  packed executor with vLLM's own between-GEMM kernels, kept KV,
      110,376-token chunks - "no engine" and the kept-KV machinery
      (arena, paged cross-attention, LSE merge, 12-row YES/NO readout)
      land together, per the study design
  A3  A2 + our three Triton kernels (norm+add+quantize,
      silu+mul+quantize, qk-norm+rope)
  A4  A3 + pinned-memory staging for chunk packing - the current
      executor (issue #9)

Predictions, stated before the run (house rule), from banked numbers:

  A0  42.8-43.2 s (the exploration's committed fp8 stock band).
      Config note: the admission budget stays at the bf16 value
      (374,891 tokens) so dtype is the only variable against A1; the
      plan-derived fp8 budget (749,782) oversubscribes the measured
      ~473k-token fp8 pool 1.6x and thrashes (47-79 s, banked in
      baseline_filter3.json).
  A1  38.7-39.0 s (banked: results/baseline_filter4.json)
  A2  ~41 s: the full executor's 9.0 us/token plus the ladder's 1.7
      us/token kernel worth. Never measured; this rung is the new
      information in the study.
  A3  39.4-39.9 s (banked: results/m1_filter_final1/2.json), possibly
      slightly under: this rung keeps the issue-#12 arena host-index
      cache and reverts only the staging copies.
  A4  34.6 s (banked: results/m1_filter.json)

Stock rungs and packed rungs run in separate containers of one app:
the engine's KV pool and the arena cannot share one 80 GB card
without changing each other's sizing. Within each group the rungs
share one boot, so inside-group differences carry no container drift;
the cross-group comparison (A1 vs A2) carries the +/-3% band.

Gates: A3 and A4 run identical kernels, so their answers must be
identical (staging changes copy mechanics, not values) - 0
disagreements required. A2 runs different quant kernels, so its
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

@app.function(timeout=5400, **GPU_KW)
def stock_rungs(n_docs: int = 10000, reps: int = 2) -> str:
    """A0 and A1 in one container: the pipelined stock client over the
    five-filter workload, fp8 KV then bf16 KV. Two engine boots, one
    per dtype; nothing else differs."""
    import gc
    import sys
    import time

    sys.path.insert(0, "/root/gpu_tests")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from baselines.stock import run_filter_chain
    from corpus import MODEL, build_corpus
    from quail.executor.loop import yes_no_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    body_ids, q_ids, flags = build_corpus(tokenizer, n_docs)
    yes, no = yes_no_ids(tokenizer)

    report = dict(
        cell="ablation_stock", n_docs=n_docs,
        workload="committed 10k five-filter, pipelined "
                 "per-(document, stage) client, prefix caching on",
        budget_tokens=STOCK_BUDGET, step_tokens=STOCK_STEP_TOKENS,
        max_num_seqs=STOCK_MAX_SEQS,
        rungs={
            "A0": dict(change="stock vLLM, fp8 KV (committed "
                              "production setting)",
                       prediction="42.8-43.2 s (exploration fp8 "
                                  "band)"),
            "A1": dict(change="KV dtype fp8 -> bf16",
                       prediction="38.7-39.0 s (banked "
                                  "baseline_filter4.json)"),
        },
        runs={})
    print(f"[ablation_stock] predictions: "
          f"{json.dumps({k: v['prediction'] for k, v in report['rungs'].items()})}",
          flush=True)

    for rung, kv_dtype in (("A0", "fp8"), ("A1", "bf16")):
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
        rows = []
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
            rows.append(row)
            print(f"[ablation_stock] {row}", flush=True)
        report["runs"][rung] = rows
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)
    return _write(report, "stock")


# -------------------------------------------------------- packed rungs

# name, Pipeline kernels, pinned staging, the change the rung adds
PACKED_RUNGS = (
    ("A2", "vllm", False,
     "packed executor, vLLM's between-GEMM kernels, kept KV, "
     "110,376-token chunks"),
    ("A3", "quail", False,
     "+ our three Triton kernels"),
    ("A4", "quail", True,
     "+ pinned-memory staging (the current executor)"),
)

PACKED_PREDICTIONS = {
    "A2": "~41 s (9.0 us/token full executor + the ladder's 1.7 "
          "us/token kernel worth); never measured",
    "A3": "39.4-39.9 s banked (m1_filter_final1/2), possibly "
          "slightly under (the #12 arena host-index cache stays)",
    "A4": "34.6 s banked (m1_filter.json)",
}


@app.function(timeout=5400, **GPU_KW)
def packed_rungs(n_docs: int = 10000, reps: int = 2) -> str:
    """A2, A3, A4 in one container, one boot: same loop, same arena,
    same attention path; only the kernel set and the staging switch
    move between rungs."""
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

    # gates: A3 vs A4 must be identical (same kernels; staging changes
    # copy mechanics, not values). A2 runs different quant kernels, so
    # its answers are reported against A3, not gated to zero.
    def disagreements(x, y):
        return sum(bit_x != bit_y
                   for d in x for bit_x, bit_y in zip(x[d], y[d]))

    a2, a3, a4 = (answers_by_rung[n] for n in ("A2", "A3", "A4"))
    report["gates"] = dict(
        a3_vs_a4_disagreements=disagreements(a3, a4),
        a2_vs_a3_disagreements=disagreements(a2, a3),
    )
    report["pass"] = report["gates"]["a3_vs_a4_disagreements"] == 0
    return _write(report, "packed")


# ---------------------------------------------------------- entrypoint

@app.local_entrypoint()
def run_all(n_docs: int = 10000, reps: int = 2,
            out: str = "results/ablation_forward.json"):
    stock_h = stock_rungs.spawn(n_docs, reps)
    packed_h = packed_rungs.spawn(n_docs, reps)
    merged = dict(stock=json.loads(stock_h.get()),
                  packed=json.loads(packed_h.get()))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"saved {out}")
