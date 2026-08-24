"""Issue #26 verification, phase 3: torch.profiler on all 5 validated
queries, kernel-class shares compared against what sol_seconds_
breakdown() predicts. Classification rules built from real kernel
names captured by tests/gpu/torch_profiler_explore.py's exploratory
run against THIS engine (DeepGEMM + Triton + FlashAttention-3 - the
old exploration's vLLM-based kernel-class rules don't transfer, see
that script's docstring).

Same _execute-seam / worker.execute.local() technique as the
exploratory script - no changes to worker.py.

One profiled call per query (not two): sol_breakdown is computed from
token counts, not from timing, so it's safe to read off the SAME
profiled call - only wall_s/sol_s would be unreliable under profiling
overhead (per the old exploration's "a profiler changes the timing it
measures" note), and this script doesn't need those, only the
kernel-class percentages and sol_breakdown's percentages.

Run from the quail/ directory (tee to a file per house rule):

    uv run modal run tests/gpu/torch_profiler_compare.py::main \\
        2>&1 | tee results/torch_profiler_compare.log
"""

import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"

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
    .add_local_python_source("quail")
    .add_local_dir("quail/calibration",
                   remote_path="/root/quail/calibration")
)

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

SOL_CHECK_QUERIES = ["IMDB-1", "IMDB-2", "IMDB-5", "BIO-2", "FEV-5"]

# Built from tests/gpu/torch_profiler_explore.py's real captured
# kernel names on IMDB-1 (results/torch_profiler_explore.json), not
# guessed or ported from the old (vLLM-based) exploration's rules.
KERNEL_CLASS_RULES = (
    ("attention", ("flashattn", "flash::")),
    ("projection", ("deep_gemm", "nvjet")),
    ("elementwise", ("silu_mul_quant", "add_rms_norm_quant",
                     "per_token_group_quant", "rms_norm_kernel",
                     "fused_add_rms_norm")),
    ("unmodeled", ("memcpy", "qk_norm_rope", "kv_row_scatter",
                   "index_elementwise", "vectorized_gather",
                   "indexselect", "reduce_kernel",
                   "vectorized_elementwise", "unrolled_elementwise",
                   "elementwise_kernel")),
)


def _classify(name: str) -> str:
    k = name.lower()
    for cls, keys in KERNEL_CLASS_RULES:
        if any(s in k for s in keys):
            return cls
    return "other"


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def compare() -> str:
    import json

    from torch.profiler import ProfilerActivity, profile

    import quail
    from quail.bench.quailb import build_sets, queries, register_sets
    from quail.planner.plan import EngineConfig
    from quail.runtime import worker as worker_mod

    d = build_sets("/results/quailb_data", sf=0.1, lf=1)
    sess = quail.Session(EngineConfig(gpus=1, model="qwen3-4b-fp8",
                                      cpu_memory_gb=80))
    # Session.store_enabled defaults to True (session.py) - quailb.py's
    # cold/warm passes turn it off/on explicitly, this script didn't,
    # so the profiled run could silently restore KV a query's own
    # earlier warmup call (or an earlier query sharing a table) had
    # already written, understating the real causal-build kernel
    # time sol_breakdown is compared against. Force cold (full
    # compute, no restores) for every query profiled here, matching
    # what sol_breakdown's causal_doc_lengths assumes when there is
    # nothing to restore from.
    sess.set_store(False)
    register_sets(sess, d)
    qdefs = queries(sess)

    # warm the container once (cold boot noise not the point here) -
    # store is off above, so this can't pollute the profiled runs'
    # KV state either way, but keeping order explicit.
    _desc0, build0 = qdefs[SOL_CHECK_QUERIES[0]]
    build0().run(_execute=lambda p: worker_mod.execute.local(p))

    results = []
    for qid in SOL_CHECK_QUERIES:
        desc, build = qdefs[qid]
        print(f"[torch_profiler_compare] {qid}: {desc}", flush=True)
        captured = {}

        def profiled_execute(payload, captured=captured):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = worker_mod.execute.local(payload)
            captured["prof"] = prof
            return out

        res = build().run(_execute=profiled_execute)
        prof = captured["prof"]

        trace_path = f"/tmp/compare_trace_{qid}.json"
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as f:
            events = json.load(f)["traceEvents"]

        gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
        classes = dict(projection=0.0, elementwise=0.0, attention=0.0,
                      unmodeled=0.0, other=0.0)
        total = 0.0
        for e in events:
            if e.get("cat") in gpu_cats and e.get("dur", 0) > 0:
                classes[_classify(e.get("name", ""))] += e["dur"]
                total += e["dur"]

        measured_pct = {k: round(v / total * 100, 2) if total else 0.0
                        for k, v in classes.items()}

        bd = res.report["sol_breakdown"]
        sol_total = sum(bd.values())
        formula_pct = {
            "projection": round(bd["projection"] / sol_total * 100, 2),
            "elementwise": round(bd["elementwise"] / sol_total * 100, 2),
            "attention": round((bd["causal_attention"]
                               + bd["streaming_attention"])
                              / sol_total * 100, 2),
        } if sol_total else {"projection": 0, "elementwise": 0,
                            "attention": 0}

        row = dict(query=qid, desc=desc, wall_s=res.report["wall_s"],
                  sol_s=res.report["sol_s"], sol_breakdown=bd,
                  measured_pct=measured_pct, formula_pct=formula_pct,
                  total_kernel_us=total)
        results.append(row)
        print(f"[torch_profiler_compare] {json.dumps(row)}", flush=True)

    with open("/results/torch_profiler_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    fc = compare.spawn()
    print(f"function call id: {fc.object_id}")
    print(fc.get())
