"""Exploratory step for the torch.profiler cross-check (issue #26
verification, phase 3): capture real kernel names from the CURRENT
engine (DeepGEMM + Triton + FlashAttention-3 - not the old
exploration's vLLM-based engine, whose kernel-class rules don't
transfer) on one query, so the classification rules for the real
comparison script are built from real names, not guessed.

Uses Query.run()'s existing _execute seam to wrap the real worker
call in torch.profiler, calling worker.execute.local(payload)
directly - runs in-process on the same container, no RPC, no changes
to worker.py at all.

Run from the quail/ directory (tee to a file per house rule):

    uv run modal run tests/gpu/torch_profiler_explore.py::main \\
        2>&1 | tee results/torch_profiler_explore.log
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


@app.function(image=image, gpu="H100!", memory=98304, timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def explore() -> str:
    import json

    from torch.profiler import ProfilerActivity, profile

    import quail
    from quail.bench.quailb import build_sets, queries, register_sets
    from quail.planner.plan import EngineConfig
    from quail.runtime import worker as worker_mod

    d = build_sets("/results/quailb_data", sf=0.1, lf=1)
    sess = quail.Session(EngineConfig(gpus=1, model="qwen3-4b-fp8",
                                      cpu_memory_gb=80))
    # Force cold (store off) - see the matching note in
    # torch_profiler_compare.py. This script warms up and profiles
    # the SAME query (IMDB-1) back to back, which would otherwise
    # let the profiled run restore KV the warmup call had just
    # written, understating real causal-build kernel time.
    sess.set_store(False)
    register_sets(sess, d)
    qdefs = queries(sess)
    _desc, build = qdefs["IMDB-1"]

    captured = {}

    def profiled_execute(payload):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = worker_mod.execute.local(payload)
        captured["prof"] = prof
        return out

    # warm the container first (cold boot noise not the point here)
    build().run(_execute=lambda p: worker_mod.execute.local(p))

    res = build().run(_execute=profiled_execute)
    prof = captured["prof"]

    trace_path = "/tmp/explore_trace.json"
    prof.export_chrome_trace(trace_path)
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]

    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
    by_name = {}
    total = 0
    for e in events:
        if e.get("cat") in gpu_cats and e.get("dur", 0) > 0:
            name = e.get("name", "")
            by_name[name] = by_name.get(name, 0) + e["dur"]
            total += e["dur"]

    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:40]
    out = dict(
        query="IMDB-1", wall_s=res.report["wall_s"],
        sol_s=res.report["sol_s"],
        sol_breakdown=res.report["sol_breakdown"],
        total_kernel_us=total,
        distinct_kernel_names=len(by_name),
        top_kernels=[[name, round(us, 1), round(us / total * 100, 2)]
                    for name, us in top],
    )
    print(json.dumps(out, indent=2), flush=True)
    with open("/results/torch_profiler_explore.json", "w") as f:
        json.dump(out, f, indent=2)
    results_vol.commit()
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main():
    fc = explore.spawn()
    print(f"function call id: {fc.object_id}")
    print(fc.get())
