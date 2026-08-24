"""Issue #26 validation: five representative QUAIL-B queries
(IMDB-1 filter-only, IMDB-2 join-only, IMDB-5 three-filter-chain then
join, BIO-2 join on a different dataset, FEV-5 two-sided filter then
join), sf=0.1, Qwen3-4B-FP8, cold + warm.

Not the full 17-query suite (see tests/gpu/quailb_full_4b.py for
that) - this is a small, cheap, shape-diverse subset to check the new
sol_s / sol_efficiency / cost_dollars fields (reports/
2026-08-23-sol-throughput-cost.md) against real measured numbers
before trusting them on a bigger run.

Prediction, stated before running: sol_efficiency should land well
under 100% on every query (the old exploration measured ~35% for the
full serving loop) - somewhere in the 10-40% range depending on query
shape - and must never exceed 100% on any query. If it does, that is
the efficiency-invariant assertion firing (quailb.py's
EFFICIENCY_TOLERANCE check), meaning a bug in sol_seconds()'s model
or in the wall_s/fresh_tokens measurement, not a real speedup.

Run from the quail/ directory (tee to a file per house rule):

    uv run modal run tests/gpu/sol_check.py::main \\
        2>&1 | tee results/sol_check_sf0.1_4b.log

Spawn, not remote: the function call id prints before anything waits
on the result, so a dropped local connection doesn't lose the run -
re-fetch with modal.FunctionCall.from_id("<id>").get().
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

# House rule: never create new Modal app names - this cell attaches
# to the existing milestone app, same as the other one-off cells.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)

SOL_CHECK_QUERIES = {"IMDB-1", "IMDB-2", "IMDB-5", "BIO-2", "FEV-5"}


@app.function(image=image, gpu="H100!", memory=98304, timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def run_sol_check() -> str:
    from quail.bench.quailb import run_suite

    suite = run_suite("/results/quailb_data", sf=0.1,
                      only=SOL_CHECK_QUERIES, model="qwen3-4b-fp8",
                      out_path="/results/sol_check_sf0.1_4b.json")
    results_vol.commit()
    return str(suite)


@app.local_entrypoint()
def main():
    fc = run_sol_check.spawn()
    print(f"function call id: {fc.object_id}")
    print(fc.get())
