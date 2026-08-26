"""Reruns IMDB-2, BIO-1, and BIO-2 (cold + warm) and checks answers
against QUAIL-B ground truth - the accuracy numbers cited in
reports/2026-08-25-vllm-opbench-baseline.md.

Run from the repository root (tee to a file per house rule):

    uv run modal run tests/gpu/join_rerun.py::main \\
        --model qwen3-4b-fp8 2>&1 | tee results/join_rerun_4b.log
    uv run modal run tests/gpu/join_rerun.py::main \\
        --model qwen3-32b-fp8 2>&1 | tee results/join_rerun_32b.log

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
# to the existing milestone app, same as quailb_bio2_32b.py and
# tests/gpu/fever_query.py.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def run_join_rerun(model: str) -> str:
    import time
    from quail.bench.quailb import run_suite

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = f"/results/benchmark/{ts}-join-rerun-{model}.json"
    suite = run_suite(
        "/results/quailb_data", sf=0.1, only={"IMDB-2", "BIO-1", "BIO-2"},
        model=model, accuracy=True,
        # Pinned: the volume's "active" collection for this corpus
        # currently points at pre-PR-#56 labels, not PR #57's fix.
        # gt_80e7582534b349bc61c087595f2e0a51 is PR #57's own collection.
        ground_truth_collection="gt_80e7582534b349bc61c087595f2e0a51",
        prediction=(
            "Cold and warm answers agree within this run. Both joins "
            "show low precision (still near-all-TRUE); BIO-1's F7 "
            "filter shows a real precision/recall tradeoff."),
        out_path=out_path)
    results_vol.commit()
    return str(suite)


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8"):
    fc = run_join_rerun.spawn(model=model)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
