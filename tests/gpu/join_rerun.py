"""Rerun the three two-table joins after the shared prompt fix."""

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


@app.function(image=image, memory=98304, timeout=7200,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def run_join_rerun(model: str) -> str:
    import json
    import time
    from quail.bench.quailb import run_suite

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = f"/results/benchmark/{ts}-join-prompt-fix-{model}.json"
    suite = run_suite(
        "/results/quailb_data", sf=0.1,
        only={"IMDB-2", "BIO-2", "FEV-2"},
        model=model, accuracy=False,
        prediction=(
            "IMDB-2 processes 2,419,233 fresh tokens, 59% fewer than "
            "5,944,233. BIO-2 processes 2,635,599, 76% fewer than "
            "10,979,399. FEV-2 processes 145,359 with evidence as "
            "the anchor."),
        out_path=out_path)
    results_vol.commit()
    return json.dumps(suite, sort_keys=True)


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8"):
    fc = run_join_rerun.spawn(model=model)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
