"""GPU smoke test for the FEVER join query: runs the loader and query end to end."""

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
# to the existing milestone app, same as tests/gpu/milestone1.py.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)


@app.function(image=image, gpu="H100!", memory=98304, timeout=2400,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def run_fever() -> str:
    from quail.bench.quailb import run_suite

    suite = run_suite("/results/quailb_data", sf=0.01, lf=1,
                      only={"B16"}, out_path="/results/fever_b16.json")
    results_vol.commit()
    return str(suite)


@app.local_entrypoint()
def main():
    # House rule: print the function call id and keep it in the tee
    # file before waiting on the result, so a dropped connection
    # doesn't lose the run - re-fetch with
    # modal.FunctionCall.from_id("<id>").get().
    fc = run_fever.spawn()
    print(f"function call id: {fc.object_id}")
    print(fc.get())
