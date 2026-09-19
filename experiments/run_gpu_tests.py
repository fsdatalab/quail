"""Run the GPU tests under tests/gpu on Modal.

    uv run modal run experiments/run_gpu_tests.py 2>&1 | tee results/gpu-tests.log

Prints pytest's output and fails when a test fails.
"""

import modal

# the uv the images sync with; pyproject.toml requires this version
UV_VERSION = "0.12.13"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    # quail's pinned dependencies and the dev group (pytest among
    # them) from uv.lock; the quail source and the tests are mounted
    .uv_sync(groups=["dev"], uv_version=UV_VERSION)
    .env({
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    })
    .add_local_python_source("quail")
    .add_local_dir("tests/gpu", remote_path="/root/tests/gpu")
)

app = modal.App("quail-milestone1")
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def run_tests(keyword: str = "") -> int:
    """Run pytest over tests/gpu and return its exit code."""
    import pytest

    args = ["/root/tests/gpu", "-x", "-q", "-s", "-p", "no:cacheprovider"]
    if keyword:
        args += ["-k", keyword]
    return int(pytest.main(args))


@app.local_entrypoint()
def main(keyword: str = ""):
    call = run_tests.spawn(keyword)
    print(f"function call id: {call.object_id}", flush=True)
    code = call.get()
    print(f"pytest exit code: {code}", flush=True)
    if code:
        raise SystemExit(code)
