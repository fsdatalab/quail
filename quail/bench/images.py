"""The Modal images the benchmark and experiment scripts run on."""

import modal

# the uv the images sync with; pyproject.toml's required-version must
# match it (tests/test_images.py checks), and the value stays a literal
# because Modal imports this module inside the container, where the
# repository's pyproject.toml is not mounted
UV_VERSION = "0.12.13"
CUDA_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
# every kernel cache under one mount, so a volume keeps them warm
CACHE_ENV = {
    "QUAIL_CACHE_DIR": "/root/.cache/kernels",
    "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
    "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
    "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
}


def _cuda_base() -> modal.Image:
    return (modal.Image.from_registry(CUDA_BASE, add_python="3.12")
            .entrypoint([])
            .apt_install("git")
            .env(CACHE_ENV))


def gpu_image(*local_dirs: tuple[str, str]) -> modal.Image:
    """The GPU image: quail's locked dependencies and the dev group.

    Args:
        local_dirs: (local path, remote path) pairs mounted after the
            quail source; Modal refuses a build step after a mount.
    """
    image = (_cuda_base()
             .uv_sync(groups=["dev"], uv_version=UV_VERSION)
             .add_local_python_source("quail"))
    for local, remote in local_dirs:
        image = image.add_local_dir(local, remote_path=remote)
    return image


def sglang_image() -> modal.Image:
    """The GPU image with SGLang's serving stack in place of vLLM."""
    return (_cuda_base()
            .uv_sync(groups=["dev"], uv_version=UV_VERSION,
                     extra_options="--no-install-package vllm")
            .uv_pip_install("sglang==0.5.18")
            .add_local_python_source("quail"))


def cpu_image() -> modal.Image:
    """The CPU image: the locked dependencies without the GPU stack."""
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .uv_sync(groups=["dev"], uv_version=UV_VERSION,
                     extra_options="--no-install-package vllm")
            .add_local_python_source("quail"))
