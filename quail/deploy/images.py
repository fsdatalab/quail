"""The Modal images the benchmark and experiment scripts run on.

Each image mounts the quail package and the bench package, which is
not in the wheel, so the benchmark runner imports inside a container.
"""

import os
from importlib.metadata import version
from pathlib import Path

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
    "FLASHINFER_WORKSPACE_BASE": "/root/.cache/kernels",
    "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
    "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
    "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
    # FlashAttention 4 (heads wider than 256) keeps compiled kernels in
    # memory only unless its disk cache is on
    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "/root/.cache/kernels/flash_attention_cute",
}


# the package that declares Quail's vLLM plugin, installed in the images
VLLM_PLUGIN_DIR = Path(__file__).parent / "vllm_plugin"

# the Rust workspace the planner's native module builds from
REPOSITORY = Path(__file__).resolve().parents[2]
NATIVE_SOURCE = "/opt/quail-native"
MATURIN = "maturin>=1.9,<2"
# the model and device spec files quail.specs reads
SPEC_DATA = Path(__file__).resolve().parents[1] / "specs" / "data"

# local variables copied into the image when set
FORWARDED_ENV = ("QUAIL_GDN_PREFILL",)


def _cuda_base() -> modal.Image:
    # a run picks the Gated DeltaNet prefill kernel from its own shell
    forwarded = {name: os.environ[name] for name in FORWARDED_ENV
                 if name in os.environ}
    return (modal.Image.from_registry(CUDA_BASE, add_python="3.12")
            .entrypoint([])
            .apt_install("git")
            .env(CACHE_ENV | forwarded))


def _with_sources(image: modal.Image) -> modal.Image:
    """Mount the quail and bench packages, with the model and device specs.

    A Python source mount copies Python files only; the specs are TOML.
    """
    return (image.add_local_python_source("quail", "bench")
            .add_local_dir(SPEC_DATA, "/root/quail/specs/data"))


def _with_native_planner(image: modal.Image) -> modal.Image:
    """Build the _quail_native module, which plans queries in Rust, into an image."""
    return (image.apt_install("curl", "build-essential")
            .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y "
                          "--profile minimal")
            .add_local_file(REPOSITORY / "Cargo.toml", f"{NATIVE_SOURCE}/Cargo.toml",
                            copy=True)
            .add_local_file(REPOSITORY / "Cargo.lock", f"{NATIVE_SOURCE}/Cargo.lock",
                            copy=True)
            .add_local_dir(REPOSITORY / "crates", f"{NATIVE_SOURCE}/crates", copy=True)
            .uv_pip_install(MATURIN, uv_version=UV_VERSION)
            .run_commands(
                # an image builds once: Cargo's default release settings,
                # not the workspace's fast-rebuild ones
                f"cd {NATIVE_SOURCE} && PATH=$HOME/.cargo/bin:$PATH "
                "CARGO_PROFILE_RELEASE_INCREMENTAL=false "
                "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 "
                f"maturin build --release -m crates/quail-native/Cargo.toml "
                f"-o {NATIVE_SOURCE}/wheels",
                # the image's environment has no pip; a wheel is a zip archive
                "python -c \"import glob, sysconfig, zipfile; "
                "zipfile.ZipFile(glob.glob("
                f"'{NATIVE_SOURCE}/wheels/quail_native-*.whl')[0]).extractall("
                "sysconfig.get_paths()['platlib'])\"",
                "python -c 'import _quail_native'"))


def gpu_image(*local_dirs: tuple[str, str],
              packages: tuple[str, ...] = (),
              commands: tuple[str, ...] = ()) -> modal.Image:
    """The GPU image: quail's locked dependencies and the dev group.

    Args:
        local_dirs: (local path, remote path) pairs mounted after the
            quail source; Modal refuses a build step after a mount.
        packages: Extra packages installed before the mounts, such as
            a profiler a cell runs.
        commands: Shell commands run before the mounts, such as an
            install into a folder outside the project's environment.
    """
    image = (_cuda_base().uv_sync(groups=["dev"], uv_version=UV_VERSION)
             .add_local_dir(VLLM_PLUGIN_DIR, "/opt/quail-vllm-plugin", copy=True)
             .uv_pip_install("/opt/quail-vllm-plugin", uv_version=UV_VERSION))
    image = _with_native_planner(image)
    if packages:
        image = image.uv_pip_install(*packages, uv_version=UV_VERSION)
    if commands:
        image = image.run_commands(*commands)
    image = _with_sources(image)
    for local, remote in local_dirs:
        image = image.add_local_dir(local, remote_path=remote)
    return image


def pypi_gpu_image(*local_dirs: tuple[str, str]) -> modal.Image:
    """A GPU image with the installed PyPI release of Quail.

    Args:
        local_dirs: (local path, remote path) pairs copied after Quail is
            installed.
    """
    package = f"quail-engine=={version('quail-engine')}"
    image = _cuda_base().uv_pip_install(package, uv_version=UV_VERSION)
    for local, remote in local_dirs:
        image = image.add_local_dir(local, remote_path=remote)
    return image


def cpu_image() -> modal.Image:
    """The CPU image: the locked dependencies without the GPU stack."""
    image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("git")
             .uv_sync(groups=["dev"], uv_version=UV_VERSION,
                      extra_options="--no-install-package vllm"))
    return _with_sources(image)
