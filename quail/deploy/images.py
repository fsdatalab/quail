"""The Modal images the benchmark and experiment scripts run on.

Each image mounts the quail package and the bench package, which is
not in the wheel, so the benchmark runner imports inside a container.
"""

import importlib.util
import os
import urllib.request
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
# where CI uploads a prebuilt native module for each key of the Rust
# sources (.github/workflows/publish.yml)
NATIVE_WHEELS_URL = ("https://github.com/fsdatalab/quail/releases/download/"
                     "native-wheels")
MATURIN = "maturin>=1.9,<2"
# the native module alone, as a project: the image mounts the quail
# package from source
NATIVE_PROJECT_TEXT = """\
[build-system]
requires = ["maturin>=1.9,<2"]
build-backend = "maturin"

[project]
name = "quail-native"
version = "0.1.0"

[tool.maturin]
manifest-path = "crates/quail-python/Cargo.toml"
module-name = "_quail_native"
"""
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


def _is_rust_source(path: Path) -> bool:
    """Whether a file under crates/ is source the dependency build leaves out."""
    # a build script too: quail-python's links its tests, which an empty
    # library does not have
    return path.suffix == ".rs"


def _prebuilt_native_url() -> str | None:
    """Return the URL of the prebuilt _quail_native for this checkout's Rust.

    CI uploads one module per key of the Rust sources
    (tools/native_key.py). None when it has uploaded none for these
    sources, when the release cannot be reached, and inside a container,
    where the image is already built.
    """
    script = REPOSITORY / "tools" / "native_key.py"
    if not modal.is_local() or not script.exists():
        return None
    spec = importlib.util.spec_from_file_location("native_key", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    url = f"{NATIVE_WHEELS_URL}/quail_native-{module.native_key()}-linux-x86_64.whl"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=10):
            return url
    except OSError:
        return None


def _with_native_planner(image: modal.Image) -> modal.Image:
    """Put the _quail_native module, which plans queries in Rust, into an image.

    The image downloads the module CI built for the checkout's Rust
    sources when there is one. Otherwise it builds the module: the
    dependencies in one layer, from the manifests and the lock file
    alone, and Quail's crates in the next, so an edit to Quail's Rust
    rebuilds only Quail's crates.
    """
    url = _prebuilt_native_url()
    if url is not None:
        # the image's environment has no pip; a wheel is a zip archive
        fetch = f"import urllib.request as r; r.urlretrieve('{url}', '/tmp/native.whl')"
        unpack = ("import sysconfig, zipfile; "
                  "wheel = zipfile.ZipFile('/tmp/native.whl'); "
                  "wheel.extractall(sysconfig.get_paths()['platlib'], [name for name "
                  "in wheel.namelist() if name.startswith('_quail_native/')])")
        return image.run_commands(
            f'python -c "{fetch}"', f'python -c "{unpack}"',
            "python -c 'import _quail_native; _quail_native.LOGICAL_RULES'")
    cargo = (f"cd {NATIVE_SOURCE} && PATH=$HOME/.cargo/bin:$PATH "
             # an image builds once: Cargo's default release settings,
             # not the workspace's fast-rebuild ones
             "CARGO_PROFILE_RELEASE_INCREMENTAL=false "
             "CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 ")
    return (image.apt_install("curl", "build-essential")
            .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y "
                          "--profile minimal")
            .uv_pip_install(MATURIN, uv_version=UV_VERSION)
            .add_local_file(REPOSITORY / "Cargo.toml", f"{NATIVE_SOURCE}/Cargo.toml",
                            copy=True)
            .add_local_file(REPOSITORY / "Cargo.lock", f"{NATIVE_SOURCE}/Cargo.lock",
                            copy=True)
            .add_local_file(REPOSITORY / "rust-toolchain.toml",
                            f"{NATIVE_SOURCE}/rust-toolchain.toml", copy=True)
            .add_local_dir(REPOSITORY / "crates", f"{NATIVE_SOURCE}/crates", copy=True,
                           ignore=_is_rust_source)
            .run_commands(
                # each crate as an empty library, so Cargo builds what
                # the crates depend on and nothing of their own
                f"cd {NATIVE_SOURCE} && for crate in crates/*/; do "
                "mkdir -p $crate/src && touch $crate/src/lib.rs; done",
                cargo + "cargo build --release --locked -p quail-python")
            .add_local_dir(REPOSITORY / "crates", f"{NATIVE_SOURCE}/crates", copy=True)
            .run_commands(
                f"cat > {NATIVE_SOURCE}/pyproject.toml <<'EOF'\n"
                f"{NATIVE_PROJECT_TEXT}EOF",
                # Cargo compares file times: a copied file older than the
                # empty library's build would leave the empty one in place
                f"find {NATIVE_SOURCE}/crates -name '*.rs' -exec touch {{}} +",
                cargo + f"maturin build --release -o {NATIVE_SOURCE}/wheels",
                # the image's environment has no pip; a wheel is a zip archive
                "python -c \"import glob, sysconfig, zipfile; "
                "zipfile.ZipFile(glob.glob("
                f"'{NATIVE_SOURCE}/wheels/quail_native-*.whl')[0]).extractall("
                "sysconfig.get_paths()['platlib'])\"",
                "python -c 'import _quail_native; _quail_native.LOGICAL_RULES'"))


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
