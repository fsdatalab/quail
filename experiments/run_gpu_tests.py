"""Run the GPU tests under tests/gpu on Modal.

    uv run modal run experiments/run_gpu_tests.py 2>&1 | tee results/gpu-tests.log

Prints pytest's output and fails when a test fails.
"""

import hashlib
import os
import uuid
from pathlib import Path

import modal

import quail
from quail.deploy.images import gpu_image

image = gpu_image(("tests/gpu", "/root/tests/gpu"))

app = modal.App("quail-milestone1")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
volumes = {
    "/results": results_volume,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}


def source_digest() -> str:
    """Hash the Quail Python sources imported by this process."""
    root = Path(quail.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=volumes)
def run_tests(keyword: str = "") -> int:
    """Run pytest over tests/gpu and return its exit code."""
    import pytest

    destination = Path("/results/gpu-tests") / uuid.uuid4().hex
    destination.mkdir(parents=True)
    os.environ["QUAIL_TEST_RESULTS"] = str(destination)
    print(f"remote source sha256: {source_digest()}", flush=True)
    print(f"result volume path: {destination}", flush=True)
    args = ["/root/tests/gpu", "-x", "-q", "-s", "-p", "no:cacheprovider",
            f"--junitxml={destination / 'pytest.xml'}"]
    if keyword:
        args += ["-k", keyword]
    try:
        return int(pytest.main(args))
    finally:
        results_volume.commit()


@app.local_entrypoint()
def main(keyword: str = "", gpus: int = 1):
    print(f"local source sha256: {source_digest()}", flush=True)
    call = run_tests.with_options(gpu=f"H100!:{gpus}").spawn(keyword)
    print(f"function call id: {call.object_id}", flush=True)
    code = call.get()
    print(f"pytest exit code: {code}", flush=True)
    if code:
        raise SystemExit(code)
