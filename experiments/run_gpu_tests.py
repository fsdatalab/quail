"""Run the GPU tests under tests/gpu on Modal.

    uv run modal run experiments/run_gpu_tests.py 2>&1 | tee results/gpu-tests.log

Prints pytest's output and fails when a test fails.
"""

import modal

from quail.deploy.images import gpu_image

# Liquid's d1 code needs a newer transformers than the project pins; the
# d1 test's reference process reads it from this folder alone
D1_REFERENCE_PACKAGES = "/opt/d1-reference"
image = gpu_image(
    ("tests/gpu", "/root/tests/gpu"),
    commands=(f"/usr/local/bin/python3 -m pip install --quiet --target "
              f"{D1_REFERENCE_PACKAGES} "
              f"transformers==5.19.0",))

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
