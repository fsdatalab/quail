"""QUAIL-B, cross-dataset entity-matching queries only (CROSS-1,
CROSS-2), sf=0.1, Qwen3-4B-FP8. First test of a join across two
different tables' worth of real product listings that were never
matched against each other by the source data (Abt-Buy is
electronics, Amazon-Google is mostly software) - no real ground
truth, unlike ABT-1/AG-1, so this only checks that the join shape
runs and reports whatever selectivity comes back.

Prediction: both queries complete without error; TRUE rate close to
0 on both, since the two catalogs barely overlap in what they sell.

Run from the quail/ directory, WITH --detach (tee to a file per house
rule):

    uv run modal run --detach tests/gpu/quailb_cross_entity.py::main \\
        2>&1 | tee results/quailb_cross_entity_4b.log

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
# to the existing milestone app, same as every other one-off cell.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results",
                                     create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
# abt/buy/amazon/google pull from a private HF dataset repo
# (ENTITY_MATCH_REPO in quailb.py) - the container needs a token.
# Read from the local hf CLI's cached login rather than a persistent
# Modal secret, so nothing sensitive is stored server-side. This
# module is re-imported inside the remote container too, where the
# local cache file does not exist - harmless there, since by that
# point the secret has already been attached to the function.
from pathlib import Path as _Path
try:
    _hf_token = _Path.home().joinpath(
        ".cache/huggingface/token").read_text().strip()
except FileNotFoundError:
    _hf_token = ""
secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})

CROSS_QUERIES = {"CROSS-1", "CROSS-2"}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol},
              secrets=[secret])
def run_cross(model: str = "qwen3-4b-fp8") -> str:
    from quail.bench.quailb import run_suite

    suite = run_suite("/results/quailb_data", sf=0.1, model=model,
                      only=CROSS_QUERIES,
                      out_path=f"/results/quailb_cross_entity_{model}.json")
    results_vol.commit()
    return str(suite)


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8"):
    fc = run_cross.spawn(model=model)
    print(f"function call id: {fc.object_id}")
    print(fc.get())
