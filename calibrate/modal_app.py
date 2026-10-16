r"""Run calibrate's GEMM measurement on Modal.

The measurement runs on the target's Modal GPU and writes to the
quail-results volume under calibrate/. --gpus N starts N containers at
once, each on its own GPU, so to_spec can take the fastest GPU:

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
    uv run modal run --detach calibrate/modal_app.py::gemm \
        --device h100-sxm --gpus 5 --prediction "..." 2>&1 | tee "$log"

Modal picks the physical GPU; each summary records its UUID, and
to_spec counts each GPU once.

The image is the NGC image the published H100 SXM rows were
measured with.
"""

from __future__ import annotations

from pathlib import Path

import modal

from calibrate import gemm as gemm_run
from calibrate import utc_stamp
from calibrate.targets import TARGETS

NGC_IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"
MAMF_DIR = Path("/opt/mamf-finder")
VOLUME_DIR = Path("/results")
ROOT = VOLUME_DIR / "calibrate"

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)

gemm_image = (
    modal.Image.from_registry(NGC_IMAGE)
    .pip_install("nvidia-ml-py")
    .run_commands(
        f"mkdir -p {MAMF_DIR}",
        f"curl -fsSL {gemm_run.MAMF_URL} -o {MAMF_DIR / gemm_run.MAMF_NAME}")
    .add_local_python_source("calibrate")
)


@app.function(image=gemm_image, gpu="H100!", timeout=3 * 3600,
              volumes={str(VOLUME_DIR): results_vol})
def measure_gemm(device: str, stamp: str, published: dict | None,
                 prediction: str) -> dict:
    """Run calibrate.gemm on this container's GPU, committing each file.

    Args:
        device: The quail.specs device name.
        stamp: Names this run's directory.
        published: The GPU's published fp8 rates and shapes, or None.
        prediction: The prediction stated before the run.
    """
    return gemm_run.run(ROOT, device, stamp, published,
                        gemm_run.fetch_mamf_finder(MAMF_DIR), prediction,
                        on_save=results_vol.commit)


@app.local_entrypoint()
def gemm(device: str = "h100-sxm", gpus: int = 1, prediction: str = ""):
    """Start the GEMM measurement on gpus GPUs and print each one's headlines.

    Args:
        device: The calibrate target, by quail.specs device name.
        gpus: Containers to start, each on its own GPU.
        prediction: The prediction stated before the run.
    """
    target = TARGETS[device]
    stamp = utc_stamp()
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    calls = {}
    for i in range(gpus):
        name = f"{stamp}-g{i}"
        calls[name] = measure_gemm.with_options(gpu=target.modal_gpu).spawn(
            device, name, target.published_fp8, prediction)
        print(f"{name}: function call id {calls[name].object_id}, "
              f"summary {ROOT / device / 'gemm' / name / 'summary.json'}", flush=True)
    for name, call in calls.items():
        summary = call.get()
        for run, result in summary["runs"].items():
            print(f"{name} {summary['gpu'].get('uuid')} {run}: "
                  f"MAMF {result['mamf']} MSMF {result['msmf']}", flush=True)
