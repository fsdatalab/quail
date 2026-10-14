r"""Run calibrate's GEMM and attention measurements on Modal.

Each measurement runs on the target's Modal GPU and writes to the
quail-results volume under calibrate/. --gpus N starts N containers at
once, each on its own GPU, so to_spec can take the fastest GPU:

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
    uv run modal run --detach calibrate/modal_app.py::gemm \
        --device h100-sxm --gpus 5 --prediction "..." 2>&1 | tee "$log"
    uv run modal run --detach calibrate/modal_app.py::attention \
        --device h100-sxm --heads all --gpus 3 --prediction "..." 2>&1 | tee "$log"

--heads all measures every head layout calibrate.targets.head_layouts()
lists; "32,8,128;64,8,128" names layouts. --smoke times one point of
each kind briefly. Modal picks the physical GPU; each summary records
its UUID, and to_spec counts each GPU once.

The GEMM image is the NGC image the published H100 SXM rows were
measured with.
"""

from __future__ import annotations

from pathlib import Path

import modal

from calibrate import attention as attention_run
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
attention_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04",
                              add_python="3.12")
    .pip_install("torch==2.13.0",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("flash-attn-3==3.0.0", extra_options="--no-deps",
                 index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("flashinfer-python==0.7.1", "nvidia-ml-py", "numpy")
    .add_local_python_source("calibrate")
)


@app.function(image=gemm_image, gpu="H100!", timeout=3 * 3600,
              volumes={str(VOLUME_DIR): results_vol})
def measure_gemm(device: str, stamp: str, published: dict | None,
                 projections: dict | None, prediction: str) -> dict:
    """Run calibrate.gemm on this container's GPU, committing each file.

    Args:
        device: The quail.specs device name.
        stamp: Names this run's directory.
        published: The GPU's published fp8 rates and shapes, or None.
        projections: A ProjectionSweep's fields, or None.
        prediction: The prediction stated before the run.
    """
    sweep = projections and gemm_run.ProjectionSweep(**projections)
    return gemm_run.run(ROOT, device, stamp, published, sweep,
                        gemm_run.fetch_mamf_finder(MAMF_DIR), prediction,
                        on_save=results_vol.commit)


@app.function(image=attention_image, gpu="H100!", timeout=4 * 3600,
              volumes={str(VOLUME_DIR): results_vol})
def measure_attention(device: str, stamp: str, heads: tuple, sweep: dict,
                      check: bool, prediction: str) -> dict:
    """Run calibrate.attention on this container's GPU, committing each file.

    Args:
        device: The quail.specs device name.
        stamp: Names this run's directory.
        heads: (query heads, KV heads, head dimension).
        sweep: An AttentionSweep's fields.
        check: Compare each library's output with torch's.
        prediction: The prediction stated before the run.
    """
    return attention_run.run(ROOT, device, stamp, heads,
                             attention_run.AttentionSweep.from_dict(sweep),
                             check, prediction, on_save=results_vol.commit)


@app.local_entrypoint()
def gemm(device: str = "h100-sxm", gpus: int = 1, model: str = "",
         prediction: str = ""):
    """Start the GEMM measurement on gpus GPUs and print each one's headlines.

    Args:
        device: The calibrate target, by quail.specs device name.
        gpus: Containers to start, each on its own GPU.
        model: A quail.specs model name whose projections to sweep, or "".
        prediction: The prediction stated before the run.
    """
    from dataclasses import asdict

    target = TARGETS[device]
    projections = model and asdict(gemm_run.projection_sweep(model))
    stamp = utc_stamp()
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    calls = {}
    for i in range(gpus):
        name = f"{stamp}-g{i}"
        calls[name] = measure_gemm.with_options(gpu=target.modal_gpu).spawn(
            device, name, target.published_fp8, projections or None, prediction)
        print(f"{name}: function call id {calls[name].object_id}, "
              f"summary {ROOT / device / 'gemm' / name / 'summary.json'}", flush=True)
    for name, call in calls.items():
        summary = call.get()
        for run, result in summary["runs"].items():
            print(f"{name} {summary['gpu'].get('uuid')} {run}: "
                  f"MAMF {result['mamf']} MSMF {result['msmf']}", flush=True)


@app.local_entrypoint()
def attention(heads: str = "all", device: str = "h100-sxm", gpus: int = 1,
              smoke: bool = False, prediction: str = ""):
    """Start the attention measurement for each layout on gpus GPUs.

    Args:
        heads: "all", or layouts such as "32,8,128;64,8,128".
        device: The calibrate target, by quail.specs device name.
        gpus: Containers to start per layout, each on its own GPU.
        smoke: Measure one point of each kind briefly and check correctness.
        prediction: The prediction stated before the run.
    """
    from dataclasses import asdict

    from calibrate.targets import call_sizes, head_layouts

    target = TARGETS[device]
    layouts = (head_layouts() if heads == "all"
               else [attention_run.parse_heads(h) for h in heads.split(";")])
    stamp = utc_stamp()
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    calls = {}
    for layout in layouts:
        chunk, kv = call_sizes(layout, device)
        sweep = (attention_run.AttentionSweep.smoke(chunk, kv) if smoke
                 else attention_run.AttentionSweep(chunk, kv))
        for i in range(gpus):
            name = f"{stamp}-{'_'.join(map(str, layout))}-g{i}"
            calls[name] = measure_attention.with_options(
                gpu=target.modal_gpu,
                env={"FLASHINFER_CUDA_ARCH_LIST": target.flashinfer_arch},
            ).spawn(device, name, layout, asdict(sweep), smoke, prediction)
            print(f"{name}: chunk {chunk}, KV {kv} tokens, function call id "
                  f"{calls[name].object_id}, summary "
                  f"{ROOT / device / 'attention' / name / 'summary.json'}", flush=True)
    for name, call in calls.items():
        summary = call.get()
        print(f"{name} {summary['gpu'].get('uuid')}", flush=True)
        for row in summary["points"]:
            print(f"  {attention_run.point_line(row)}", flush=True)
