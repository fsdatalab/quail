r"""Measure a GPU's fp8 GEMM ceilings with stas00's mamf-finder.

Runs mamf-finder.py unchanged, at a pinned commit, on the NGC image the
published H100 SXM rows were measured with. mamf-finder times
torch._scaled_mm with one fp8 scale per tensor and an fp8 output:
cuBLAS, not Quail's DeepGEMM kernels. The runs share one container, so
they share one GPU:

  published_fp8  the GPU's published MAMF and MSMF shapes, to check it
                 against the published rates; a GPU without published
                 rates runs mamf-finder's own shape search as "search"
  qkv, o,        a model's four projections at their N and K, sweeping
  gate_up, down  M, the tokens in a chunk, from 16 to 131072; --smoke
                 skips these

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
    uv run modal run --detach calibrate/gemm.py --device h100-sxm \
        --model qwen3-4b-fp8 --prediction "..." 2>&1 | tee "$log"

The mamf-finder logs go under /results/calibrate/<device>/gemm/<stamp>/
and the summary to /results/calibrate/<device>/gemm.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import modal

MAMF_COMMIT = "30ddd97054eb01035c0a5079a58f011df603b525"
MAMF_URL = ("https://raw.githubusercontent.com/stas00/ml-engineering/"
            f"{MAMF_COMMIT}/compute/accelerator/benchmarks/mamf-finder.py")
MAMF_SHA256 = "24239a3cba82746cabb1cf756afbd4467ed1a1b884eafd842b67cbf948ef7256"
MAMF_PATH = "/opt/mamf-finder.py"
NGC_IMAGE = "nvcr.io/nvidia/pytorch:26.04-py3"
RESULTS = Path("/results/calibrate")

M_TOKENS = (16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
            4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
            98304, 110376, 131072)

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.from_registry(NGC_IMAGE)
    .pip_install("nvidia-ml-py")
    .run_commands(
        f"curl -fsSL {MAMF_URL} -o {MAMF_PATH}",
        f"echo '{MAMF_SHA256}  {MAMF_PATH}' | sha256sum -c -",
    )
    .add_local_python_source("calibrate")
)

HEADLINE = re.compile(
    r"^(MAMF|MSMF) \(max .*?\):\s+(\d+) TFLOPS @ (\d+x\d+x\d+) \(MxNxK\)"
    r"(?:\s+(\d+)W)?(?: (\d+)MHz)?(?: (\d+)C)?")
SCOUT_ROW = re.compile(
    r"^\s*\d+\s+(\d+)x(\d+)x(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+"
    r"(?:\s+(\d+)\s+(\d+))?(?:\s+<-.*)?\s*$")


def parse_log(text: str) -> dict:
    """Return the MAMF and MSMF headlines and the scout rows of a mamf-finder log.

    Args:
        text: The log mamf-finder wrote with --output_file.

    Returns:
        A dict with "mamf" and "msmf" (None when missing) and "scouts", one
        entry per shape in the order measured.
    """
    out = {"mamf": None, "msmf": None, "scouts": []}
    seen = set()
    for line in text.splitlines():
        if match := HEADLINE.match(line):
            name, tflops, shape, watts, mhz, temp = match.groups()
            out[name.lower()] = {
                "tflops": int(tflops), "shape": shape,
                "watts": watts and int(watts), "mhz": mhz and int(mhz),
                "celsius": temp and int(temp)}
        elif match := SCOUT_ROW.match(line):
            m, n, k, mean, median, best, watts, mhz = match.groups()
            if (m, n, k) in seen:
                continue
            seen.add((m, n, k))
            out["scouts"].append({
                "m": int(m), "n": int(n), "k": int(k),
                "mean_tflops": float(mean), "median_tflops": float(median),
                "max_tflops": float(best),
                "watts": watts and int(watts), "max_mhz": mhz and int(mhz)})
    return out


def mamf_runs(published: dict | None, shapes: dict, run_dir: Path) -> dict:
    """Return mamf-finder's shape arguments for each run, by run name.

    Args:
        published: The GPU's published fp8 rates and shapes, or None.
        shapes: Projection name -> (N, K).
        run_dir: Where the published shapes file is written.
    """
    if published is None:
        runs = {"search": ["--search", "auto"]}
    else:
        shapes_file = run_dir / "published_fp8_shapes.txt"
        shapes_file.write_text(
            f"{published['mamf_shape']}\n{published['msmf_shape']}\n")
        runs = {"published_fp8": ["--shapes_file", str(shapes_file)]}
    for name, (n, k) in shapes.items():
        runs[name] = ["--m", *map(str, M_TOKENS), "--n", str(n), "--k", str(k)]
    return runs


def gpu_info() -> dict:
    """Return nvidia-smi's name, power limits, clocks, and driver for GPU 0."""
    fields = ("name", "power.limit", "power.max_limit", "clocks.max.sm",
              "driver_version")
    row = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=True).stdout.strip()
    return dict(zip(fields, (value.strip() for value in row.split(","))))


@app.function(image=image, gpu="H100!", timeout=3 * 3600,
              volumes={"/results": results_vol})
def measure(stamp: str, device: str, model: str, published: dict | None,
            shapes: dict, smoke: bool, prediction: str) -> dict:
    """Run every mamf-finder configuration on one GPU and save the summary.

    Args:
        stamp: The UTC timestamp naming this run's directory.
        device: The quail.specs device name.
        model: The quail.specs model name the shapes are from.
        published: The GPU's published fp8 rates and shapes, or None.
        shapes: Projection name -> (N, K).
        smoke: Time only the published shapes or the search; the
            summary stays in the run's directory.
        prediction: The prediction stated before the run.

    Returns:
        The summary, also written to /results/calibrate/<device>/gemm.json
        unless smoke.
    """
    run_dir = RESULTS / device / "gemm" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    print(f"gpu: {json.dumps(gpu)}", flush=True)
    summary = {"stamp": stamp, "device": device, "model": model, "gpu": gpu,
               "image": NGC_IMAGE, "mamf_finder": MAMF_URL,
               "published_fp8": published, "smoke": smoke,
               "prediction": prediction, "runs": {}}
    runs = mamf_runs(published, {} if smoke else shapes, run_dir)
    for name, shape_args in runs.items():
        log_path = run_dir / f"{name}.txt"
        argv = [sys.executable, MAMF_PATH, "--dtype", "float8_e4m3fn",
                *shape_args, "--output_file", str(log_path), "--no-verbose",
                "--notes", f"quail calibrate {stamp} {name}"]
        print(f"{name}: {' '.join(argv[1:])}", flush=True)
        subprocess.run(argv, check=True)
        parsed = parse_log(log_path.read_text())
        summary["runs"][name] = {"argv": argv[1:], "log": str(log_path),
                                 **parsed}
        print(f"{name}: MAMF {json.dumps(parsed['mamf'])} "
              f"MSMF {json.dumps(parsed['msmf'])}", flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        results_vol.commit()
    if not smoke:
        (RESULTS / device / "gemm.json").write_text(json.dumps(summary, indent=2))
        results_vol.commit()
    return summary


@app.local_entrypoint()
def main(device: str = "h100-sxm", model: str = "qwen3-4b-fp8",
         smoke: bool = False, prediction: str = ""):
    """Start the measurement on the device's GPU and print each run's headlines."""
    from datetime import datetime, timezone

    from calibrate.targets import TARGETS, gemm_shapes
    from quail.specs import MODELS

    target = TARGETS[device]
    shapes = gemm_shapes(MODELS[model])
    print(f"prediction: {prediction or '(none stated)'}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.with_options(gpu=target.modal_gpu).spawn(
        stamp, device, model, target.published_fp8, shapes, smoke, prediction)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"logs: {RESULTS / device / 'gemm' / stamp}", flush=True)
    summary = call.get()
    for name, run in summary["runs"].items():
        print(f"{name}: MAMF {json.dumps(run['mamf'])} "
              f"MSMF {json.dumps(run['msmf'])}", flush=True)
    if not smoke:
        print(f"summary: {RESULTS / device / 'gemm.json'}", flush=True)
