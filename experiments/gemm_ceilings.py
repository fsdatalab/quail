"""Measure H100 fp8 GEMM ceilings with stas00's mamf-finder, independent of Quail.

Runs mamf-finder.py unchanged, at the commit and on the NGC image the
published H100 SXM rows were measured with, on one H100. Nothing here
imports quail. The runs share one container, so they share one GPU:

  published_fp8  the published H100 fp8 MAMF and MSMF shapes, to check
                 this GPU against the published 1550 and 1358 TFLOPS
  qkv, o,        Qwen3 4B's four linear projections at their N and K,
  gate_up, down  sweeping M, the tokens in a chunk, from 16 to 131072

mamf-finder times torch._scaled_mm with one fp8 scale per tensor and an
fp8 output: cuBLAS, not DeepGEMM with block scales and a bf16 output.

    mkdir -p results/benchmark
    log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-gemm-ceilings.log"
    uv run modal run --detach experiments/gemm_ceilings.py 2>&1 | tee "$log"

The mamf-finder logs go under /results/ablations/gemm_ceilings/<stamp>/
and the summary to /results/ablations/gemm_ceilings_h100_fp8.json.
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

RUNS_DIR = Path("/results/ablations/gemm_ceilings")
SUMMARY_PATH = Path("/results/ablations/gemm_ceilings_h100_fp8.json")

# H100 SXM rows of the comparison table and its note 17, README of
# stas00/ml-engineering compute/accelerator at 8d028c29.
PUBLISHED_FP8 = {
    "mamf_tflops": 1550, "mamf_shape": "1280x16896x16384",
    "msmf_tflops": 1358, "msmf_shape": "1536x2816x16384",
    "msmf_watts": 674, "msmf_mhz": 1824,
    "stack": "NGC 26.04, torch 2.12.0a0, CUDA 13.2, 700 W, siblings idle",
}

# Qwen/Qwen3-4B-FP8 config.json: hidden_size 2560, 32 query heads and 8
# KV heads of head_dim 128, intermediate_size 9728. (N, K) per layer.
QWEN3_4B_GEMMS = {
    "qkv": (6144, 2560),
    "o": (2560, 4096),
    "gate_up": (19456, 2560),
    "down": (2560, 9728),
}
M_TOKENS = (16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
            4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536,
            98304, 110376, 131072)

PREDICTION_TEXT = (
    "published_fp8: MAMF within 5% of 1550 TFLOPS and MSMF within 5% of "
    "1358 TFLOPS, if Modal's H100 has a 700 W power limit; lower MSMF would "
    "point at a lower power limit or busy GPUs on the same board. Qwen3 4B "
    "GEMMs: K of 2560 to 9728 is short next to the K of 16384 at the "
    "published peaks, so each projection's best MSMF lands at 75-95% of the "
    "published 1358 TFLOPS, reached at M of 8192 or more. Below M of about "
    "400 the weight read sets the time, so TFLOPS grows about linearly with "
    "M; for example, gate_up at M = 16 cannot exceed 106 TFLOPS (49.8 MB of "
    "weights at 3.35 TB/s)."
)

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
image = (
    modal.Image.from_registry(NGC_IMAGE)
    .pip_install("nvidia-ml-py")
    .run_commands(
        f"curl -fsSL {MAMF_URL} -o {MAMF_PATH}",
        f"echo '{MAMF_SHA256}  {MAMF_PATH}' | sha256sum -c -",
    )
)

HEADLINE = re.compile(
    r"^(MAMF|MSMF) \(max .*?\):\s+(\d+) TFLOPS @ (\d+x\d+x\d+) \(MxNxK\)"
    r"(?:\s+(\d+)W)?(?: (\d+)MHz)?(?: (\d+)C)?")
SCOUT_ROW = re.compile(
    r"^\s*\d+\s+(\d+)x(\d+)x(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+"
    r"(?:\s+(\d+)\s+(\d+))?\s*$")


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


def gpu_info() -> dict:
    """Return nvidia-smi's name, power limits, clocks, and driver for GPU 0."""
    fields = ("name", "power.limit", "power.max_limit", "clocks.max.sm",
              "driver_version", "pci.bus_id")
    row = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=True).stdout.strip()
    return dict(zip(fields, (value.strip() for value in row.split(","))))


@app.function(image=image, gpu="H100!", timeout=3 * 3600,
              volumes={"/results": results_vol})
def measure(stamp: str) -> dict:
    """Run every mamf-finder configuration on one GPU and save the summary.

    Args:
        stamp: The UTC timestamp naming this run's directory.

    Returns:
        The summary also written to SUMMARY_PATH.
    """
    run_dir = RUNS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    shapes_file = run_dir / "published_fp8_shapes.txt"
    shapes_file.write_text(
        f"{PUBLISHED_FP8['mamf_shape']}\n{PUBLISHED_FP8['msmf_shape']}\n")
    runs = {"published_fp8": ["--shapes_file", str(shapes_file)]}
    for name, (n, k) in QWEN3_4B_GEMMS.items():
        runs[name] = (["--m", *map(str, M_TOKENS)]
                      + ["--n", str(n), "--k", str(k)])

    gpu = gpu_info()
    print(f"gpu: {json.dumps(gpu)}", flush=True)
    summary = {"stamp": stamp, "gpu": gpu, "image": NGC_IMAGE,
               "mamf_finder": MAMF_URL, "published_fp8": PUBLISHED_FP8,
               "prediction": PREDICTION_TEXT, "runs": {}}
    for name, shape_args in runs.items():
        log_path = run_dir / f"{name}.txt"
        argv = [sys.executable, MAMF_PATH, "--dtype", "float8_e4m3fn",
                *shape_args, "--output_file", str(log_path), "--no-verbose",
                "--notes", f"quail gemm_ceilings {stamp} {name}"]
        print(f"{name}: {' '.join(argv[1:])}", flush=True)
        subprocess.run(argv, check=True)
        parsed = parse_log(log_path.read_text())
        summary["runs"][name] = {"argv": argv[1:], "log": str(log_path),
                                 **parsed}
        print(f"{name}: MAMF {json.dumps(parsed['mamf'])} "
              f"MSMF {json.dumps(parsed['msmf'])}", flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        results_vol.commit()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    return summary


@app.local_entrypoint()
def main():
    """Start the measurement and print each configuration's headlines."""
    from datetime import datetime, timezone

    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    call = measure.spawn(stamp)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"logs: {RUNS_DIR / stamp}", flush=True)
    summary = call.get()
    print(f"gpu: {json.dumps(summary['gpu'])}", flush=True)
    for name, run in summary["runs"].items():
        print(f"{name}: MAMF {json.dumps(run['mamf'])} "
              f"MSMF {json.dumps(run['msmf'])}", flush=True)
    print(f"summary: {SUMMARY_PATH}", flush=True)
