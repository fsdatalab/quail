r"""Measure a GPU's fp8 GEMM ceilings with stas00's mamf-finder.

Runs mamf-finder.py unchanged, at a pinned commit. mamf-finder times
torch._scaled_mm with one fp8 scale per tensor and an fp8 output:
cuBLAS, not Quail's DeepGEMM kernels. The published H100 SXM rows were
measured on the NGC image calibrate.modal_app uses. The runs share one
process's GPU:

  published_fp8  the GPU's published MAMF and MSMF shapes, to check it
                 against the published rates; a GPU without published
                 rates runs mamf-finder's own shape search as "search"
  qkv, o,        with --model only: the model's four projections at
  gate_up, down  their N and K, sweeping M, the tokens in a chunk

On a machine with the GPU:

    uv run python -m calibrate.gemm --device h100-sxm --root results/calibrate

The mamf-finder logs go under <root>/<device>/gemm/<stamp>/ and the
summary to <root>/<device>/gemm.json. calibrate.modal_app runs the same
code on Modal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

MAMF_COMMIT = "30ddd97054eb01035c0a5079a58f011df603b525"
MAMF_URL = ("https://raw.githubusercontent.com/stas00/ml-engineering/"
            f"{MAMF_COMMIT}/compute/accelerator/benchmarks/mamf-finder.py")
MAMF_SHA256 = "24239a3cba82746cabb1cf756afbd4467ed1a1b884eafd842b67cbf948ef7256"
MAMF_NAME = f"mamf-finder-{MAMF_COMMIT[:8]}.py"

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


@dataclass(frozen=True)
class ProjectionSweep:
    """A model's dense projections, timed across chunk sizes.

    Attributes:
        model: The quail.specs model name the shapes are from.
        shapes: Projection name -> (N, K).
        m_tokens: The chunk sizes M, in tokens.
    """

    model: str
    shapes: dict[str, tuple[int, int]]
    m_tokens: tuple[int, ...] = (16, 32, 64, 128, 256, 384, 512, 768, 1024,
                                 1536, 2048, 3072, 4096, 6144, 8192, 12288,
                                 16384, 24576, 32768, 49152, 65536, 98304,
                                 131072)


def mamf_runs(published: dict | None, projections: ProjectionSweep | None,
              run_dir: Path) -> dict:
    """Return mamf-finder's shape arguments for each run, by run name.

    Args:
        published: The GPU's published fp8 rates and shapes, or None.
        projections: A model's projections to sweep, or None.
        run_dir: Where the published shapes file is written.
    """
    if published is None:
        runs = {"search": ["--search", "auto"]}
    else:
        shapes_file = run_dir / "published_fp8_shapes.txt"
        shapes_file.write_text(
            f"{published['mamf_shape']}\n{published['msmf_shape']}\n")
        runs = {"published_fp8": ["--shapes_file", str(shapes_file)]}
    if projections is not None:
        m_args = ["--m", *map(str, projections.m_tokens)]
        for name, (n, k) in projections.shapes.items():
            runs[name] = [*m_args, "--n", str(n), "--k", str(k)]
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


def fetch_mamf_finder(directory: Path) -> Path:
    """Return the pinned mamf-finder.py in directory, downloading it if absent.

    Args:
        directory: Where to keep the script.

    Raises:
        ValueError: If the file's sha256 is not the pinned one.
    """
    path = directory / MAMF_NAME
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MAMF_URL) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MAMF_SHA256:
        raise ValueError(f"{path} has sha256 {digest}, expected {MAMF_SHA256}")
    return path


def run(root: Path, device: str, stamp: str, published: dict | None,
        projections: ProjectionSweep | None, mamf_finder: Path,
        prediction: str = "", on_save: Callable[[], None] = lambda: None) -> dict:
    """Run every mamf-finder configuration on GPU 0 and save the summary.

    Args:
        root: The results directory.
        device: The quail.specs device name.
        stamp: The UTC timestamp naming this run's directory.
        published: The GPU's published fp8 rates and shapes, or None.
        projections: A model's projections to sweep, or None.
        mamf_finder: The pinned mamf-finder.py.
        prediction: The prediction stated before the run.
        on_save: Called after each file is written.

    Returns:
        The summary, also written to <root>/<device>/gemm.json.
    """
    run_dir = root / device / "gemm" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    print(f"gpu: {json.dumps(gpu)}", flush=True)
    summary = {"stamp": stamp, "device": device, "gpu": gpu,
               "mamf_finder": MAMF_URL, "published_fp8": published,
               "projections": projections and asdict(projections),
               "prediction": prediction, "runs": {}}
    for name, shape_args in mamf_runs(published, projections, run_dir).items():
        log_path = run_dir / f"{name}.txt"
        argv = [sys.executable, str(mamf_finder), "--dtype", "float8_e4m3fn",
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
        on_save()
    (root / device / "gemm.json").write_text(json.dumps(summary, indent=2))
    on_save()
    return summary


def projection_sweep(model: str) -> ProjectionSweep:
    """Return the projection sweep of a quail.specs model."""
    from calibrate.targets import gemm_shapes
    from quail.specs import MODELS

    return ProjectionSweep(model, gemm_shapes(MODELS[model]))


def main(argv=None) -> None:
    """Measure the GPU this process sees and print where the summary went."""
    from calibrate import utc_stamp
    from calibrate.targets import TARGETS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True,
                        help="quail.specs device name, such as h100-sxm")
    parser.add_argument("--root", type=Path, default=Path("results/calibrate"),
                        help="results directory")
    parser.add_argument("--model", default="",
                        help="quail.specs model whose projections to sweep")
    parser.add_argument("--mamf-finder", type=Path,
                        help="pinned mamf-finder.py; downloaded into --root "
                             "when omitted")
    parser.add_argument("--prediction", default="")
    args = parser.parse_args(argv)
    target = TARGETS.get(args.device)
    run(args.root, args.device, utc_stamp(),
        target and target.published_fp8,
        projection_sweep(args.model) if args.model else None,
        args.mamf_finder or fetch_mamf_finder(args.root), args.prediction)
    print(f"summary: {args.root / args.device / 'gemm.json'}", flush=True)


if __name__ == "__main__":
    main()
