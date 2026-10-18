"""Turn calibrate's saved GEMM summaries into a device's mamf_fp8_flops.

Pass the GEMM summaries, or directories holding them, from any number of
GPUs. For example, after a Modal run:

    modal volume get quail-results calibrate/h100-sxm .
    uv run python -m calibrate.to_spec h100-sxm

mamf_fp8_flops is the best MAMF of any mamf-finder run, so the
speed-of-light estimate stays a lower bound on every GPU of the type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(paths) -> list[dict]:
    """Return the GEMM summaries under paths.

    Args:
        paths: Summary files, or directories searched for summary.json.
    """
    gemm = []
    for path in map(Path, paths):
        files = sorted(path.rglob("summary.json")) if path.is_dir() else [path]
        for file in files:
            summary = json.loads(file.read_text())
            if "runs" in summary:
                gemm.append(summary)
    return gemm


def gpu_id(summary: dict) -> str:
    """Return the GPU's UUID, or the run's stamp when none was recorded."""
    return summary.get("gpu", {}).get("uuid") or summary["stamp"]


def best_mamf(summary: dict) -> float:
    """Return the best MAMF of one GEMM summary's runs, in TFLOPS."""
    return max((run["mamf"]["tflops"] for run in summary["runs"].values()
                if run.get("mamf")), default=0.0)


def mamf_fp8_flops(gemm: list[dict]) -> float:
    """Return the best MAMF over every GEMM summary, in FLOP/s."""
    return max(best_mamf(s) for s in gemm) * 1e12


def spec_text(gemm: list[dict]) -> str:
    """Return the DeviceSpec's mamf_fp8_flops field."""
    mamf = ", ".join(f"{best_mamf(s):.0f}" for s in gemm)
    return (f"    mamf_fp8_flops={mamf_fp8_flops(gemm):.4g},"
            f"  # best MAMF over {len({gpu_id(s) for s in gemm})} GPUs: {mamf}\n")


def main(argv=None) -> None:
    """Print mamf_fp8_flops for the summaries under the given paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path,
                        help="summary.json files or directories holding them")
    args = parser.parse_args(argv)
    print(spec_text(load(args.paths)))


if __name__ == "__main__":
    main()
