"""Plot GPU memory after each isolated backend group exits.

Rebuild from the repository root:

    W=/tmp/quail-same-gpu-confirmation
    mkdir -p $W
    modal volume get quail-results \
      benchmarks/quailb/family-runs/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json \
      $W/manifest.json
    uv run --with matplotlib python \
      reports/make_same_gpu_benchmark_processes_plots.py $W
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import BLUE, GREEN, DARK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.workdir / "manifest.json").read_text())
    cleanup = manifest["family_run"]["families"][0]["process_cleanup"]
    labels = ["Quail", "Stock and\npipelined vLLM"]
    values = [
        item["gpu_memory_used_mib_after_exit"][0] for item in cleanup
    ]

    plt.style.use(Path(__file__).parent / "quail.mplstyle")
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(labels, values, color=[GREEN, BLUE], width=0.55)
    axis.set_ylabel("GPU memory after process exit (MiB)")
    axis.set_ylim(0, max(values) + 2)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value} MiB",
            ha="center",
            va="bottom",
            color=DARK,
        )
    figure.tight_layout()
    output = Path(__file__).parent / "plots" \
        / "same_gpu_benchmark_processes.png"
    figure.savefig(output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
