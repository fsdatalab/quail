"""Plot the request backend confirmation results.

Pull the measured files before running this script:

    modal volume get quail-results benchmarks/quailb/runs/qb_20260902T065022Z_afca9ed1/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-quail.json $W/quail.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260902T065022Z_3bd87741/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-stock_vllm.json $W/stock_vllm.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260902T065022Z_6f846e04/20260902T065022Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_vllm.json $W/pipelined_vllm.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260902T064443Z_8fd34020/20260902T064443Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_sglang.json $W/pipelined_sglang.json

Then run:

    uv run --with matplotlib python reports/make_request_backend_confirmation_plots.py $W
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot_colors import BLUE, GRAY, GREEN, ORANGE


METHODS = (
    ("quail", "Quail", BLUE),
    ("stock_vllm", "Stock vLLM", GRAY),
    ("pipelined_vllm", "Pipelined vLLM", GREEN),
    ("pipelined_sglang", "Pipelined SGLang", ORANGE),
)


def _query_row(path: Path) -> dict:
    report = json.loads(path.read_text())
    return report["passes"]["single"]["queries"][0]


def main(workdir: Path) -> None:
    """Write the confirmation plot from downloaded result files."""
    rows = [_query_row(workdir / f"{key}.json")
            for key, _, _ in METHODS]
    labels = [label for _, label, _ in METHODS]
    colors = [color for _, _, color in METHODS]
    runtime = [float(row["wall_s"]) for row in rows]
    throughput = [
        int(row["input_document_rows"]) / float(row["wall_s"])
        for row in rows
    ]

    plt.style.use(Path(__file__).parent / "quail.mplstyle")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    y = list(range(len(METHODS)))

    for axis, values, label, fmt in (
        (axes[0], runtime, "Query time (seconds)", "{:.2f}"),
        (axes[1], throughput, "Throughput (documents/second)", "{:.1f}"),
    ):
        bars = axis.barh(y, values, color=colors)
        axis.set_xlabel(label)
        axis.set_yticks(y, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            axis.text(
                value + max(values) * 0.025,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
            )

    figure.subplots_adjust(wspace=0.35)
    output = Path(__file__).parent / "plots" / \
        "request_backend_confirmation.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} WORKDIR")
    main(Path(sys.argv[1]))
