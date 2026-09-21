r"""Rebuild the blog profile figures from saved summaries.

    W=/tmp/quail-blog-profiles
    Q=ablations/blog-profiles-20260907T211904Z/quail
    A=ablations/vllm-join-profile-20260907T214358Z
    mkdir -p "$W/quail/AGENT-1" "$W/pipelined_vllm/AGENT-1"
    uv run modal volume get quail-results \
      "$Q/AGENT-1/analysis.json" "$W/quail/AGENT-1/analysis.json"
    uv run modal volume get quail-results \
      "$A/blog-analysis.json" "$W/pipelined_vllm/AGENT-1/analysis.json"
    uv run python blogs/intro/make_profile_plots.py "$W"

The CPU row includes only operations at nesting depth zero.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
METHODS = (("quail", "Quail"), ("pipelined_vllm", "vLLM baseline"))
BLUE = "#4C72B0"
GRAY = "#BDBDBD"
GREEN = "#55A868"
ORANGE = "#DD8452"


def operation_color(name):
    """Return the color for one top-level CPU operation."""
    if name.startswith("["):
        return GRAY
    if name.startswith(("vllm.scheduler.", "quail.executor.")):
        return ORANGE
    return BLUE


def add_rectangles(axis, spans):
    """Draw interval rows."""
    vertices = []
    colors = []
    for left, right, row, fill in spans:
        if right <= left:
            continue
        vertices.append(
            [(left, row), (right, row), (right, row + 0.72), (left, row + 0.72)]
        )
        colors.append(fill)
    axis.add_collection(
        PolyCollection(vertices, facecolors=colors, edgecolors="none")
    )


def plot(query, profiles):
    """Save one five-second comparison as PDF and PNG."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), sharey=True)
    phase = profiles[0]["phase"]["phase"]
    fig.suptitle(f"{query} {phase}", fontsize=23, weight="bold", y=0.98)

    for axis, ((_, label), profile) in zip(axes, zip(METHODS, profiles)):
        window = profile["window"]
        offset = window["start"]
        duration = window["end"] - offset

        gpu_spans = [
            (left - offset, right - offset, 1, GREEN)
            for left, right in window["gpu"]
        ]
        cpu_events = [
            (left - offset, right - offset, name)
            for left, right, depth, name in window["cpu"]
            if depth == 0
        ]
        cpu_spans = [
            (left, right, 0, operation_color(name))
            for left, right, name in cpu_events
        ]
        add_rectangles(axis, gpu_spans + cpu_spans)

        axis.set_title(label, loc="left", pad=12, weight="bold")
        axis.set_xlim(0, duration)
        axis.set_ylim(-0.18, 1.9)
        axis.set_xticks(range(6))
        axis.set_yticks([1.36, 0.36])
        axis.set_yticklabels(["GPU active", "CPU operations"])
        axis.set_xlabel("seconds")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0, pad=10)
        axis.grid(False)

    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="GPU active"),
            Patch(facecolor=ORANGE, label="Quail executor or vLLM scheduler"),
            Patch(facecolor=BLUE, label="PyTorch or CUDA API call"),
            Patch(facecolor=GRAY, label="No recorded CPU operation"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.79, bottom=0.25, wspace=0.25)
    stem = HERE / "figures" / f"{query.lower().replace('-', '')}_profile_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(workdir):
    """Build the AGENT-1 profile comparison."""
    query = "AGENT-1"
    profiles = [
        json.loads((workdir / method / query / "analysis.json").read_text())
        for method, _ in METHODS
    ]
    plot(query, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    main(parser.parse_args().workdir)
