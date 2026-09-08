r"""Rebuild the blog's H100 profile comparisons from saved summaries.

    W=/tmp/quail-blog-profiles
    Q=ablations/blog-profiles-20260907T211904Z/quail
    B=ablations/vllm-join-profile-20260907T013304Z
    A=ablations/vllm-join-profile-20260907T214358Z
    for METHOD in quail pipelined_vllm; do
      for QUERY in BIO-3 AGENT-1; do
        mkdir -p "$W/$METHOD/$QUERY"
      done
    done
    uv run modal volume get quail-results \
      "$Q/BIO-3/analysis.json" "$W/quail/BIO-3/analysis.json"
    uv run modal volume get quail-results \
      "$Q/AGENT-1/analysis.json" "$W/quail/AGENT-1/analysis.json"
    uv run modal volume get quail-results \
      "$B/blog-analysis.json" "$W/pipelined_vllm/BIO-3/analysis.json"
    uv run modal volume get quail-results \
      "$A/blog-analysis.json" "$W/pipelined_vllm/AGENT-1/analysis.json"
    uv run python reports/make_blog_profile_plots.py "$W"

New summaries are derived by experiments/analyze_blog_profiles.py. Each new
window starts at floor(phase duration / 2 - 2.5) seconds and lasts five seconds.
BIO-3 vLLM reuses the existing 480 to 485 second window and saved trace data.
The GPU rows show the exact union of recorded kernels, copies, and memsets.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle
from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE

HERE = Path(__file__).resolve().parent
METHODS = (("quail", "Quail"), ("pipelined_vllm", "Pipelined vLLM"))


def color(name):
    """Distinguish engine functions from PyTorch and CUDA API calls."""
    if name.startswith("["):
        return GRAY
    if name.startswith(("vllm.scheduler.", "quail.executor.")):
        return ORANGE
    return BLUE


def rectangles(axis, spans):
    """Draw time intervals without a minimum visible width."""
    vertices, colors = [], []
    for left, right, depth, fill in spans:
        if right <= left:
            continue
        vertices.append([(left, depth), (right, depth),
                         (right, depth + 0.85), (left, depth + 0.85)])
        colors.append(fill)
    axis.add_collection(PolyCollection(vertices, facecolors=colors, edgecolors="none"))


def plot(query, profiles):
    """Save matching five-second GPU and CPU views as vector PDF and PNG."""
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams["pdf.fonttype"] = 42
    levels = max(max(event[2] for event in row["window"]["cpu"])
                 for row in profiles) + 1
    fig = plt.figure(figsize=(18, 5.7 + 0.25 * levels))
    fig.set_layout_engine("none")
    phase = profiles[0]["phase"]["phase"]
    fig.suptitle(f"{query} {phase}", fontsize=21, weight="bold", y=0.98)
    for index, ((_, label), row) in enumerate(zip(METHODS, profiles)):
        window = row["window"]
        offset = window["start"]
        duration = window["end"] - offset
        left = 0.07 + index * 0.5
        gpu = fig.add_axes((left, 0.69, 0.40, 0.17))
        gpu.set_title(label, loc="left", fontsize=17, pad=18)
        spans, position = [], 0
        for a, b in window["gpu"]:
            a, b = a - offset, b - offset
            spans.extend([(position, a, 1, DARK), (a, b, 0, GREEN)])
            position = b
        spans.append((position, duration, 1, DARK))
        rectangles(gpu, spans)
        gpu.set(xlim=(0, duration), ylim=(1.95, -0.1), yticks=[0.425, 1.425],
                yticklabels=["GPU active", "GPU idle"], xticks=range(6))
        gpu.tick_params(labelsize=12)
        cpu = fig.add_axes((left, 0.17, 0.40, 0.40), sharex=gpu)
        cpu.set_title("CPU operations", loc="left", fontsize=13, pad=10)
        spans = []
        for a, b, depth, name in window["cpu"]:
            a, b = a - offset, b - offset
            spans.append((a, b, depth, color(name)))
            if b - a > 0.45:
                patch = Rectangle((a, depth), b - a, 0.85, transform=cpu.transData)
                text = cpu.text(a + 0.01, depth + 0.43, name, fontsize=8,
                                va="center", clip_on=True)
                text.set_clip_path(patch)
        rectangles(cpu, spans)
        names = (["quail.executor.loop.pack_chunk",
                  "quail.executor.attention.Pipeline.forward_chunk",
                  "cudaEventSynchronize"] if index == 0 else
                 ["vllm.scheduler.schedule", "vllm.scheduler.update_from_output"])
        for line, name in enumerate(names):
            matches = [event for event in window["cpu"] if event[3] == name]
            if not matches:
                continue
            a, b, depth, _ = max(matches, key=lambda event: event[1] - event[0])
            cpu.annotate(
                name, xy=((a + b) / 2 - offset, depth + 0.425),
                xytext=(0.01, -2.75 + line * 0.85),
                textcoords=("axes fraction", "data"), fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": color(name), "lw": 0.7},
            )
        cpu.set(xlim=(0, duration), ylim=(levels, -3.3), yticks=[], xlabel="seconds")
        cpu.tick_params(labelsize=12)
        cpu.xaxis.label.set_size(12)
    fig.legend(handles=[
        Patch(facecolor=ORANGE, label="Quail executor / vLLM scheduler (CPU)"),
        Patch(facecolor=BLUE, label="PyTorch / CUDA API calls (CPU)"),
        Patch(facecolor=GRAY, label="[no recorded CPU operation]"),
    ], loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    stem = HERE / "plots" / f"{query.lower().replace('-', '')}_profile_comparison"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main(workdir):
    """Build the BIO-3 and AGENT-1 comparisons."""
    for query in ("BIO-3", "AGENT-1"):
        profiles = [json.loads((workdir / method / query / "analysis.json").read_text())
                    for method, _ in METHODS]
        plot(query, profiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    main(parser.parse_args().workdir)
