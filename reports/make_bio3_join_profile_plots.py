r"""Rebuild the BIO-3 interactive HTML and five-second PDF with PNG preview.

    W=/tmp/quail-bio3-vllm-profile
    mkdir -p "$W"
    RUN=ablations/vllm-join-profile-20260907T013304Z
    uv run modal volume get quail-results "$RUN/result.json" "$W/result.json"
    uv run modal volume get quail-results \
      "$RUN/cpu-flamegraph.json" "$W/cpu-flamegraph.json"
    uv run modal volume get quail-results \
      "$RUN/gpu-timeline.f64.gz" "$W/gpu-timeline.f64.gz"
    uv run modal volume get quail-results "$RUN/cpu-window.json" "$W/cpu-window.json"
    uv run --with matplotlib python reports/make_bio3_join_profile_plots.py "$W"

The summary aggregates nested CPU intervals from join-0/worker.trace.json.gz
within the join annotation in join-0/driver.trace.json.gz, using
experiments.profile_flamegraph.read_cpu_flamegraph. Names are unchanged.
"""

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle
from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE

HERE = Path(__file__).resolve().parent


def color(node):
    """Return the color for a recorded operation category."""
    if node["name"].startswith("vllm.scheduler."):
        return ORANGE
    if node["name"].startswith("[") or node["name"] == "Worker main thread":
        return GRAY
    return BLUE


def load_window(workdir, timeline, source):
    """Combine the saved CPU window with GPU intervals on the same clock."""
    window = json.loads((workdir / "cpu-window.json").read_text())
    assert window["source"] == source
    window["end"] = min(window["end"], window["start"] + 5)
    start, end = window["start"], window["end"]
    window["cpu"] = [[max(start, a), min(end, b), depth, name]
                     for a, b, depth, name in window["cpu"]
                     if b > start and a < end]
    window["gpu"] = [(max(start, a / 1e6), min(end, b / 1e6))
                     for a, b in struct.iter_unpack(
                         "<dd", gzip.decompress(timeline))
                     if b / 1e6 > start and a / 1e6 < end]
    return window


def plot_window(window):
    """Save a chronological GPU and CPU view of the selected five seconds."""
    plt.style.use(HERE / "quail.mplstyle")
    start, end = window["start"], window["end"]
    levels = max(event[2] for event in window["cpu"]) + 1
    fig = plt.figure(figsize=(18, 4.9 + 0.32 * levels))
    fig.set_layout_engine("none")
    fig.text(0.09, 0.92, "BIO-3 query with vLLM",
             fontsize=18, weight="bold")
    gpu = fig.add_axes((0.09, 0.69, 0.89, 0.17))
    position = start
    vertices, fills = [], []

    def add(a, b, y, height, fill):
        if b > a:
            vertices.append([(a, y), (b, y), (b, y + height), (a, y + height)])
            fills.append(fill)

    for a, b in window["gpu"]:
        add(position, a, 1, 0.8, DARK)
        add(a, b, 0, 0.8, GREEN)
        position = b
    add(position, end, 1, 0.8, DARK)
    gpu.add_collection(
        PolyCollection(vertices, facecolors=fills, edgecolors="none"))
    gpu.set(xlim=(start, end), ylim=(1.9, -0.1), yticks=[0.4, 1.4],
            yticklabels=["GPU active", "GPU idle"],
            xticks=range(int(start), int(end) + 1))
    gpu.tick_params(axis="both", labelsize=12)
    cpu = fig.add_axes((0.09, 0.17, 0.89, 0.40), sharex=gpu)
    cpu.set_title("CPU operations", loc="left", fontsize=14)
    vertices, fills = [], []
    for a, b, depth, name in window["cpu"]:
        add(a, b, depth, 0.9, color({"name": name}))
        if b - a > (end - start) * 0.035:
            patch = Rectangle((a, depth), b - a, 0.9, transform=cpu.transData)
            label = cpu.text(a + 0.004, depth + 0.46, name, fontsize=9,
                             va="center", clip_on=True)
            label.set_clip_path(patch)
    cpu.add_collection(
        PolyCollection(vertices, facecolors=fills, edgecolors="none"))
    schedules = [event for event in window["cpu"]
                 if event[3] == "vllm.scheduler.schedule"]
    if schedules:
        a, b, depth, name = max(schedules, key=lambda event: event[1] - event[0])
        cpu.annotate(name, xy=((a + b) / 2, depth + 0.45),
                     xytext=((a + b) / 2, -0.65), ha="center", fontsize=11,
                     arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.7})
    cpu.set(xlim=(start, end), ylim=(levels, -0.1), yticks=[], xlabel="seconds")
    cpu.tick_params(axis="x", labelsize=12)
    cpu.xaxis.label.set_size(12)
    fig.legend(handles=[
        Patch(facecolor=BLUE, label="PyTorch / CUDA API calls (CPU)"),
        Patch(facecolor=ORANGE, label="vllm.scheduler.*"),
        Patch(facecolor=GRAY, label="[no recorded CPU operation]"),
    ], loc="lower left", bbox_to_anchor=(0.09, 0.015), ncol=3, fontsize=12,
        borderaxespad=0, handlelength=1.2, handleheight=1.2, columnspacing=3)
    destination = HERE / "plots/bio3_join_window"
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def write_interactive(root, timeline, window):
    """Save a standalone interactive flame graph."""
    data = json.dumps(root).replace("<", "\\u003c")
    html = Path(__file__).with_name("bio3_join_profile_template.html").read_text()
    html = html.replace("__DARK__", DARK).replace("__GREEN__", GREEN)
    html = html.replace("__BLUE__", BLUE).replace("__ORANGE__", ORANGE)
    html = html.replace("__GRAY__", GRAY)
    html = html.replace("__DATA__", data).replace("__COLORS__", json.dumps({
        "orange": ORANGE, "blue": BLUE, "gray": GRAY, "green": GREEN, "dark": DARK,
    }))
    html = html.replace("__TIMELINE_DATA__",
                        base64.b64encode(timeline).decode("ascii"))
    html = html.replace("__TIMELINE_JS__", (HERE / "gpu_timeline.js").read_text())
    html = html.replace("__CPU_WINDOW__",
                        json.dumps(window).replace("<", "\\u003c"))
    html = html.replace("__CPU_WINDOW_JS__",
                        (HERE / "cpu_gpu_window.js").read_text())
    (HERE / "plots/bio3_join_profile.html").write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    workdir = parser.parse_args().workdir
    result = json.loads((workdir / "result.json").read_text())
    summary = json.loads((workdir / "cpu-flamegraph.json").read_text())
    assert result["query"] == "BIO-3"
    assert summary["source"] == result["result_volume_path"]
    timeline = (workdir / "gpu-timeline.f64.gz").read_bytes()
    window = load_window(workdir, timeline, summary["source"])
    plot_window(window)
    write_interactive(summary["flamegraph"], timeline, window)
