"""Build BIO-4 join profile figures for the launch blog.

Uses the blog Inter theme (static instances). Omits synthetic
"[no recorded CPU operation]" fillers so GPU-idle gaps stay white.
Does not put window GPU-percent jargon in titles.

    python make_bio4_profile_figures.py /path/to/bio4-plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

INK = "#101828"
BLUE = "#4C72B0"
GREEN = "#55A868"
ORANGE = "#DD8452"
HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
if not FONT_DIR.exists():
    FONT_DIR = Path("/workspace/quail-blog-data/out/fonts")


def _register_fonts():
    for path in sorted(FONT_DIR.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(path))
    regular = font_manager.FontProperties(fname=str(FONT_DIR / "Inter-Regular.ttf"))
    semibold = font_manager.FontProperties(fname=str(FONT_DIR / "Inter-SemiBold.ttf"))
    bold = font_manager.FontProperties(fname=str(FONT_DIR / "Inter-Bold.ttf"))
    return regular, semibold, bold


def _load(path: Path):
    return json.loads(path.read_text())


def _operation_color(name: str) -> str:
    if name.startswith(("vllm.scheduler.", "quail.executor.")):
        return ORANGE
    return BLUE


def _add_rectangles(axis, spans):
    vertices, colors = [], []
    for left, right, row, fill in spans:
        if right <= left:
            continue
        vertices.append(
            [
                (left, row),
                (right, row),
                (right, row + 0.72),
                (left, row + 0.72),
            ]
        )
        colors.append(fill)
    if vertices:
        axis.add_collection(
            PolyCollection(vertices, facecolors=colors, edgecolors="none")
        )


def _theme(regular) -> None:
    plt.rcParams.update(
        {
            "font.family": regular.get_name(),
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel(axis, profile, title: str, semibold) -> None:
    window = profile["window"]
    offset = window["start"]
    duration = window["end"] - offset
    gpu_spans = [
        (left - offset, right - offset, 1, GREEN)
        for left, right in window["gpu"]
    ]
    cpu_spans = [
        (left - offset, right - offset, 0, _operation_color(name))
        for left, right, depth, name in window["cpu"]
        if depth == 0 and not name.startswith("[")
    ]
    _add_rectangles(axis, gpu_spans + cpu_spans)
    axis.set_title(title, loc="left", pad=12, color=INK, fontproperties=semibold)
    axis.set_xlim(0, duration)
    axis.set_ylim(-0.18, 1.9)
    axis.set_xticks(range(6))
    axis.set_yticks([1.36, 0.36])
    axis.set_yticklabels(["GPU active", "CPU operations"])
    axis.set_xlabel("seconds")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.grid(False)


def _legend_handles():
    return [
        Patch(facecolor=GREEN, label="GPU active"),
        Patch(facecolor=ORANGE, label="Quail executor or vLLM scheduler"),
        Patch(facecolor=BLUE, label="PyTorch or CUDA API call"),
    ]


def build(workdir: Path, out_dir: Path) -> None:
    regular, semibold, bold = _register_fonts()
    quail = _load(workdir / "quail" / "BIO-4" / "analysis.json")
    vllm = _load(workdir / "pipelined_vllm" / "BIO-4" / "analysis.json")
    phase = quail["phase"]["phase"]

    _theme(regular)
    fig, axis = plt.subplots(figsize=(10, 4.2))
    fig.suptitle(
        f"BIO-4 {phase}, vLLM baseline",
        fontsize=20,
        y=0.98,
        color=INK,
        fontproperties=bold,
    )
    _panel(axis, vllm, "vLLM baseline", semibold)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
        prop=regular,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.28)
    stem = out_dir / "bio4_vllm_bubbles"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    _theme(regular)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), sharey=True)
    fig.suptitle(
        f"BIO-4 {phase}",
        fontsize=22,
        y=0.98,
        color=INK,
        fontproperties=bold,
    )
    _panel(axes[0], quail, "Quail", semibold)
    _panel(axes[1], vllm, "vLLM baseline", semibold)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
        prop=regular,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.79, bottom=0.25, wspace=0.25)
    stem = out_dir / "bio4_profile_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    build(args.workdir, args.out)
    print(f"wrote figures under {args.out}")


if __name__ == "__main__":
    main()
