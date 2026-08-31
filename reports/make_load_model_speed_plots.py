"""Figure for reports/2026-08-31-load-model-speed.md.

Draws load_model_speed.png: one stacked horizontal bar per
configuration, each segment one phase of the cold load, totals and
the saving annotated.

Usage:

    uv run --with matplotlib python reports/make_load_model_speed_plots.py $W

where $W holds the files pulled from the quail-results volume:

    modal volume get quail-results ablations/load_model_profile_main.json $W/
    modal volume get quail-results ablations/load_model_profile_pinned_seed.json $W/
    modal volume get quail-results ablations/load_model_profile_pinned.json $W/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, GRAY, GREEN, ORANGE  # noqa: E402

plt.style.use(Path(__file__).parent / "quail.mplstyle")

OUT = Path(__file__).parent / "plots"

PHASES = [
    ("import_torch_s", "import torch", GRAY),
    ("import_vllm_s", "import vllm", BLUE),
    ("engine_config_s", "engine config", ORANGE),
    ("get_model_s", "get_model", GREEN),
]

ROWS = [
    ("main", "before:\ndefault branch,\nephemeral vLLM cache"),
    ("pinned_seed", "after, first container\n(seeds the volume cache)"),
    ("pinned", "after,\nevery later container"),
]


def main(workdir: str) -> None:
    runs = {}
    for variant, _ in ROWS:
        path = Path(workdir) / f"load_model_profile_{variant}.json"
        runs[variant] = json.loads(path.read_text())

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ys = range(len(ROWS) - 1, -1, -1)
    for y, (variant, label) in zip(ys, ROWS):
        phases = runs[variant]["phases"]
        left = 0.0
        for key, _, color in PHASES:
            v = phases[key]
            ax.barh(y, v, left=left, color=color, height=0.55)
            if v >= 2.0:
                ax.text(left + v / 2, y, f"{v:.1f}", ha="center",
                        va="center", fontsize=8.5,
                        color="#333333" if color == GRAY else "white")
            left += v
        ax.text(left + 0.6, y, f"{phases['total_s']:.1f} s total",
                va="center", fontsize=9.5, color="#333333")
    before = runs["main"]["phases"]["total_s"]
    steady = runs["pinned"]["phases"]["total_s"]
    seed = runs["pinned_seed"]["phases"]["total_s"]
    ax.text(steady + 9.5, 0,
            f"−{before - steady:.1f} s vs before "
            f"({before / steady:.1f}x)",
            va="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([label for _, label in ROWS])
    ax.set_xlabel("seconds")
    ax.set_title("Cold load_model, Qwen3 4B fp8 on one H100")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in PHASES]
    ax.legend(handles, [name for _, name, _ in PHASES],
              loc="upper right", bbox_to_anchor=(0.99, 1.02))
    ax.set_xlim(0, seed * 1.32)
    fig.savefig(OUT / "load_model_speed.png", dpi=300)
    print(f"saved {OUT / 'load_model_speed.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
