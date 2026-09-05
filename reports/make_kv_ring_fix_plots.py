"""Figures for reports/2026-08-30-kv-ring-fix.md.

Reads the pre-fix and post-fix instrumented runs of IMDB-3 (and
BIO-2 for the no-regression number) and draws:

- kv_ring_fix_timeline.png: IMDB-3 filter chunk sizes over the run,
  pre-fix churn against the post-fix scan ring, one axis.
- kv_ring_fix_walls.png: IMDB-3 phase walls, pre against post.

Usage:

    uv run --with matplotlib python reports/make_kv_ring_fix_plots.py $W

where $W holds the files pulled from the quail-results volume:

    modal volume get quail-results experiments/discrepancy_imdb3.json $W/
    modal volume get quail-results experiments/ringfix_imdb3.json $W/
    modal volume get quail-results experiments/ringfix_tokens_head_imdb3.json $W/
    modal volume get quail-results experiments/ringfix_bio2.json $W/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED  # noqa: E402

plt.style.use(Path(__file__).parent / "quail.mplstyle")

OUT = Path(__file__).parent / "plots"


def filter_phase(run):
    return next(p for p in run["unprofiled"]["phases"]
                if p["op"] == "filter")


def join_phase(run):
    return next(p for p in run["unprofiled"]["phases"]
                if p["op"] == "join")


def chunk_series(phase):
    t0 = phase["chunks"][0]["t"]
    return ([c["t"] - t0 for c in phase["chunks"]],
            [c["tokens"] for c in phase["chunks"]])


def timeline(before, after):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bx, by = chunk_series(filter_phase(before))
    ax_, ay = chunk_series(filter_phase(after))
    ax.plot(bx, by, ".", color=RED, ms=3,
            label="unbounded retention")
    ax.plot(ax_, ay, "-o", color=BLUE, ms=6, lw=1,
            label="scan reserve")
    ax.set_yscale("log")
    ax.set_xlabel("seconds since the filter started")
    ax.set_ylabel("tokens in the forward pass (log scale)")
    bw = filter_phase(before)["wall_s"]
    aw = filter_phase(after)["wall_s"]
    ax.annotate(
        f"retention filled the arena; every admission\n"
        f"evicted first: {len(by):,} passes, {bw:.1f} s",
        xy=(bx[-1], by[-1]), xytext=(bx[-1] - 26, 3800), color=RED)
    ax.annotate(
        f"retention capped beside the reserve:\n"
        f"{len(ay)} full passes, {aw:.1f} s",
        xy=(ax_[-1], ay[-1]),
        xytext=(ax_[-1] + 1.5, ay[-1] * 0.45), color=BLUE)
    ax.legend(loc="upper right", frameon=False, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(OUT / "kv_ring_fix_timeline.png", dpi=300)


def walls(before, old_score, after):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    groups = [
        ("filter", filter_phase(before)["wall_s"],
         filter_phase(old_score)["wall_s"],
         filter_phase(after)["wall_s"]),
        ("join", join_phase(before)["wall_s"],
         join_phase(old_score)["wall_s"],
         join_phase(after)["wall_s"]),
    ]
    x = range(len(groups))
    width = 0.25
    labels = ("unbounded", "saved seconds", "prefix tokens")
    colors = (GRAY, ORANGE, BLUE)
    offsets = (-width, 0, width)
    for i, (_name, *values) in enumerate(groups):
        for value, label, color, offset in zip(
                values, labels, colors, offsets):
            ax.bar(i + offset, value, width, color=color, label=(
                label if i == 0 else None))
            ax.text(i + offset, value + 0.6, f"{value:.1f}",
                    ha="center", color=DARK)
    speed = groups[0][1] / groups[0][3]
    ax.text(0, groups[0][1] * 0.55, f"{speed:.1f}x", ha="center",
            color=BLUE)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{group[0]} phase" for group in groups])
    ax.set_ylabel("seconds")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "kv_ring_fix_walls.png", dpi=300)


def main(workdir):
    w = Path(workdir)
    before = json.loads((w / "discrepancy_imdb3.json").read_text())
    old_score = json.loads((w / "ringfix_imdb3.json").read_text())
    after = json.loads((w / "ringfix_tokens_head_imdb3.json").read_text())
    OUT.mkdir(exist_ok=True)
    timeline(before, after)
    walls(before, old_score, after)

    for name, run in (("unbounded retention", before),
                      ("saved seconds", old_score),
                      ("prefix tokens", after)):
        u = run["unprofiled"]
        f = filter_phase(run)
        print(f"{name}: engine {u['engine_wall_s']}s, filter "
              f"{f['wall_s']}s in {f['n_chunks']} passes "
              f"(mean {f['tokens'] // f['n_chunks']:,} tokens), "
              f"join {join_phase(run)['wall_s']}s, "
              f"evictions {len(u['evictions'])}, "
              f"regret {u['regret_tokens']:,}, "
              f"hit tokens {u['kv_hit_tokens']:,}")
    bio = w / "ringfix_bio2.json"
    if bio.exists():
        u = json.loads(bio.read_text())["unprofiled"]
        print(f"BIO-2 scan ring: engine {u['engine_wall_s']}s, "
              f"regret {u['regret_tokens']:,}, "
              f"evictions {len(u['evictions'])}")


if __name__ == "__main__":
    main(sys.argv[1])
