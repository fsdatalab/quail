r"""Plot headroom admission on three QUAIL-B map queries against vLLM.

Reads the records experiments/cells/answer_headroom.py saved for three
map queries on Qwen3 4B FP8, one H100 per run: CUAD-22 and CUAD-21 at
scale factor 0.5, and AGENT-6 at 0.1. Each query ran four ways: vLLM
with all of its prompts in one generate call, in document order and
shuffled, and Quail at zero headroom and with headroom. The runs saved
the queries under their IDs before quail-bench#64 renumbered them, CUAD-23
and CUAD-22, so RUNS maps the saved IDs to the current ones. Writes two
PNG plots. Pull the run directories, then run from the repository root:

    W=/tmp/map-headroom; mkdir -p "$W"
    for run in 20261031T174222Z-6e4398fa 20261031T174230Z-7189ec8c; do
      uv run modal volume get quail-results \
        "ablations/answer_headroom/$run" "$W/"
    done
    uv run --with matplotlib python reports/make_map_headroom_plots.py "$W"

The SoL of each query is the record the cell's sol entry point wrote
from the headroom run's answers: the least time any schedule of the
query's work can take, with unlimited KV.
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED

HERE = Path(__file__).parent
plt.style.use(HERE / "quail.mplstyle")
OUT = HERE / "plots"
# (current ID, saved ID, run directory, trace file stem, document noun,
# description)
RUNS = (
    ("CUAD-22", "CUAD-23", "20261031T174222Z-6e4398fa",
     "map_obligations__definitions__s", "contract",
     "16 maps per contract\n224 contracts, scale factor 0.5"),
    ("CUAD-21", "CUAD-22", "20261031T174222Z-6e4398fa",
     "map_obligations__plain_terms", "contract",
     "3 maps per contract\n224 contracts, scale factor 0.5"),
    ("AGENT-6", "AGENT-6", "20261031T174230Z-7189ec8c",
     "map_commands__plan__edits", "snapshot",
     "3 maps per trace snapshot\n1,772 snapshots, scale factor 0.1"),
)
MODES = (("stock_vllm", "vLLM, ordered", DARK),
         ("shuffled_vllm", "vLLM, shuffled", GRAY),
         ("headroom=0", "Quail, headroom=0", ORANGE),
         ("headroom", "Quail, headroom", BLUE))
MEASURES = (("wall_s", "seconds", "Query time"),
            ("answer_evictions", "evictions", "Answer evictions"),
            ("recomputed_tokens", "tokens", "Recomputed tokens"))
SAMPLE = 8  # tokens between the saved points of a survival curve
POOL_PAGES = 353_136 // 16  # Qwen3 4B FP8 KV on an H100, in 16-token pages


def slug(query):
    return "".join(c if c.isalnum() else "_" for c in query)


def load_results(root):
    """Return each (query, mode)'s run record and each query's SoL seconds."""
    rows, sols = {}, {}
    for query, saved, run, *_ in RUNS:
        for mode, *_ in MODES:
            path = root / run / f"{mode}-{slug(saved)}.json"
            if path.exists():
                rows[(query, mode)] = json.loads(path.read_text())[
                    "queries"][saved][0]
        sol = root / run / "headroom-answers" / f"{slug(saved)}.sol.json"
        if sol.exists():
            sols[query] = json.loads(sol.read_text())["seconds"]
    return rows, sols


def plot_results(root):
    rows, sols = load_results(root)
    queries = [query for query, *_ in RUNS]
    descriptions = {query: description for query, *_, description in RUNS}
    figure, axes = plt.subplots(len(MEASURES), 1, figsize=(11, 12))
    width = 0.2
    for axis, (key, unit, title) in zip(axes, MEASURES):
        for i, (mode, label, color) in enumerate(MODES):
            for q, query in enumerate(queries):
                x = q + (i - 1.5) * width
                row = rows.get((query, mode))
                if row is None:
                    axis.text(x, 0, "x", ha="center", va="bottom")
                    continue
                value = row[key]
                axis.bar(x, value, width * 0.92, color=color,
                         label=label if q == 0 else None)
                text = f"{value / 1e6:.2f}M" if value >= 1e5 else f"{value:,.0f}"
                if key == "wall_s" and mode != "stock_vllm":
                    text += f"\n{rows[(query, 'stock_vllm')]['wall_s'] / value:.2f}x"
                axis.text(x, value, text, ha="center", va="bottom", fontsize=7)
        if key == "wall_s":
            for q, query in enumerate(queries):
                if query in sols:
                    axis.hlines(sols[query], q - 2 * width, q + 2 * width,
                                color=RED, lw=1.4, ls="--")
                    axis.text(q + 2 * width, sols[query], f"SoL\n{sols[query]:.0f}",
                              color=RED, fontsize=7, va="center")
            axis.set_ylim(0, max(row["wall_s"] for row in rows.values()) * 1.35)
            axis.legend(ncol=4, loc="upper right")
        if key == "recomputed_tokens":
            axis.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
        axis.set_xticks(range(len(queries)),
                        [f"{query}\n{descriptions[query]}" for query in queries])
        axis.set_ylabel(unit)
        axis.set_title(title + (" (speedup over vLLM ordered under each time)"
                                if key == "wall_s" else ""), loc="left")
    figure.suptitle("Three QUAIL-B map queries, Qwen3 4B FP8, one H100 per run",
                    x=0.01, y=1.0, ha="left", fontweight="bold")
    figure.savefig(OUT / "map_headroom_results.png", dpi=300, bbox_inches="tight")


def load_trace(root, run, stem):
    (path,) = glob.glob(f"{root}/{run}/headroom-traces/*-{stem}*.json")
    return json.loads(Path(path).read_text())


def plot_admission(root):
    figure, axes = plt.subplots(4, len(RUNS), figsize=(15, 15))
    for column, (query, _, run, stem, noun, _) in enumerate(RUNS):
        trace = load_trace(root, run, stem)
        names = trace["label"].removeprefix("map ").split(", ")
        granted = defaultdict(list)
        for _, bucket, pages in trace["granted"]:
            granted[bucket].append(pages)
        # each map's survival curve in the length range with the most answers
        axis = axes[0, column]
        common = max(granted, key=lambda b: len(granted[b]))
        curves = []
        for q in range(len(names)):
            curve = np.asarray(trace["curves"].get(f"{q}:{common}", []))
            if len(curve):
                below = np.flatnonzero(curve < 0.5)
                curves.append((below[0] if len(below) else len(curve), q, curve))
        curves.sort()
        for rank, (_, q, curve) in enumerate(curves):
            tokens = np.arange(len(curve)) * SAMPLE
            ends = rank in (0, len(curves) - 1)
            many = len(curves) > 3
            color = (BLUE if rank == 0 else RED) if ends or not many else GRAY
            if not many and not ends:
                color = DARK
            axis.plot(tokens, curve, color=color, lw=2 if ends else 1.1,
                      label=names[q] if ends or not many else None)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("answer tokens")
        axis.set_ylabel("share of answers")
        axis.legend()
        axis.set_title(f"{query}: answers still running, per map\n"
                       f"({noun}s of {2 ** common:,}-{2 ** (common + 1) - 1:,} "
                       "tokens)")
        # headroom at admission by document length
        axis = axes[1, column]
        buckets = sorted(granted)
        means = [np.mean(granted[b]) for b in buckets]
        axis.bar(range(len(buckets)), means, color=BLUE, width=0.7)
        for i, b in enumerate(buckets):
            axis.text(i, means[i], f"{means[i]:.0f}\n({len(granted[b]) // len(names)})",
                      ha="center", va="bottom", fontsize=7)
        axis.set_xticks(range(len(buckets)),
                        [f"{2 ** b:,}-\n{2 ** (b + 1) - 1:,}" for b in buckets],
                        fontsize=7)
        axis.set_xlabel(f"{noun} tokens")
        axis.set_ylabel("pages")
        axis.set_ylim(0, max(max(means), 1) * 1.3)
        axis.set_title(f"{query}: headroom at admission\n"
                       f"(mean per answer; {noun}s in parentheses)")
        # headroom reserved and running answers over the run
        answers, pages = defaultdict(int), defaultdict(float)
        for refresh, _, _, count, mean in trace["rebuilds"]:
            answers[refresh] += count
            pages[refresh] += count * mean
        refreshes = np.array(sorted(answers))
        held = np.array([pages[r] for r in refreshes])
        running = np.array([answers[r] for r in refreshes])
        admitted = Counter(r for r, _, _ in trace["granted"])
        axis = axes[2, column]
        axis.plot(refreshes, held, color=BLUE, lw=1)
        axis.axhline(POOL_PAGES, color=DARK, lw=1, ls="--")
        axis.text(refreshes[-1], POOL_PAGES, f"KV pool {POOL_PAGES:,}", ha="right",
                  va="bottom", fontsize=8)
        axis.set_ylim(0, POOL_PAGES * 1.12)
        axis.set_xlabel("admission refresh")
        axis.set_ylabel("pages")
        axis.set_title(f"{query}: headroom reserved\n(pages not yet written)")
        axis = axes[3, column]
        axis.plot(refreshes, running, color=DARK, lw=1)
        axis.vlines(sorted(admitted), 0, running.max() * 0.04, color=RED, lw=0.8)
        axis.set_ylim(0, running.max() * 1.1)
        axis.set_xlabel("admission refresh")
        axis.set_ylabel("answers")
        axis.set_title(f"{query}: running answers\n(red: {len(admitted)} "
                       "refreshes that admitted)")
    figure.suptitle("Headroom admission on three QUAIL-B map queries, Qwen3 4B FP8, "
                    "one H100", x=0.01, ha="left", fontweight="bold")
    figure.savefig(OUT / "map_headroom_admission.png", dpi=300, bbox_inches="tight")


def main(root):
    root = Path(root)
    plot_results(root)
    plot_admission(root)


if __name__ == "__main__":
    main(sys.argv[1])
