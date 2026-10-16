r"""Plot the AI.EXTRACT draft-length sweeps and the adaptive policy's traces.

Reads the records that experiments/cells/extract_draft.py saved for eight
extraction queries on Qwen3 4B FP8: one sweep record per query, with the
fixed draft lengths and the adaptive policy timed in one container, and
one logged adaptive run per query, with every round's draft and result.
Writes two PNG plots. Pull the records, then run from the repository root
on a checkout that has quail.cost.draft_policy (e33ed43 or later):

    W=/tmp/extract-draft; mkdir -p "$W"
    P=ablations/extract_draft
    for f in finalcuad1-CUAD-1-sf1.0 finalcuad3-CUAD-3-sf1.0 \\
             finalactor-reviews-sf0.1 finalacting-reviews-sf0.1 \\
             finalsecname-sec_filings-sf1.0 finalsecdate-sec_filings-sf1.0 \\
             finalsecreason-sec_filings-sf1.0 finalsecevent-sec_filings-sf1.0; do
      uv run modal volume get quail-results "$P/$f-qwen3-4b-fp8-d16.json" "$W/"
      uv run modal volume get quail-results \\
        "$P/${f/final/trace}-qwen3-4b-fp8-dadaptive.json" "$W/"
    done
    uv run --with matplotlib python reports/make_extract_draft_plots.py "$W"

In a sweep record the first pass over the settings is the warmup and
is left out; the two passes after it are the measured runs. The traces
replay the policy on the logged settled answers and drafted rounds.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from plot_colors import BLUE, DARK, ORANGE

from quail.cost.draft_policy import LIVE_BUCKETS, DraftPolicy

HERE = Path(__file__).parent
plt.style.use(HERE / "quail.mplstyle")
OUT = HERE / "plots"
# (record stem, panel title, typical KV read per round in fed tokens)
QUERIES = [
    ("cuad1-CUAD-1-sf1.0",
     "CUAD-1 governing law: 7,600-token contracts, short answers", 90),
    ("cuad3-CUAD-3-sf1.0",
     "CUAD-3 liability cap: 7,600-token contracts, long answers", 90),
    ("actor-reviews-sf0.1",
     "IMDB actor named: 380-token reviews, short answers", 5),
    ("acting-reviews-sf0.1",
     "IMDB view of the acting: 380-token reviews, long answers", 5),
    ("secname-sec_filings-sf1.0",
     "SEC-2 departing CEO's name: 1,700-token filings, mostly null", 22),
    ("secdate-sec_filings-sf1.0",
     "SEC-3 departure date: 1,700-token filings, mostly null", 22),
    ("secreason-sec_filings-sf1.0",
     "SEC-4 departure reason: 1,700-token filings, mostly null", 22),
    ("secevent-sec_filings-sf1.0",
     "SEC-7 event sentence: 1,700-token filings, 24-token answers", 22),
]
CHECKPOINTS = (0, 8, 64, 512)


def summarize(path):
    """Return {setting: (wall seconds per measured run, GPU busy share, rounds)}."""
    record = json.load(open(path))
    settings = record["sweep"]
    timed = [r for r in record["runs"] if r["mode"] == "timed"]
    warm = timed[len(settings):]
    out = {}
    for setting in settings:
        runs = [r for r in warm if r["draft_tokens"] == setting]
        walls = [r["measurements"]["wall_s"] for r in runs]
        p = runs[-1]["passes"]
        busy = (p["gpu_s"] + p.get("readout_gpu_s", 0.0)) / p["window_s"]
        out[setting] = (walls, busy, runs[-1]["node_metrics"]["ai-extract:0"]["steps"])
    return out


def sweep_figure(src):
    present = [(s, t) for s, t, _ in QUERIES
               if (src / f"final{s}-qwen3-4b-fp8-d16.json").exists()]
    rows = (len(present) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.25 * rows), squeeze=False)
    for ax in axes.ravel()[len(present):]:
        ax.set_axis_off()
    for ax, (stem, title) in zip(axes.ravel(), present):
        results = summarize(src / f"final{stem}-qwen3-4b-fp8-d16.json")
        fixed = [s for s in results if s != "adaptive"]
        x = np.arange(len(fixed))
        means = [float(np.mean(results[s][0])) for s in fixed]
        every = [w for s in results for w in results[s][0]]
        span = max(every) - min(every) + 0.5
        for i, s in enumerate(fixed):
            walls, busy, _ = results[s]
            ax.scatter([i] * len(walls), walls, color=BLUE, s=14, zorder=3)
            ax.scatter([i], [means[i]], marker="_", s=160, color=BLUE, zorder=4)
            ax.text(i, max(walls) + 0.03 * span,
                    f"{means[i]:.1f} s\nGPU {100 * busy:.0f}%",
                    ha="center", fontsize=7.5, color=DARK)
        ax.plot(x, means, color=BLUE, linewidth=1.2,
                label="fixed draft length: runs (dots) and their mean (bar)")
        xa = len(fixed) + 0.4
        walls, busy, _ = results["adaptive"]
        mean = float(np.mean(walls))
        ax.axhline(mean, color=ORANGE, linewidth=1.0, linestyle="--",
                   label=f"adaptive policy: mean {mean:.1f} s")
        ax.scatter([xa] * len(walls), walls, color=ORANGE, s=14, zorder=3)
        ax.scatter([xa], [mean], marker="_", s=160, color=ORANGE, zorder=4)
        ax.text(xa, max(walls) + 0.03 * span,
                f"{mean:.1f} s\nGPU {100 * busy:.0f}%",
                ha="center", fontsize=7.5, color=DARK)
        ax.set_xticks(list(x) + [xa], [str(s) for s in fixed] + ["adaptive"])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylim(min(every) - 0.12 * span, max(every) + 0.24 * span)
        ax.set_xlabel("draft_tokens")
        ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Extraction query time by draft length, Qwen3 4B FP8, one H100; "
                 "each panel is one container", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "extract_draft_sweep.png", dpi=300, bbox_inches="tight")
    for stem, title in present:
        print(title)
        for s, (walls, busy, steps) in summarize(
                src / f"final{stem}-qwen3-4b-fp8-d16.json").items():
            print(f"  {str(s):>9}: {np.mean(walls):6.2f} s"
                  f"  runs {[round(w, 2) for w in walls]}"
                  f"  GPU {100 * busy:3.0f}%  rounds {steps}")


def shades(color, n):
    """Return n shades of one color, from a pale tint to near black."""
    c = np.array(to_rgb(color))
    white, black = np.ones(3), np.zeros(3)
    out = []
    for t in np.linspace(0.0, 1.0, n):
        if t <= 0.5:
            a = t / 0.5
            out.append((1 - a) * (0.25 * c + 0.75 * white) + a * c)
        else:
            a = (t - 0.5) / 0.5
            out.append((1 - a) * c + a * (0.35 * c + 0.65 * black))
    return out


def rounds_in_order(rounds):
    """Return (draft length, live positions, first draft token written) per round.

    Draft rows and verify rows are each in time order per document and
    question; the k-th drafted round of a pair matches its k-th verify
    row that fed tokens.
    """
    drafts = defaultdict(list)
    for doc, q, position, live, _ in rounds["drafts"]:
        drafts[doc, q].append((position, live))
    out = []
    seen = defaultdict(int)
    for doc, q, fed, discard, done, written in rounds["verified"]:
        if fed == 0:
            continue
        position, live = drafts[doc, q][seen[doc, q]]
        seen[doc, q] += 1
        gamma = fed - 1
        ended_at_once = done and written - position <= 1
        accepted = gamma > 0 and discard < gamma and not ended_at_once
        out.append((gamma, live, accepted))
    return out


def trace_figure(src):
    present = [(s, t, r) for s, t, r in QUERIES
               if (src / f"trace{s}-qwen3-4b-fp8-dadaptive.json").exists()]
    fig, axes = plt.subplots(len(present), 3, figsize=(18, 3.6 * len(present)),
                             squeeze=False)
    for (stem, title, rho), (left, middle, right) in zip(present, axes):
        record = json.load(open(src / f"trace{stem}-qwen3-4b-fp8-dadaptive.json"))
        rounds = next(r for r in record["runs"] if r["mode"] == "logged")["rounds"]
        steps = rounds_in_order(rounds)
        gamma = np.array([g for g, _, _ in steps])
        live = np.array([m for _, m, _ in steps])
        x = np.arange(len(gamma))
        one = live == 1
        left.scatter(x[one], gamma[one], s=3, color=BLUE, alpha=0.5, linewidths=0,
                     label=f"one live position ({one.sum()} rounds)")
        left.scatter(x[~one], gamma[~one], s=3, color=ORANGE, alpha=0.5,
                     linewidths=0,
                     label=f"several live positions ({(~one).sum()} rounds)")
        window = max(1, len(gamma) // 60)
        if len(gamma) > window:
            smooth = np.convolve(gamma, np.ones(window) / window, mode="valid")
            left.plot(np.arange(len(smooth)) + window // 2, smooth, color=DARK,
                      linewidth=1.2, label=f"mean over {window} rounds")
        left.set_title(f"{title}\ndraft length per round, in query order", fontsize=9)
        left.set_xlabel("round")
        left.set_ylabel("draft tokens")
        left.legend(loc="upper right", frameon=False, fontsize=7.5, markerscale=3)

        lengths = [len(w) for _, _, w, _, _, null, _ in rounds["settled"]
                   if not null and w]
        observed = [(m, a) for _, m, a in steps if m > 1]
        marks = [c for c in CHECKPOINTS if c < len(lengths)] + [len(lengths)]
        blues, oranges = shades(BLUE, len(marks)), shades(ORANGE, len(marks))
        for i, n in enumerate(marks):
            policy = DraftPolicy(63, 64, 16)
            for length in lengths[:n]:
                policy.settle(length)
            share = n / max(1, len(lengths))
            for m, accepted in observed[:int(share * len(observed))]:
                policy.observe(m, accepted)
            curve = policy.survival(1)
            label = (f"after {n} answers" if n
                     else "planner seed: geometric, mean 16")
            label += f"; one-match draft {policy.draft_length(rho, 1, 1)}"
            middle.plot(np.arange(len(curve)), curve, color=blues[i],
                        linewidth=1.6, label=label)
            ms = [m for m in LIVE_BUCKETS if m > 1]
            label = f"after {n} answers" if n else "seed: 1 / live positions"
            label += (f"; draft at 2 live {policy.draft_length(rho, 1, 2)}, "
                      f"at 10 live {policy.draft_length(rho, 1, 10)}")
            right.plot(ms, [policy.source_odds(m) for m in ms], color=oranges[i],
                       linewidth=1.6, marker="o", markersize=3.5, label=label)
        middle.set_title("answer-length curve: P(more than j tokens left) at the "
                         "first round\nlegend: the one-match draft it gives for "
                         f"rho = {rho}", fontsize=9)
        middle.set_xlabel("j, tokens")
        middle.set_ylabel("probability")
        middle.set_ylim(0, 1.02)
        middle.legend(loc="upper right", frameon=False, fontsize=7)
        right.set_title("source odds: P(the position followed is the answer's "
                        f"source)\nlearned from {(~one).sum()} drafted rounds with "
                        "several positions", fontsize=9)
        right.set_xscale("log")
        right.set_xlabel("live positions (log scale)")
        right.set_ylabel("probability")
        right.set_ylim(0, 1.02)
        right.legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "extract_draft_trace.png", dpi=300, bbox_inches="tight")


def main():
    src = Path(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    sweep_figure(src)
    trace_figure(src)


if __name__ == "__main__":
    main()
