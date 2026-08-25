"""Plots for the QUAIL-B speed-of-light table.

Reads the answers make_sol_quailb.py wrote. They live on the
quail-results volume, so pull them into a workdir first:

    W=<workdir>
    modal volume get quail-results /sol/sol_quailb_sf0.1.json $W/
    uv run --with matplotlib python reports/make_sol_quailb_plots.py $W
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, DARK, ORANGE  # noqa

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "plots"
plt.style.use(Path(__file__).parent / "quail.mplstyle")

W = Path(sys.argv[1])
D = json.loads((W / "sol_quailb_sf0.1.json").read_text())
Q = D["queries"]
ORDER = list(Q)


def sol(q, model):
    return Q[q]["models"][model]["sol_s"]


def plot_sol_per_query():
    """Both models on one log axis: the suite spans four orders of
    magnitude, so a linear axis would show only BIO-2."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = range(len(ORDER))
    w = 0.4
    a = [sol(q, "qwen3-4b-fp8") for q in ORDER]
    b = [sol(q, "qwen3-32b-fp8") for q in ORDER]
    ax.bar([i - w / 2 for i in x], a, w, color=BLUE, label="Qwen3-4B-fp8")
    ax.bar([i + w / 2 for i in x], b, w, color=ORANGE, label="Qwen3-32B-fp8")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va * 1.12, f"{va:.3g}", ha="center", fontsize=6,
                color=BLUE, rotation=90)
        ax.text(i + w / 2, vb * 1.12, f"{vb:.3g}", ha="center", fontsize=6,
                color=ORANGE, rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("speed of light, seconds on one H100 "
                  "(log scale, 4 orders of magnitude)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=7)
    ax.set_ylim(0.015, 3000)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_per_query.png", dpi=150)


# the document set each held column belongs to. FEVER shows up
# twice: FEV-1 filters claims and holds them, while the FEVER joins
# hold the evidence passages instead.
HELD_NAME = {"reviews.body": "IMDB", "reports.report": "BioDEX",
             "claims.claim": "FEVER", "evidence.text": "FEVER",
             "citations.destination_context": "LePaRD",
             "citations.passage_text": "LePaRD"}


def plot_attention_share():
    """Where each query's compute goes, ordered by the length of the
    documents whose KV is held.

    Attention pairs are quadratic in that length, so the share climbs
    left to right and jumps at BioDEX's 4,146-token reports. The 32B
    dot is always below the 4B one: attention scales 3.56x with model
    size where the dense projections scale 8.59x.

    Held length is the x ordering rather than the x value. Twenty-one
    of the 26 queries hold documents between 233 and 370 tokens, so
    on a true length axis they stack into one band and their labels
    cannot be read.
    """
    order = sorted(ORDER, key=lambda q: (Q[q]["held_mean_doc_tokens"], q))
    x = list(range(len(order)))

    def share(q, model):
        m = Q[q]["models"][model]
        return 100 * m["t_attention"] / m["t_compute"]

    lo = [share(q, "qwen3-32b-fp8") for q in order]
    hi = [share(q, "qwen3-4b-fp8") for q in order]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.vlines(x, lo, hi, color=DARK, lw=0.7, alpha=0.3)
    ax.scatter(x, hi, s=34, color=BLUE, label="Qwen3-4B-fp8", zorder=3)
    ax.scatter(x, lo, s=34, color=ORANGE, label="Qwen3-32B-fp8", zorder=3)
    for i, q in enumerate(order):
        if q.startswith("BIO"):
            ax.text(i, hi[i] + 1.4, f"{hi[i]:.0f}%", ha="center",
                    fontsize=7.5, color=BLUE)
            ax.text(i, lo[i] - 3.4, f"{lo[i]:.0f}%", ha="center",
                    fontsize=7.5, color=ORANGE)

    # one bracket per held corpus, with the length that puts it there
    trans = ax.get_xaxis_transform()
    start = 0
    for i in range(len(order) + 1):
        held = Q[order[start]]["held_column"]
        if i < len(order) and Q[order[i]]["held_column"] == held:
            continue
        mean = Q[order[start]]["held_mean_doc_tokens"]
        mid = (start + i - 1) / 2
        ax.plot([start - 0.35, i - 1 + 0.35], [-0.235, -0.235],
                transform=trans, color=DARK, lw=0.7, alpha=0.5,
                clip_on=False)
        ax.text(mid, -0.30, f"{HELD_NAME[held]}\n{mean:,.0f} tokens",
                transform=trans, ha="center", va="top", fontsize=7.5,
                color=DARK)
        start = i
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=7.5)
    ax.set_ylabel("attention share of compute, percent")
    ax.set_xlabel("queries, ordered by the length of the documents "
                  "whose KV is held", labelpad=78)
    ax.set_ylim(-4, 47)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


OUT.mkdir(exist_ok=True)
for old in ("sol_quailb_measured.png", "sol_quailb_kv_reuse.png"):
    (OUT / old).unlink(missing_ok=True)
plot_sol_per_query()
plot_attention_share()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
