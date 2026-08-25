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


# Where each cluster's labels go: (label x, first label y, y step,
# text alignment). Twenty-one of the 26 queries hold documents
# between 233 and 370 tokens, so their dots pile into one narrow band
# and the labels have to sit out in the empty space with a leader
# line back. The two free regions are left of 233 and right of 370.
LABEL_SLOTS = {
    11.4: (15.0, 8.0, 0.0, "left"),
    233.1: (17.0, 11.0, 5.6, "left"),
    298.8: (760.0, 8.5, 4.6, "left"),
    370.2: (2300.0, 9.0, 5.2, "left"),
    4146.0: (7000.0, 30.0, 3.6, "left"),
}


def plot_attention_share():
    """Where each query's compute goes, against the length of the
    documents whose KV is held.

    Attention pairs are quadratic in that length, so the share climbs
    with it and jumps at BioDEX's 4,146-token reports. The 32B dot
    sits below the 4B one everywhere: attention scales 3.56x with
    model size where the dense projections scale 8.59x.

    Queries whose dots coincide share a label. LEP-1, LEP-5, LEP-6
    and LEP-8 land on the same point because LEP1 leaves 4 documents
    of 200 and LEP3 leaves none, so their later stages and joins cost
    nothing.
    """
    def share(q, model):
        m = Q[q]["models"][model]
        return 100 * m["t_attention"] / m["t_compute"]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.vlines([Q[q]["held_mean_doc_tokens"] for q in ORDER],
              [share(q, "qwen3-32b-fp8") for q in ORDER],
              [share(q, "qwen3-4b-fp8") for q in ORDER],
              color=DARK, lw=0.7, alpha=0.25)
    for model, colour, label in (("qwen3-4b-fp8", BLUE, "Qwen3-4B-fp8"),
                                 ("qwen3-32b-fp8", ORANGE,
                                  "Qwen3-32B-fp8")):
        ax.scatter([Q[q]["held_mean_doc_tokens"] for q in ORDER],
                   [share(q, model) for q in ORDER],
                   s=34, color=colour, label=label, zorder=3)

    # one label per distinct dot, listing every query that lands on it
    for held, (lx, ly, dy, ha) in LABEL_SLOTS.items():
        here = [q for q in ORDER
                if abs(Q[q]["held_mean_doc_tokens"] - held) < 0.05]
        # merge only dots that genuinely coincide, so LEP-1, 5, 6 and
        # 8 share a label but IMDB-2 and IMDB-3 keep their own
        merged = []
        for q in sorted(here, key=lambda q: share(q, "qwen3-4b-fp8")):
            y = share(q, "qwen3-4b-fp8")
            if merged and y - merged[-1][0] < 0.05:
                merged[-1][1].append(q)
            else:
                merged.append((y, [q]))
        for j, (y, qs) in enumerate(merged):
            ty = ly + j * dy
            ax.annotate(", ".join(qs), xy=(held, y), xytext=(lx, ty),
                        fontsize=7.5, color=DARK, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", color=DARK,
                                        lw=0.5, alpha=0.4,
                                        shrinkA=0, shrinkB=4))

    ax.set_xscale("log")
    ax.set_xlim(8, 26_000)
    ax.set_ylim(-2, 46)
    ax.set_xlabel("mean length of the documents whose KV is held, "
                  "tokens (log scale)")
    ax.set_ylabel("attention share of compute, percent")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


OUT.mkdir(exist_ok=True)
for old in ("sol_quailb_measured.png", "sol_quailb_kv_reuse.png"):
    (OUT / old).unlink(missing_ok=True)
plot_sol_per_query()
plot_attention_share()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
