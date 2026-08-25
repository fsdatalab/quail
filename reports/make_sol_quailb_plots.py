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


# Each context length, the document set it comes from, and where to
# put its label. The three middle clusters are ~20pt apart on a log
# axis, so their labels sit at staggered heights to clear each other.
CONTEXTS = {
    11.4: ("FEVER", 6.0),
    233.1: ("LePaRD", 11.0),
    298.8: ("IMDB", 18.0),
    370.2: ("FEVER", 25.0),
    4146.0: ("BioDEX", 44.0),
}


def plot_attention_share():
    """Attention's share of the compute, against the context each new
    token attends over.

    A token attending over a 4,146-token report scores 4,146 pairs; a
    token attending over an 11-token claim scores 11. So the longer
    the context, the more of the work is attention. The 32B dot sits
    below the 4B one everywhere, because attention scales 3.56x with
    model size where the dense projections scale 8.59x.

    Points are not labelled by query. Twenty-one of the 26 queries
    share three context lengths and four of them land on one point,
    so per-query labels need leader lines long enough to obscure the
    data. The report's table gives the per-query numbers.
    """
    def share(q, model):
        m = Q[q]["models"][model]
        return 100 * m["t_attention"] / m["t_compute"]

    x = [Q[q]["held_mean_doc_tokens"] for q in ORDER]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.vlines(x, [share(q, "qwen3-32b-fp8") for q in ORDER],
              [share(q, "qwen3-4b-fp8") for q in ORDER],
              color=DARK, lw=0.7, alpha=0.25)
    for model, colour, label in (("qwen3-4b-fp8", BLUE, "Qwen3-4B-fp8"),
                                 ("qwen3-32b-fp8", ORANGE,
                                  "Qwen3-32B-fp8")):
        ax.scatter(x, [share(q, model) for q in ORDER], s=34,
                   color=colour, label=label, zorder=3)

    for ctx, (name, ty) in CONTEXTS.items():
        top = max(share(q, "qwen3-4b-fp8") for q in ORDER
                  if abs(Q[q]["held_mean_doc_tokens"] - ctx) < 0.05)
        n = sum(1 for q in ORDER
                if abs(Q[q]["held_mean_doc_tokens"] - ctx) < 0.05)
        ax.vlines(ctx, top + 0.8, ty - 1.6, color=DARK, lw=0.6,
                  alpha=0.35)
        ax.text(ctx, ty, f"{name}\n{ctx:,.0f} tokens, "
                         f"{n} quer{'y' if n == 1 else 'ies'}",
                ha="center", va="bottom", fontsize=7.5, color=DARK)

    ax.set_xscale("log")
    ax.set_xlim(7, 9000)
    ax.set_ylim(-2, 50)
    ax.set_xlabel("mean context each new token attends over, tokens "
                  "(log scale)")
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
