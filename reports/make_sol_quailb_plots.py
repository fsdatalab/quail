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


def plot_attention_share():
    """Where the time goes, against the length of the documents whose
    KV is held. Attention pairs are quadratic in that length, so the
    longer the held document, the more of the compute is attention.

    The curves are the same equations applied to a synthetic single
    document, so they are drawn from the model rather than fitted to
    the 26 points. The points sit off the curve where a query's join
    streams extra tokens past the held documents."""
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    curve = D["attention_share_curve"]
    cx = curve["document_tokens"]
    for model, colour, label in (("qwen3-4b-fp8", BLUE, "Qwen3-4B-fp8"),
                                 ("qwen3-32b-fp8", ORANGE,
                                  "Qwen3-32B-fp8")):
        ax.plot(cx, curve["attention_share"][model], color=colour,
                lw=1.2, alpha=0.55)
        ax.scatter([Q[q]["held_mean_doc_tokens"] for q in ORDER],
                   [100 * Q[q]["models"][model]["t_attention"]
                    / Q[q]["models"][model]["t_compute"] for q in ORDER],
                   s=30, color=colour, label=label, zorder=3)
    for corpus, xt, yt, dx, dy in (
            ("claims, 11 tokens", 11.4, 0.6, 4, 12),
            ("excerpts, 233", 233.1, 3.7, -46, -12),
            ("reviews, 299", 298.8, 4.9, 8, 12),
            ("evidence, 370", 370.2, 6.3, 52, 2),
            ("reports, 4,146", 4146.0, 40.0, -22, 10)):
        ax.annotate(corpus, (xt, yt), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, color=DARK,
                    ha="center")
    ax.set_xscale("log")
    ax.set_xlim(5, 40_000)
    ax.set_ylim(-3, 62)
    ax.set_xlabel("mean length of the documents whose KV is held, "
                  "tokens (log scale)")
    ax.set_ylabel("attention share of compute, percent")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.text(6000, 4, "lines: the same equations over one\n"
                     "synthetic document\npoints: the 26 QUAIL-B queries",
            fontsize=7.5, color=DARK, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


OUT.mkdir(exist_ok=True)
for old in ("sol_quailb_measured.png", "sol_quailb_kv_reuse.png"):
    (OUT / old).unlink(missing_ok=True)
plot_sol_per_query()
plot_attention_share()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
