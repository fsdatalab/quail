"""Plots for the QUAIL-B speed-of-light table.

    uv run --with matplotlib python reports/make_sol_quailb_plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED  # noqa

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "plots"
plt.style.use(Path(__file__).parent / "quail.mplstyle")

Q = json.loads((ROOT / "results" / "sol_quailb_sf0.1.json").read_text()
               )["queries"]
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
    """Why 32B is not a flat 8.59x: the attention term scales only
    3.56x, so the queries it dominates scale by less."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    share = [100 * Q[q]["models"]["qwen3-4b-fp8"]["t_attention"]
             / Q[q]["models"]["qwen3-4b-fp8"]["t_compute"] for q in ORDER]
    ratio = [sol(q, "qwen3-32b-fp8") / sol(q, "qwen3-4b-fp8") for q in ORDER]
    bio = [q.startswith("BIO") for q in ORDER]
    ax.scatter([s for s, k in zip(share, bio) if not k],
               [r for r, k in zip(ratio, bio) if not k],
               s=34, color=GRAY, label="IMDB, FEVER, LePaRD (83 to 370 "
                                       "token documents)")
    ax.scatter([s for s, k in zip(share, bio) if k],
               [r for r, k in zip(ratio, bio) if k],
               s=44, color=RED, label="BioDEX (4,146 token reports)")
    for q, s, r in zip(ORDER, share, ratio):
        if q in ("BIO-1", "BIO-2", "BIO-5", "IMDB-2", "FEV-1"):
            ax.annotate(q, (s, r), textcoords="offset points",
                        xytext=(7, -3), fontsize=7, color=DARK)
    ax.set_xlabel("attention share of compute at 4B, percent")
    ax.set_ylabel("32B speed of light / 4B speed of light")
    ax.axhline(8.59, color=DARK, lw=0.7, ls=":")
    ax.text(1, 8.68, "8.59x, the parameter-count ratio: what a query "
                     "with no attention would cost",
            fontsize=7, color=DARK)
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


OUT.mkdir(exist_ok=True)
for old in ("sol_quailb_measured.png", "sol_quailb_kv_reuse.png"):
    (OUT / old).unlink(missing_ok=True)
plot_sol_per_query()
plot_attention_share()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
