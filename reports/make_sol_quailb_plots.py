"""Plots for the QUAIL-B speed-of-light table.

    uv run --with matplotlib python reports/make_sol_quailb_plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_colors import (BLUE, DARK, GRAY, LIGHT_GRAY, ORANGE,  # noqa: E402
                         RED)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "plots"
plt.style.use(Path(__file__).parent / "quail.mplstyle")

D = json.loads((ROOT / "results" / "sol_quailb_sf0.1.json").read_text())
Q = D["queries"]
VAL = D["validation"]["queries"]
ORDER = [q for q in Q]


def sol(q, m):
    return Q[q]["models"][m]["sol_s"]


def plot_sol_per_query():
    """Both models on one log axis: the whole suite spans four orders
    of magnitude, so a linear axis would show only BIO-2."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = range(len(ORDER))
    w = 0.4
    a = [sol(q, "qwen3-4b-fp8") for q in ORDER]
    b = [sol(q, "qwen3-32b-fp8") for q in ORDER]
    ax.bar([i - w / 2 for i in x], a, w, color=BLUE, label="Qwen3-4B-fp8")
    ax.bar([i + w / 2 for i in x], b, w, color=ORANGE,
           label="Qwen3-32B-fp8")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va * 1.12, f"{va:.3g}", ha="center", fontsize=6,
                color=BLUE, rotation=90)
        ax.text(i + w / 2, vb * 1.12, f"{vb:.3g}", ha="center", fontsize=6,
                color=ORANGE, rotation=90)
    ax.set_yscale("log")
    ax.set_ylabel("speed of light, seconds (log scale, 4 orders of "
                  "magnitude)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ORDER, rotation=90, fontsize=7)
    ax.set_ylim(0.015, 3000)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_per_query.png", dpi=150)


def plot_attention_share():
    """Why 32B does not cost a flat 8.6x: the attention term scales
    only 3.6x, so the queries it dominates scale by less."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    share = [100 * Q[q]["models"]["qwen3-4b-fp8"]["t_attention"]
             / Q[q]["models"]["qwen3-4b-fp8"]["t_compute"] for q in ORDER]
    ratio = [sol(q, "qwen3-32b-fp8") / sol(q, "qwen3-4b-fp8")
             for q in ORDER]
    bio = [q.startswith("BIO") for q in ORDER]
    ax.scatter([s for s, k in zip(share, bio) if not k],
               [r for r, k in zip(ratio, bio) if not k],
               s=34, color=GRAY, label="IMDB, FEVER, LePaRD")
    ax.scatter([s for s, k in zip(share, bio) if k],
               [r for r, k in zip(ratio, bio) if k],
               s=44, color=RED, label="BioDEX (4,146-token reports)")
    for q, s, r in zip(ORDER, share, ratio):
        if q in ("BIO-1", "BIO-2", "BIO-5", "IMDB-2", "FEV-1"):
            ax.annotate(q, (s, r), textcoords="offset points",
                        xytext=(7, -3), fontsize=7, color=DARK)
    ax.set_xlabel("attention share of compute at 4B, percent")
    ax.set_ylabel("32B speed of light / 4B speed of light")
    ax.axhline(8.59, color=DARK, lw=0.7, ls=":")
    ax.text(1, 8.68, "8.59x: the parameter-count ratio, what a "
                     "query with no attention would cost",
            fontsize=7, color=DARK)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_attention_share.png", dpi=150)


def plot_measured_against_floor():
    """The five queries with a measured wall. A linear axis, because
    the quantity is a fraction: how much of the wall the floor
    already accounts for. The walls themselves span 1.8 to 140
    seconds and are given as labels rather than lengths."""
    qs = list(VAL)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = list(range(len(qs)))
    frac = [VAL[q]["sol_s_4b"] / VAL[q]["engine_wall_s"] for q in qs]
    ax.barh(y, [1.0] * len(qs), 0.5, color=LIGHT_GRAY)
    ax.barh(y, frac, 0.5, color=BLUE)
    for i, q in enumerate(qs):
        f = frac[i]
        ax.text(f - 0.012, i, f"{100 * f:.0f}%", va="center", ha="right",
                fontsize=8.5, color="white", weight="bold")
        ax.text(1.02, i, f"{VAL[q]['sol_s_4b']:.2f} s floor  /  "
                         f"{VAL[q]['engine_wall_s']:.1f} s measured",
                va="center", fontsize=8, color=DARK)
    ax.set_yticks(y)
    ax.set_yticklabels(qs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.72)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of the measured wall the speed-of-light floor "
                  "accounts for, Qwen3-4B-fp8")
    fig.tight_layout()
    fig.savefig(OUT / "sol_quailb_measured.png", dpi=150)


OUT.mkdir(exist_ok=True)
plot_sol_per_query()
plot_attention_share()
plot_measured_against_floor()
print("wrote", *(p.name for p in sorted(OUT.glob("sol_quailb_*.png"))))
