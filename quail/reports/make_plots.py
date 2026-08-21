"""Plots for the 2026-08-18 and 2026-08-19 reports (QUAIL-B, vs
stock).

Reads the measured JSON artifacts in results/ and writes PNGs into
reports/plots/. Run from the quail/ directory:

    uv run --with matplotlib python reports/make_plots.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
RESULTS = ROOT / "results"
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import QUAIL, STOCK, GOOD, DARK


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


# ---- figure: quail vs stock vLLM ------------------------------------

post = load("m1_filter.json")["runs"]
stockf = load("baseline_filter4.json")["runs"]
quail_filter = sum(r["wall"] for r in post) / len(post)
stock_filter = sum(r["wall"] for r in stockf) / len(stockf)

stockj_data = load("baseline_join.json")["runs"]
stockj = sum(r["wall"] for r in stockj_data) / len(stockj_data)
quail_join = load("dispatch_gate.json")["gpus1"]["join"]["wall_s"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))

for ax, title, stock_val, quail_val, stock_label in [
    (a1, "5-filter chain, 10k docs", stock_filter, quail_filter, "Stock vLLM\n(tuned)"),
    (a2, "72k-pair join", stockj, quail_join, "Stock vLLM\n(grouped)"),
]:
    bars = ax.bar(
        [stock_label, "Quail"], [stock_val, quail_val],
        color=[STOCK, QUAIL], width=0.52, edgecolor="white", linewidth=0.8,
    )
    for b, v in zip(bars, [stock_val, quail_val]):
        ax.text(
            b.get_x() + b.get_width() / 2, v + stock_val * 0.02,
            f"{v:.1f} s", ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color=DARK,
        )
    speedup = stock_val / quail_val
    ax.text(
        1, quail_val + stock_val * 0.10,
        f"{speedup:.1f}x" if speedup >= 2 else f"{speedup:.2f}x",
        ha="center", fontsize=11, fontweight="bold", color=GOOD,
    )
    ax.set_title(title, fontsize=11, fontweight="normal", pad=8)
    ax.set_ylabel("wall time (s)")
    ax.set_ylim(0, stock_val * 1.22)
    ax.grid(axis="y", alpha=0.4)

fig.savefig(OUT / "vs_stock.png")
plt.close(fig)

# ---- figure: QUAIL-B SF=0.1, cold vs warm ---------------------------

suite = load("quailb_sf0.1_kvwrite.json")
cold = {q["query"]: q for q in suite["passes"]["cold"]["queries"]}
warm = {q["query"]: q for q in suite["passes"]["warm"]["queries"]}
qids = list(cold)
y = range(len(qids))
cw = [cold[q]["wall_s"] for q in qids]
ww = [warm[q]["wall_s"] for q in qids]
restored = [sum(s.get("restored_docs", 0)
                for s in (warm[q].get("store") or {}).values())
            for q in qids]

fig, ax = plt.subplots(figsize=(10, 7.2))
h = 0.36
ax.barh([i - h / 2 for i in y], cw, height=h, color=STOCK,
        label="Cold (no store)", edgecolor="white", linewidth=0.5)
ax.barh([i + h / 2 for i in y], ww, height=h, color=QUAIL,
        label="Warm (store)", edgecolor="white", linewidth=0.5)
for i, q in enumerate(qids):
    ax.text(cw[i] * 1.05, i - h / 2, f"{cw[i]:.0f} s",
            va="center", fontsize=7.5, color="#777777")
    label = f"{ww[i]:.0f} s"
    if restored[i]:
        label += f"  ({restored[i]} restored)"
    ax.text(ww[i] * 1.05, i + h / 2, label,
            va="center", fontsize=7.5, color=QUAIL, fontweight="medium")
ax.set_yticks(list(y))
descs = [f"{q}: {cold[q]['desc']}" for q in qids]
ax.set_yticklabels(descs, fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(8, 3000)
ax.set_xlabel("wall time (s, log scale)")
ax.set_title(
    f"QUAIL-B SF=0.1  ·  cold {suite['passes']['cold']['pass_wall_s']:.0f} s"
    f"  ·  warm {suite['passes']['warm']['pass_wall_s']:.0f} s",
    fontsize=11, fontweight="bold", loc="left")
ax.legend(loc="lower right", fontsize=9, framealpha=0)
ax.grid(axis="x", alpha=0.3)
fig.savefig(OUT / "quailb_cold_warm.png")
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
