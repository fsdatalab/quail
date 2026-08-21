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
from plot_colors import BLUE, GRAY, GREEN, DARK


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

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))

for ax, title, stock_val, quail_val, stock_strategy in [
    (a1, "5-filter chain, 10k docs", stock_filter, quail_filter,
     "Stock vLLM\nseparate requests"),
    (a2, "72k-pair join", stockj, quail_join,
     "Stock vLLM\nanchor-major order"),
]:
    speedup = stock_val / quail_val
    speedup_txt = (f"{speedup:.1f}x" if speedup >= 2
                   else f"{speedup:.2f}x")

    bars = ax.bar(
        [stock_strategy, "Quail"], [stock_val, quail_val],
        color=[GRAY, BLUE], width=0.45,
    )
    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        stock_val + stock_val * 0.03,
        f"{stock_val:.1f} s", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#777777",
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        quail_val + stock_val * 0.03,
        f"{quail_val:.1f} s  ({speedup_txt})",
        ha="center", va="bottom", fontsize=11,
        fontweight="bold", color=BLUE,
    )
    ax.set_title(title, fontsize=11, fontweight="normal", pad=10)
    ax.set_ylim(0, stock_val * 1.25)
    ax.set_yticks([])

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
h = 0.35
ax.barh([i - h / 2 for i in y], cw, height=h, color=GRAY,
        label="Cold pass (store disabled)")
ax.barh([i + h / 2 for i in y], ww, height=h, color=BLUE,
        label="Warm pass (store enabled)")
for i, q in enumerate(qids):
    x_end = max(cw[i], ww[i])
    ax.text(x_end * 1.12, i - h / 2, f"{cw[i]:.0f} s",
            va="center", fontsize=7.5, color="#999999")
    label = f"{ww[i]:.0f} s"
    if restored[i]:
        label += f"  ({restored[i]} restored)"
    ax.text(x_end * 1.12, i + h / 2, label,
            va="center", fontsize=7.5, color=BLUE)
ax.set_yticks(list(y))
descs = [f"{q}: {cold[q]['desc']}" for q in qids]
ax.set_yticklabels(descs, fontsize=8, color="#555555")
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(8, 4500)
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["10 s", "100 s", "1000 s"])
ax.set_title(
    f"QUAIL-B SF=0.1  ·  cold {suite['passes']['cold']['pass_wall_s']:.0f} s"
    f"  ·  warm {suite['passes']['warm']['pass_wall_s']:.0f} s",
    fontsize=11, fontweight="bold", loc="left")
ax.legend(loc="lower right", fontsize=9, framealpha=0)
fig.savefig(OUT / "quailb_cold_warm.png")
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
