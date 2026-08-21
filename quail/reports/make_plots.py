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


# ---- figure 2: quail vs stock vLLM ------------------------------------

post = load("m1_filter.json")["runs"]
stockf = load("baseline_filter4.json")["runs"]
quail_filter = sum(r["wall"] for r in post) / len(post)
stock_filter = sum(r["wall"] for r in stockf) / len(stockf)

stockj_data = load("baseline_join.json")["runs"]
stockj = sum(r["wall"] for r in stockj_data) / len(stockj_data)
quail_join = load("dispatch_gate.json")["gpus1"]["join"]["wall_s"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
labels = ["Stock vLLM\n(tuned)", "Quail"]
vals = [stock_filter, quail_filter]
cols = [GRAY, GREEN]
bars = a1.bar(labels, vals, color=cols, width=0.5)
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f} s",
            ha="center", fontsize=10)
a1.set_ylabel("wall time (seconds, lower is better)")
a1.set_title("5-filter chain, 10,000 documents", fontsize=11)
a1.set_ylim(0, 46)
a1.text(1, quail_filter + 4, f"{stock_filter / quail_filter:.2f}x faster",
        ha="center", fontsize=10, color=GREEN)

labels = ["Stock vLLM\n(grouped)", "Quail"]
vals = [stockj, quail_join]
bars = a2.bar(labels, vals, color=[GRAY, GREEN], width=0.5)
for b, v in zip(bars, vals):
    a2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f} s",
            ha="center", fontsize=10)
a2.set_title("72,000-pair join", fontsize=11)
a2.set_ylabel("wall time (seconds, lower is better)")
a2.set_ylim(0, 85)
a2.text(1, quail_join + 3, f"{stockj / quail_join:.1f}x faster",
        ha="center", fontsize=10, color=GREEN)
fig.savefig(OUT / "vs_stock.png")
plt.close(fig)

# ---- figure 3: QUAIL-B SF=0.1, cold vs warm -------------------------

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

fig, ax = plt.subplots(figsize=(9, 6.8))
h = 0.38
ax.barh([i - h / 2 for i in y], cw, height=h, color=GRAY,
        label="Cold pass (store disabled)")
ax.barh([i + h / 2 for i in y], ww, height=h, color=BLUE,
        label="Warm pass (store enabled)")
for i, q in enumerate(qids):
    ax.text(cw[i] * 1.04, i - h / 2, f"{cw[i]:.0f} s",
            va="center", fontsize=7.5, color=DARK)
    label = f"{ww[i]:.0f} s"
    if restored[i]:
        label += f" ({restored[i]} docs restored)"
    ax.text(ww[i] * 1.04, i + h / 2, label,
            va="center", fontsize=7.5, color=BLUE)
ax.set_yticks(list(y))
descs = [f"{q}: {cold[q]['desc']}" for q in qids]
ax.set_yticklabels(descs, fontsize=8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(8, 3000)
ax.set_xlabel("wall time (seconds, log scale)")
ax.set_title(
    f"QUAIL-B at SF=0.1: cold pass {suite['passes']['cold']['pass_wall_s']:.0f} s, "
    f"warm pass {suite['passes']['warm']['pass_wall_s']:.0f} s",
    fontsize=11)
ax.legend(loc="lower right", fontsize=9)
fig.savefig(OUT / "quailb_cold_warm.png")
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
