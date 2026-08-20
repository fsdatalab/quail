"""Plots for the 2026-08-18 and 2026-08-19 reports (QUAIL-B, vs
stock).

Reads the measured JSON artifacts in results/ and writes PNGs into
reports/plots/. Run from the quail/ directory:

    uv run --with matplotlib python reports/make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)

ACCENT = "#2979FF"
GRAY = "#9E9E9E"
GREEN = "#43A047"
DARK = "#424242"


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
a1.spines[["top", "right"]].set_visible(False)

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
a2.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "vs_stock.png", dpi=150)
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
ax.barh([i + h / 2 for i in y], ww, height=h, color=ACCENT,
        label="Warm pass (store enabled)")
for i, q in enumerate(qids):
    ax.text(cw[i] * 1.04, i - h / 2, f"{cw[i]:.0f} s",
            va="center", fontsize=7.5, color=DARK)
    label = f"{ww[i]:.0f} s"
    if restored[i]:
        label += f" ({restored[i]} docs restored)"
    ax.text(ww[i] * 1.04, i + h / 2, label,
            va="center", fontsize=7.5, color=ACCENT)
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
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "quailb_cold_warm.png", dpi=150)
plt.close(fig)

print("wrote", *[p.name for p in sorted(OUT.glob("*.png"))])
