"""Plots for reports/2026-08-26-bio6-full-terms-and-judge-identity.md.

BIO-6's two join stages before and after the severe_terms revert, and
what relabelling under the current join prompts did to every join
predicate's TRUE rate.

Pull the inputs into a workdir first (the volume lives in the
fsdatalab Modal workspace):

    W=$(mktemp -d)
    C=/ground_truth/quailb/schema_v1/collections
    A=gt_04231c5de83cdf9e7e68fc03849959d6
    modal volume get quail-results \
        $C/gt_306dac4fc83883c7a5bcc86f4d103f32/summary.json $W/before.json
    modal volume get quail-results $C/$A/summary.json $W/after.json
    modal volume get quail-results \
        /ablations/join_relabel_diff_$A.json $W/relabel_diff.json
    uv run --with matplotlib python reports/make_bio6_full_terms_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, ORANGE

W = Path(sys.argv[1])
with open(W / "before.json") as f:
    before = json.load(f)["label_sets"]
with open(W / "after.json") as f:
    after = json.load(f)["label_sets"]

J1 = "quailb.biodex.report.experienced_reaction"
J2 = "quailb.biodex.report.experienced_severe_reaction"

# ---- BIO-6's two stages, before and after -----------------------------
stages = ["J1\nREACTION", "J2\nREACTION_SEVERE"]
was = [before[J1]["rows"], before[J2]["rows"]]
now = [after[J1]["rows"], after[J2]["rows"]]

fig, ax = plt.subplots(figsize=(6.0, 3.6))
x = range(len(stages))
width = 0.3
ax.bar([i - width / 2 for i in x], was, width, color=GRAY,
       label="with severe_terms (64 terms)")
ax.bar([i + width / 2 for i in x], now, width, color=BLUE,
       label="full terms table (614 terms)")
for i, (a, b) in enumerate(zip(was, now)):
    ax.text(i - width / 2, a, f"{a:,}", ha="center", va="bottom")
    ax.text(i + width / 2, b, f"{b:,}", ha="center", va="bottom")
growth = now[1] / was[1]
ax.annotate(f"{growth:.0f}x more pairs in the second stage",
            xy=(1 + width / 2, now[1]), xytext=(0.45, now[1] * 1.06),
            ha="left", va="bottom", color=BLUE)
ax.set_xticks(list(x))
ax.set_xticklabels(stages)
ax.set_xlim(-0.6, len(stages) - 0.4)
ax.set_ylabel("report-term pairs judged")
ax.set_ylim(0, max(now) * 1.30)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{int(v):,}"))
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
path = OUT / "20260826-bio6-stage-sizes.png"
fig.savefig(path, dpi=150)
print(f"wrote {path}")

# ---- what relabelling actually changed --------------------------------
with open(W / "relabel_diff.json") as f:
    diff = json.load(f)["predicates"]

compared = {code: d for code, d in diff.items() if d["compared"]}
order = sorted(compared, key=lambda c: compared[c]["answer_differences"]
               / compared[c]["pairs"])

fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(order) + 1.7))
y = range(len(order))
rates = [100.0 * compared[c]["answer_differences"] / compared[c]["pairs"]
         for c in order]
ax.barh(list(y), rates, color=ORANGE, height=0.55)
for i, code in enumerate(order):
    d = compared[code]
    ax.text(rates[i] + max(rates) * 0.02, i,
            f"{d['answer_differences']:,} of {d['pairs']:,}",
            va="center")
ax.set_yticks(list(y))
ax.set_yticklabels(order)
ax.set_xlim(0, max(rates) * 1.45)
ax.set_xlabel("answers that changed when the predicate was relabelled "
              "(% of pairs)")
fig.tight_layout()
path = OUT / "20260826-join-relabel-answer-diffs.png"
fig.savefig(path, dpi=150)
print(f"wrote {path}")
