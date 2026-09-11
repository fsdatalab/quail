r"""Plot FEV-10 paired by the equality, by apply(), and by apply_table().

Pull the comparison directory off the volume (the stamp is in the
report), then run this script on it:

    W=/tmp/quail-foreign-pairs; mkdir -p "$W"
    uv run modal volume get quail-results \
      ablations/foreign-pairs-<stamp> "$W" --force
    uv run --with matplotlib python reports/make_foreign_operator_plots.py \
      "$W/foreign-pairs-<stamp>"

Writes plots/foreign_operator.png: query seconds, recomputed KV
tokens, and fresh input tokens per variant. Differences against the
equality run are derived here from the saved summaries.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, ORANGE  # noqa: E402

VARIANTS = (("equality", "equality\njoin(on=...)", GRAY),
            ("per_batch", "per batch\napply(fn)", BLUE),
            ("barrier", "barrier\napply_table(fn)", ORANGE))
FIELDS = (("seconds", "seconds", "Query time, excluding startup", "{:.2f}"),
          ("regret", "thousands of tokens", "Recomputed KV tokens", "{:.1f}"),
          ("fresh", "thousands of tokens", "Fresh input tokens", "{:.1f}"))


def load(workdir):
    rows = {}
    for key, _label, _color in VARIANTS:
        summary = json.loads((workdir / key / "summary.json").read_text())
        report = summary["report"]
        rows[key] = dict(
            seconds=summary["query_seconds"],
            regret=report["regret_tokens"] / 1e3,
            fresh=report["fresh_tokens"] / 1e3,
            pairs=summary["evaluated_document_pairs"],
            rows=summary["rows"])
    return rows


def main(workdir):
    rows = load(Path(workdir))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(VARIANTS))
    for axis, (field, unit, title, fmt) in zip(axes, FIELDS):
        values = [rows[key][field] for key, _, _ in VARIANTS]
        bars = axis.bar(x, values, color=[c for _, _, c in VARIANTS],
                        width=0.6)
        for bar, value in zip(bars, values):
            axis.annotate(fmt.format(value),
                          (bar.get_x() + bar.get_width() / 2, value),
                          ha="center", va="bottom", xytext=(0, 3),
                          textcoords="offset points", fontsize=9)
        base = rows["equality"][field]
        for bar, (key, _, _) in zip(bars, VARIANTS):
            if key == "equality":
                continue
            delta = rows[key][field] - base
            sign = "+" if delta >= 0 else ""
            axis.annotate(f"{sign}{fmt.format(delta)} vs equality",
                          (bar.get_x() + bar.get_width() / 2, rows[key][field]),
                          ha="center", va="bottom", xytext=(0, 14),
                          textcoords="offset points", fontsize=8)
        axis.set_xticks(x)
        axis.set_xticklabels([label for _, label, _ in VARIANTS],
                             fontsize=8)
        axis.set_ylabel(unit)
        axis.set_title(title)
        # an all-zero panel still gets a readable axis
        axis.set_ylim(0, max(max(values), 1.0) * 1.3)
    fig.suptitle("FEV-10: the same-page pairing by the built-in equality, "
                 "by a per-batch apply(), and by a barrier apply_table()")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "foreign_operator.png", dpi=300)
    print(f"wrote {OUT / 'foreign_operator.png'}")


if __name__ == "__main__":
    main(sys.argv[1])
