r"""Plot FEV-10 (join over pairs) against FEV-5 and FEV-9.

Pull the run directory off the volume (the stamp is in the report),
then run this script on it:

    W=/tmp/quail-pair-join; mkdir -p "$W"
    uv run modal volume get quail-results \
      benchmarks/quailb/<stamp>-pair-join "$W" --force
    uv run modal volume get quail-results \
      sol/2026-09-11-fev10-prefix-reuse.json "$W/sol.json" --force
    uv run --with matplotlib python reports/make_pair_join_plots.py \
      "$W/<stamp>-pair-join" "$W/sol.json"

Writes plots/pair_join.png: query seconds, evaluated pairs, and fresh
input tokens per query, with the FEV-5 cross join beside FEV-10 and
the speed of light estimate as a line across each bar. The reductions
are derived here from the saved run record.
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
from plot_colors import BLUE, DARK, GRAY  # noqa: E402

QUERIES = ("FEV-5", "FEV-10", "FEV-9")
FIELDS = (("seconds", "seconds", "Query time, excluding startup", "{:.2f}"),
          ("pairs", "thousands of pairs", "Evaluated document pairs",
           "{:.1f}"),
          ("fresh", "millions of tokens", "Fresh input tokens", "{:.2f}"))


def load(workdir):
    record = json.loads((workdir / "quail" / "fever" / "run.json").read_text())
    rows = {}
    for item in record["queries"]:
        if item["id"] not in QUERIES:
            continue    # FEV-1 only absorbs the cold boot
        metrics = item["metrics"]
        measurements = item["measurements"]
        rows[item["id"]] = dict(
            seconds=metrics["runtime_s"],
            pairs=metrics["evaluated_document_pairs"] / 1e3,
            fresh=measurements["fresh_tokens"] / 1e6,
            regret=measurements["regret_tokens"],
            agreement=metrics["accuracy"]["answer_accuracy"]["accuracy"],
            precision=metrics["accuracy"]["output_accuracy"]["precision"],
            recall=metrics["accuracy"]["output_accuracy"]["recall"],
            rows=metrics["accuracy"]["output_accuracy"]["predicted_rows"])
    return rows


def load_sol(path):
    """Per query, the SoL seconds, pairs, and tokens on Qwen3 4B fp8."""
    record = json.loads(Path(path).read_text())
    out = {}
    for query, entry in record["queries"].items():
        estimate = entry["models"]["qwen3-4b-fp8"]
        out[query] = dict(seconds=estimate["sol_s"],
                          pairs=estimate["join_pair_evaluations"] / 1e3,
                          fresh=estimate["tokens"] / 1e6)
    return out


def main(workdir, sol_path=None):
    rows = load(Path(workdir))
    sol = load_sol(sol_path) if sol_path else {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(QUERIES))
    for axis, (field, unit, title, fmt) in zip(axes, FIELDS):
        values = [rows[query][field] for query in QUERIES]
        colors = [GRAY if query == "FEV-5" else BLUE for query in QUERIES]
        bars = axis.bar(x, values, color=colors, width=0.6)
        for bar, value in zip(bars, values):
            axis.annotate(fmt.format(value),
                          (bar.get_x() + bar.get_width() / 2, value),
                          ha="center", va="bottom", xytext=(0, 3),
                          textcoords="offset points", fontsize=9)
        for bar, query in zip(bars, QUERIES):
            if query not in sol:
                continue
            estimate = sol[query][field]
            axis.hlines(estimate, bar.get_x() - 0.08,
                        bar.get_x() + bar.get_width() + 0.08,
                        color=DARK, linewidth=1.2)
            axis.annotate(f"SoL {fmt.format(estimate)}",
                          (bar.get_x() + bar.get_width() + 0.1, estimate),
                          ha="left", va="center", fontsize=8, color=DARK)
        cross, paired = rows["FEV-5"][field], rows["FEV-10"][field]
        if cross:
            axis.annotate(f"FEV-10 is {100 * (1 - paired / cross):.1f}% below "
                          f"FEV-5", (0.5, max(values)),
                          ha="center", va="bottom", xytext=(0, 14),
                          textcoords="offset points", fontsize=9)
        axis.set_xticks(x)
        axis.set_xticklabels(QUERIES)
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.set_ylim(0, max(values) * 1.25)
    fig.suptitle("FEV-10 asks SUPPORT of a claim and its own page only; "
                 "FEV-5 is the same query over every pair")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "pair_join.png", dpi=300)
    print(f"wrote {OUT / 'pair_join.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
