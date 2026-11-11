"""Summarize the position-bias records into tables and figures.

Reads results/position_bias/{model}/{task}.json and simulation.json,
and writes summary.json, swap.png, topk.png, and simulation.png beside
them.

    uv run python -m experiments.position_bias.analyze
"""

import json
import statistics
from pathlib import Path

ROOT = Path("results/position_bias")
TASK_NAMES = ("sales", "support", "legal")
TASK_TITLES = {"sales": "Sales", "support": "Customer Support", "legal": "Legal"}
# (model, short label, family), in plotting order
MODELS = (
    ("decision-2.0-kai-0.6b-bf16", "Kai 0.6B", "decision"),
    ("qwen3-0.6b-bf16", "Qwen3 0.6B", "generative"),
    ("d1-2b-bf16", "d1 2B", "decision"),
    ("lfm2-2.6b-bf16", "LFM2 2.6B", "generative"),
    ("qwen3-4b-fp8", "Qwen3 4B", "generative"),
    ("qwen3.6-35b-a3b-fp8", "Qwen3.6 35B-A3B", "generative"),
    ("qwen3.8-27b-fp8", "Qwen3.8 27B", "generative"),
    ("jev-1.13.0", "Jev", "api"),
)
RULE_TITLES = {"pointwise": "Pointwise", "raw": "Pairwise, raw",
               "calibrated": "Pairwise, calibrated",
               "both_orders": "Pairwise, both orders"}
# the reference palette's categorical slots 1 to 4, light mode
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FAMILY_COLORS = {"generative": SERIES[0], "decision": SERIES[1], "api": SERIES[2]}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"


def load_records() -> dict:
    """(model, task) -> record, for every record present."""
    records = {}
    for model, _, _ in MODELS:
        for task in TASK_NAMES:
            path = ROOT / model / f"{task}.json"
            if path.exists():
                records[model, task] = json.loads(path.read_text())
    return records


def summarize(record: dict) -> dict:
    """One record's headline numbers."""
    swap = record["swap"]["summary"]
    row = {"pointwise": record["pointwise"]["precision"],
           "pointwise_seconds": record["pointwise"]["seconds"],
           "mixed_same_slot": swap["mixed"]["same_slot_rate"],
           "mixed_first_both": swap["mixed"]["first_slot_both_rate"],
           "mixed_accuracy": swap["mixed"]["accuracy"],
           "tied_same_slot": swap["tied"]["same_slot_rate"],
           "bias": swap["all"]["bias"], "bias_stderr": swap["all"]["bias_stderr"],
           "pair_seconds": record.get("pair_seconds")}
    for rule in ("raw", "calibrated", "both_orders"):
        runs = [entry for entry in record["quickselect"] if entry["rule"] == rule]
        values = [entry["precision"] for entry in runs]
        row[rule] = statistics.mean(values)
        row[rule + "_min"], row[rule + "_max"] = min(values), max(values)
        row[rule + "_comparisons"] = statistics.mean(
            entry["num_comparisons"] for entry in runs)
    return row


def _axes_style(axes):
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def plot_swap(summary, path):
    """Same-slot rate on mixed pairs, one panel per task."""
    import matplotlib.pyplot as plt

    figure, panels = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                                  facecolor=SURFACE)
    for axes, task in zip(panels, TASK_NAMES):
        _axes_style(axes)
        present = [(label, family, summary[model][task])
                   for model, label, family in MODELS
                   if task in summary.get(model, {})]
        positions = range(len(present))
        axes.bar(positions, [row["mixed_same_slot"] for _, _, row in present],
                 width=0.7, color=[FAMILY_COLORS[family] for _, family, _ in present],
                 edgecolor=SURFACE, linewidth=2)
        for position, (_, _, row) in zip(positions, present):
            axes.text(position, row["mixed_same_slot"] + 0.02,
                      f"{row['mixed_same_slot']:.0%}", ha="center",
                      fontsize=7, color=TEXT_PRIMARY)
        axes.set_xticks(list(positions), [label for label, _, _ in present],
                        rotation=40, ha="right")
        axes.set_title(TASK_TITLES[task], fontsize=10, color=TEXT_PRIMARY)
        axes.set_ylim(0, 1.08)
    panels[0].set_ylabel("Same slot in both orders\n(pairs with different labels)",
                         fontsize=8, color=TEXT_SECONDARY)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color)
               for color in FAMILY_COLORS.values()]
    figure.legend(handles, ["Generative", "Decision", "API (Jev)"],
                  loc="upper right", fontsize=8, frameon=False, ncols=3)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160, facecolor=SURFACE)


def plot_topk(summary, path):
    """Top-k precision under pointwise scoring and each pairwise rule."""
    import matplotlib.pyplot as plt

    rules = ("pointwise", "raw", "calibrated", "both_orders")
    figure, panels = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                                  facecolor=SURFACE)
    width = 0.2
    for axes, task in zip(panels, TASK_NAMES):
        _axes_style(axes)
        present = [(label, summary[model][task]) for model, label, _ in MODELS
                   if task in summary.get(model, {})]
        for offset, (rule, color) in enumerate(zip(rules, SERIES)):
            xs = [index + (offset - 1.5) * width for index in range(len(present))]
            values = [row[rule] for _, row in present]
            axes.bar(xs, values, width=width, color=color, edgecolor=SURFACE,
                     linewidth=1.5, label=RULE_TITLES[rule])
            if rule != "pointwise":
                lows = [row[rule] - row[rule + "_min"] for _, row in present]
                highs = [row[rule + "_max"] - row[rule] for _, row in present]
                axes.errorbar(xs, values, yerr=[lows, highs], fmt="none",
                              ecolor=TEXT_SECONDARY, elinewidth=0.8, capsize=1.5)
        axes.set_ylim(0, 1.05)
        axes.set_ylabel(f"{TASK_TITLES[task]}\ntop-k precision", fontsize=8,
                        color=TEXT_SECONDARY)
        axes.set_xticks(range(len(present)), [label for label, _ in present])
    panels[0].legend(loc="upper left", fontsize=8, frameon=False, ncols=4,
                     bbox_to_anchor=(0, 1.25))
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)


def plot_simulation(summary, path):
    """Raw quickselect precision against the same-slot rate, simulated and measured."""
    import matplotlib.pyplot as plt

    curves = json.loads((ROOT / "simulation.json").read_text())
    figure, panels = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True,
                                  facecolor=SURFACE)
    for axes, task in zip(panels, TASK_NAMES):
        _axes_style(axes)
        for accuracy, style in ((1.0, "-"), (0.9, "--")):
            rows = [row for row in curves[task] if row["rule"] == "raw"
                    and row["content_accuracy"] == accuracy]
            axes.plot([row["position_rate"] for row in rows],
                      [row["precision"] for row in rows], style,
                      color=TEXT_SECONDARY, linewidth=2,
                      label=f"simulated, {accuracy:.0%} right on content")
        for model, label, family in MODELS:
            row = summary.get(model, {}).get(task)
            if row is None:
                continue
            axes.scatter(row["mixed_same_slot"], row["raw"], s=40, zorder=3,
                         color=FAMILY_COLORS[family], edgecolors=SURFACE,
                         linewidths=2)
            axes.annotate(label, (row["mixed_same_slot"], row["raw"]),
                          textcoords="offset points", xytext=(5, 4),
                          fontsize=7, color=TEXT_PRIMARY)
        axes.set_title(TASK_TITLES[task], fontsize=10, color=TEXT_PRIMARY)
        axes.set_xlabel("Same slot in both orders (different-label pairs)",
                        fontsize=8, color=TEXT_SECONDARY)
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1.05)
    panels[0].set_ylabel("Raw pairwise top-k precision", fontsize=8,
                         color=TEXT_SECONDARY)
    panels[0].legend(fontsize=7, frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE)


def main():
    """Write summary.json and the three figures."""
    records = load_records()
    summary = {}
    for (model, task), record in records.items():
        summary.setdefault(model, {})[task] = summarize(record)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    plot_swap(summary, ROOT / "swap.png")
    plot_topk(summary, ROOT / "topk.png")
    plot_simulation(summary, ROOT / "simulation.png")
    for model, label, _ in MODELS:
        for task in TASK_NAMES:
            row = summary.get(model, {}).get(task)
            if row:
                print(f"{label:16} {task:8} pointwise {row['pointwise']:.2f} "
                      f"raw {row['raw']:.2f} calibrated {row['calibrated']:.2f} "
                      f"both {row['both_orders']:.2f} same-slot "
                      f"{row['mixed_same_slot']:.2f} bias {row['bias']:+.3f}")


if __name__ == "__main__":
    main()
