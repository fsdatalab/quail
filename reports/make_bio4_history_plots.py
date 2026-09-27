r"""Plot BIO-4 time and tokens for each dated configuration of Quail's history.

Pull each run directory's configuration summaries and startup samples
off the volume (the run directories are in the report), then run this
script on them. A configuration in a later directory replaces the same
configuration in an earlier one:

    for R in <run directory> ...; do
      W=/tmp/bio4-history/$(basename "$R"); mkdir -p "$W"
      uv run modal volume get quail-results "ablations/$R/configurations" \
        "$W" --force
      uv run modal volume get quail-results "ablations/$R/startup" "$W" --force
    done
    uv run --with matplotlib python reports/make_bio4_history_plots.py \
      /tmp/bio4-history/<first> /tmp/bio4-history/<second>

Pass --table to print the report's tables instead of drawing. Startup
seconds are the median of the configuration's startup samples, the one
from its query run included.

Writes three figures to plots/: bio4_history_versus_vllm.png (startup
and query seconds of the vLLM baselines, Quail's first engine, and Quail
today), bio4_history_steps.png (the same for every Quail configuration
in merge order, with each step's change), and bio4_history_tokens.png
(fresh input tokens, with the recomputed KV part marked, above each
configuration's answer agreement). Speedups and
percentages are derived here from the saved summaries.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, LIGHT_GRAY, ORANGE, RED  # noqa: E402

# bar order: the vLLM baselines, then Quail's steps in merge order
ORDER = (
    "vllm-defaults", "vllm-tuned", "vllm-pipelined", "vllm-today",
    "quail-engine-vllm-kernels", "quail-engine", "pinned_staging",
    "attention_paths", "skip_arena_writes", "join_search", "compile_once",
    "filter_kv_reuse", "scan_ring", "boot_cache",
    "shared_retention", "join_continuous_batching", "projection_pushdown",
    "plan_on_estimates", "filter_join_streaming", "gigatoken",
)
SHORT = {
    "vllm-defaults": "vLLM defaults",
    "vllm-tuned": "vLLM tuned",
    "vllm-pipelined": "vLLM pipelined",
    "vllm-today": "vLLM pipelined + Gigatoken",
    "quail-engine-vllm-kernels": "first engine, vLLM kernels",
    "quail-engine": "fused kernels",
    "pinned_staging": "pinned copies",
    "attention_paths": "attention paths",
    "skip_arena_writes": "skip KV writes",
    "join_search": "join search",
    "compile_once": "compile once",
    "filter_kv_reuse": "filter KV reuse",
    "scan_ring": "scan ring",
    "boot_cache": "boot cache",
    "shared_retention": "shared retention",
    "join_continuous_batching": "join batching",
    "projection_pushdown": "projection pushdown",
    "plan_on_estimates": "plan on estimates",
    "filter_join_streaming": "filter-join streaming",
    "gigatoken": "Gigatoken",
}


def load(workdirs) -> dict:
    """Name -> summary for every saved configuration that completed.

    A configuration in a later workdir replaces one in an earlier
    workdir. Each summary gains `startup_samples`, the startup seconds
    measured in the summary's own workdir, and `startup_median_s`.
    """
    summaries = {}
    for workdir in workdirs:
        found = {}
        for path in sorted((workdir / "configurations").glob("*.json")):
            summary = json.loads(path.read_text())
            if summary["status"] == "complete" and summary["name"] in SHORT:
                summary["startup_samples"] = [summary["startup"]["startup_s"]]
                found[summary["name"]] = summary
        for path in sorted((workdir / "startup").glob("*.json")):
            sample = json.loads(path.read_text())
            if sample["name"] in found:
                found[sample["name"]]["startup_samples"].append(
                    sample["startup_s"])
        summaries.update(found)
    for summary in summaries.values():
        summary["startup_median_s"] = float(
            np.median(summary["startup_samples"]))
    return summaries


def when(summary) -> datetime:
    return datetime.fromisoformat(summary["as_of"].replace("Z", "+00:00"))


def seconds(summary) -> tuple[float, float]:
    """(query seconds, startup seconds) of one configuration."""
    return summary["metrics"]["runtime_s"], summary["startup_median_s"]


def tick(summary) -> str:
    pr = summary.get("pull_request")
    number = f" #{pr}" if pr else ""
    return f"{SHORT[summary['name']]}\n{when(summary):%b %-d}{number}"


def agreement(summary) -> float:
    """Share of predicate answers that match the reference labels."""
    return summary["metrics"]["accuracy"]["answer_accuracy"]["accuracy"]


def label(summary) -> str:
    """Row label: what the configuration adds, its date, and its PR."""
    pr = summary.get("pull_request")
    number = f", #{pr}" if pr else ""
    return f"{SHORT[summary['name']]} ({when(summary):%b %-d}{number})"


VERSUS_LABELS = {
    "quail-engine-vllm-kernels": "Quail, first engine (Aug 18)",
    "gigatoken": "Quail today (code of Sep 16)",
}


def stacked_rows(ax, rows, colors, text, labels=None):
    """Horizontal query bars with startup stacked after them, top to bottom."""
    y = np.arange(len(rows))
    query = np.array([seconds(r)[0] for r in rows])
    startup = np.array([seconds(r)[1] for r in rows])
    ax.barh(y, query, color=colors, height=0.62)
    ax.barh(y, startup, left=query, color=LIGHT_GRAY, height=0.62,
            edgecolor=GRAY, linewidth=0.5)
    for i, row in enumerate(rows):
        ax.text(query[i] + startup[i], i, "  " + text(row), va="center",
                fontsize=9, color=DARK)
    ax.set_yticks(y, labels or [label(r) for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    top = max(query + startup)
    ax.set_xticks([t for t in MaxNLocator(nbins=6).tick_values(0, top)
                   if t <= top * 1.05])
    ax.xaxis.grid(True, color=LIGHT_GRAY, linewidth=0.6)
    ax.set_axisbelow(True)


def versus_plot(summaries, names, out):
    """The vLLM baselines, Quail's first engine, and Quail today."""
    picks = [n for n in names if summaries[n]["backend"] != "quail"]
    picks += [n for n in ("quail-engine-vllm-kernels", "gigatoken")
              if n in summaries]
    rows = [summaries[n] for n in picks]
    colors = [ORANGE if r["backend"] != "quail" else BLUE for r in rows]
    vllm_today = summaries.get("vllm-today")

    def text(row):
        words = (f"{sum(seconds(row)):,.0f} s  ({seconds(row)[0]:,.0f} query"
                 f" + {seconds(row)[1]:,.0f} startup)")
        if row["backend"] == "quail" and vllm_today:
            ratio = sum(seconds(vllm_today)) / sum(seconds(row))
            words += f", {ratio:.1f}x less than vLLM today"
        return words

    fig, ax = plt.subplots(figsize=(12, 4.6))
    stacked_rows(ax, rows, colors, text, labels=[
        VERSUS_LABELS.get(r["name"], label(r)) for r in rows])
    ax.set_xlim(0, max(sum(seconds(r)) for r in rows) * 1.45)
    ax.set_title("BIO-4 at sf=0.5, startup + query: the vLLM baselines "
                 "(orange) and Quail (blue)")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def steps_plot(summaries, names, out):
    """Quail's configurations in merge order, with each step's change."""
    rows = [summaries[n] for n in names if summaries[n]["backend"] == "quail"]

    def text(row):
        q, s = seconds(row)
        words = f"{q:,.0f} query + {s:,.0f} startup"
        i = rows.index(row)
        if i:
            before = seconds(rows[i - 1])
            dq, ds = q / before[0] - 1, s / before[1] - 1
            changes = []
            if abs(dq) >= 0.03:
                changes.append(f"query {dq:+.0%}")
            if abs(ds) >= 0.15:
                changes.append(f"startup {ds:+.0%}")
            if changes:
                words += "   \u2190 " + ", ".join(changes)
        return words

    fig, ax = plt.subplots(figsize=(12, 7.5))
    stacked_rows(ax, rows, [BLUE] * len(rows), text)
    ax.set_xlim(0, max(sum(seconds(r)) for r in rows) * 1.6)
    ax.set_title("BIO-4 at sf=0.5 on Quail's code at each merge date: "
                 "query seconds (blue) + startup seconds (gray)")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def token_plot(summaries, names, out):
    fresh = np.array([summaries[n]["metrics"]["fresh_tokens"] or 0
                      for n in names], dtype=float)
    regret = np.array([summaries[n]["metrics"].get("regret_tokens") or 0
                       for n in names], dtype=float)
    agree = np.array([100 * agreement(summaries[n]) for n in names])
    x = np.arange(len(names))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(15, 9.5), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 1.3]})
    top.bar(x, (fresh - regret) / 1e6, color=BLUE, width=0.7)
    top.bar(x, regret / 1e6, bottom=(fresh - regret) / 1e6, color=RED,
            width=0.7)
    for i, value in enumerate(fresh):
        top.text(i, value / 1e6, f"{value / 1e6:.1f}", ha="center",
                 va="bottom", fontsize=7.5, color=DARK)
    top.set_ylabel("millions of tokens")
    top.set_title("BIO-4: fresh input tokens (blue) and the recomputed KV "
                  "tokens among them (red)")
    colors = [ORANGE if summaries[n]["backend"] != "quail" else BLUE
              for n in names]
    bottom.bar(x, agree, color=colors, width=0.7)
    for i, value in enumerate(agree):
        bottom.text(i, value, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=DARK)
    bottom.set_ylim(0, 105)
    bottom.set_ylabel("percent")
    bottom.set_title("Answer agreement with the reference labels")
    bottom.set_xticks(x, [tick(summaries[n]) for n in names], rotation=90,
                      fontsize=8)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def tables(summaries, names):
    """Print the report's time, token, and accuracy tables as markdown."""
    print("| Configuration | Merged | Query s | Startup s (median, range) "
          "| Input tokens/s | $/query |")
    print("|---|---|---:|---:|---:|---:|")
    for n in names:
        s = summaries[n]
        m = s["metrics"]
        samples = s["startup_samples"]
        pr = f" #{s['pull_request']}" if s.get("pull_request") else ""
        print(f"| {SHORT[n]} | {when(s):%b %-d}{pr} | {m['runtime_s']:,.1f} "
              f"| {s['startup_median_s']:,.1f} ({min(samples):,.1f} to "
              f"{max(samples):,.1f}, n={len(samples)}) "
              f"| {m['input_tokens_per_second']:,.0f} | {m['cost_usd']:.4f} |")
    print()
    print("| Configuration | Fresh tokens | Recomputed KV tokens "
          "| KV regret % | Evaluated pairs |")
    print("|---|---:|---:|---:|---:|")
    for n in names:
        m = summaries[n]["metrics"]
        print(f"| {SHORT[n]} | {m['fresh_tokens']:,} | {m['regret_tokens']:,} "
              f"| {m['kv_regret_percent']:.1f} "
              f"| {m['evaluated_document_pairs']:,} |")
    print()
    print("| Configuration | Answer agreement % | Returned rows "
          "| Output precision % | Output recall % |")
    print("|---|---:|---:|---:|---:|")
    for n in names:
        accuracy = summaries[n]["metrics"]["accuracy"]
        answers = accuracy["answer_accuracy"]
        rows = accuracy["output_accuracy"]
        print(f"| {SHORT[n]} | {100 * answers['accuracy']:.1f} "
              f"| {rows['predicted_rows']:,} | {100 * rows['precision']:.2f} "
              f"| {100 * rows['recall']:.2f} |")


def main():
    workdirs = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("--")]
    summaries = load(workdirs)
    names = [n for n in ORDER if n in summaries]
    missing = [n for n in ORDER if n not in summaries]
    if missing:
        print(f"not measured: {', '.join(missing)}")
    if "--table" in sys.argv[2:]:
        tables(summaries, names)
        return
    OUT.mkdir(exist_ok=True)
    versus_plot(summaries, names, OUT / "bio4_history_versus_vllm.png")
    steps_plot(summaries, names, OUT / "bio4_history_steps.png")
    token_plot(summaries, names, OUT / "bio4_history_tokens.png")
    print(f"wrote three figures to {OUT}")


if __name__ == "__main__":
    main()
