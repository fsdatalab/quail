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

Writes three figures to plots/: bio4_history_progress.png (end-to-end
and query seconds of Quail against the merge date, with the vLLM
baselines as points), bio4_history_time.png (startup and query seconds
stacked, one bar per configuration in merge order), and
bio4_history_tokens.png (fresh input tokens, with the recomputed KV part
marked, above each configuration's answer agreement). Speedups and
percentages are derived here from the saved summaries.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

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
    "vllm-today": "vLLM + Gigatoken",
    "quail-engine-vllm-kernels": "Quail engine",
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


def progress_plot(summaries, names, out, notable=0.05):
    """End-to-end and query seconds against merge date.

    A Quail step is labeled when it changes end-to-end or query seconds
    by more than `notable` of the previous step's value; the label adds
    the answer agreement when the step moves it by 2 points or more.
    The y-axis turns logarithmic when the values span more than 10x.
    """
    quail = [n for n in names if summaries[n]["backend"] == "quail"]
    dates = [when(summaries[n]) for n in quail]
    query = np.array([seconds(summaries[n])[0] for n in quail])
    total = query + np.array([seconds(summaries[n])[1] for n in quail])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.step(dates, total, where="post", color=DARK, linewidth=1.6)
    ax.step(dates, query, where="post", color=BLUE, linewidth=1.6)
    ax.plot(dates, total, "o", color=DARK, markersize=4)
    ax.plot(dates, query, "o", color=BLUE, markersize=4)
    ax.text(dates[-1], total[-1], "  startup + query", color=DARK,
            fontsize=9, va="bottom")
    ax.text(dates[-1], query[-1], "  query", color=BLUE, fontsize=9,
            va="top")
    for i in range(1, len(quail)):
        before = (total[i - 1], query[i - 1])
        after = (total[i], query[i])
        change = max(abs(a - b) / b for a, b in zip(after, before))
        if change <= notable:
            continue
        text = (f"{SHORT[quail[i]]} (#{summaries[quail[i]]['pull_request']})\n"
                f"end to end {after[0] / before[0] - 1:+.0%}, "
                f"query {after[1] / before[1] - 1:+.0%}")
        was, now = (agreement(summaries[quail[i - 1]]),
                    agreement(summaries[quail[i]]))
        if abs(now - was) >= 0.02:
            text += f"\nanswer agreement {was:.1%} to {now:.1%}"
        ax.annotate(text, xy=(dates[i], total[i]),
                    xytext=(6, 26 if i % 2 else 50), textcoords="offset points",
                    fontsize=8, color=DARK,
                    arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.6))
    for n in names:
        if summaries[n]["backend"] == "quail":
            continue
        q, s = seconds(summaries[n])
        ax.plot([when(summaries[n])], [q + s], "D", color=ORANGE,
                markersize=6)
        ax.annotate(f"{SHORT[n]}\n{q + s:,.0f} s", xy=(when(summaries[n]),
                    q + s), xytext=(6, -4), textcoords="offset points",
                    fontsize=8, color=ORANGE, va="top")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    everything = [sum(seconds(summaries[n])) for n in names] + list(query)
    if max(everything) / min(everything) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("seconds (log scale)")
    else:
        ax.set_ylabel("seconds")
        ax.set_ylim(bottom=0)
    ax.set_title("BIO-4 time on the Quail code of each merge date; "
                 "vLLM baselines as diamonds")
    fig.savefig(out, dpi=300)
    plt.close(fig)


def time_plot(summaries, names, out):
    query = np.array([seconds(summaries[n])[0] for n in names])
    startup = np.array([seconds(summaries[n])[1] for n in names])
    colors = [ORANGE if summaries[n]["backend"] != "quail" else BLUE
              for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(x, query, color=colors, width=0.7)
    ax.bar(x, startup, bottom=query, color=LIGHT_GRAY, width=0.7,
           edgecolor=GRAY, linewidth=0.5)
    for i, (q, s) in enumerate(zip(query, startup)):
        ax.text(i, q + s, f"{q:,.0f}\n+{s:,.0f}", ha="center", va="bottom",
                fontsize=7.5, color=DARK)
    ax.set_xticks(x, [tick(summaries[n]) for n in names], rotation=90,
                  fontsize=8)
    ax.set_ylabel("seconds")
    ax.set_title("BIO-4: query seconds (color) and startup seconds (gray) "
                 "per configuration, in merge order")
    first = next(i for i, n in enumerate(names)
                 if summaries[n]["backend"] == "quail")
    ax.axvline(first - 0.5, color=GRAY, linewidth=0.8)
    baseline = summaries.get("vllm-today")
    today = summaries.get("gigatoken")
    if baseline and today and "gigatoken" in names:
        speedup = seconds(baseline)[0] / seconds(today)[0]
        top = sum(seconds(today))
        ax.text(names.index("gigatoken"), top * 1.35,
                f"query {speedup:.1f}x\nfaster than\nvLLM + Gigatoken",
                ha="center", va="bottom", fontsize=8.5, color=DARK)
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
    progress_plot(summaries, names, OUT / "bio4_history_progress.png")
    time_plot(summaries, names, OUT / "bio4_history_time.png")
    token_plot(summaries, names, OUT / "bio4_history_tokens.png")
    print(f"wrote three figures to {OUT}")


if __name__ == "__main__":
    main()
