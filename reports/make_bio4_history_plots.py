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

Writes three figures to plots/, each by date: bio4_history_quail.pdf
(Quail's query and startup seconds at each merge, every change
labeled), bio4_history_versus_vllm.pdf (the same for the vLLM baselines
and Quail), and bio4_history_tokens.pdf (fresh input tokens, recomputed
KV tokens, and answer agreement). Percentages and ratios are derived
here from the saved summaries.
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
from plot_colors import BLUE, DARK, GRAY, GREEN, LIGHT_GRAY, ORANGE  # noqa: E402

# the vLLM baselines, then Quail's steps in merge order
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


# what each configuration adds over the one before, in plain words
CHANGE = {
    "vllm-defaults": "vLLM at default settings",
    "vllm-tuned": "vLLM with tuned batch and CUDA graph settings",
    "vllm-pipelined": "vLLM with each document's filters pipelined",
    "vllm-today": "vLLM tokenizes prompt text with Gigatoken",
    "quail-engine-vllm-kernels": "Quail's first engine, on vLLM's kernels",
    "quail-engine": "fused Triton kernels",
    "pinned_staging": "copy inputs to the GPU without blocking",
    "attention_paths": "separate attention code for filters and joins",
    "skip_arena_writes": "don't store filter KV that nothing reads",
    "join_search": "pick the join order and anchor by search",
    "compile_once": "compile kernels once, later boots load them",
    "filter_kv_reuse": "joins reuse the filters' KV; cheap filters first",
    "scan_ring": "cap kept KV so new work always fits",
    "boot_cache": "keep vLLM's cache on a volume; pin model version",
    "shared_retention": "keep KV for every join input, not just one",
    "join_continuous_batching": "join work admitted continuously",
    "projection_pushdown": "read only the columns the query uses",
    "plan_on_estimates": "plan while tokenization is still running",
    "filter_join_streaming": "filter results stream straight into the join",
    "gigatoken": "Gigatoken tokenizer",
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


def agreement(summary) -> float:
    """Share of predicate answers that match the reference labels."""
    return summary["metrics"]["accuracy"]["answer_accuracy"]["accuracy"]


def change(rows, i) -> str:
    """The change from the configuration before, when it is more than noise.

    Query changes under 3% are within the spread of two runs of the same
    code; startup changes under 15% are within the spread of its samples.
    """
    if not i:
        return ""
    (q, s), (pq, ps) = seconds(rows[i]), seconds(rows[i - 1])
    parts = []
    if abs(q / pq - 1) >= 0.03:
        parts.append(f"query {q / pq - 1:+.0%}")
    if abs(s / ps - 1) >= 0.15:
        parts.append(f"startup {s / ps - 1:+.0%}")
    return ", ".join(parts).replace("-", "\u2212")


def spread(targets, gap) -> list[float]:
    """Positions in the order of the sorted targets, at least gap apart.

    Each run of crowded labels is centered on the mean of its targets.
    """
    blocks = []
    for target in targets:
        blocks.append([target, [target]])
        while len(blocks) > 1:
            (start, members), (next_start, _) = blocks[-2], blocks[-1]
            if next_start >= start + gap * len(members):
                break
            members = members + blocks[-1][1]
            center = sum(members) / len(members)
            blocks[-2:] = [[center - gap * (len(members) - 1) / 2, members]]
    return [start + gap * k for start, members in blocks
            for k in range(len(members))]


def rail_labels(ax, points, texts, rail, weights, gap_in=0.26):
    """Labels on a slanted rail above the data, with leaders to points.

    Returns the label height in inches, so the caller can leave room.
    """
    x0, x1 = ax.get_xlim()
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    gap = gap_in * (x1 - x0) / width_in
    xs = spread([x for x, _ in points], gap)
    tallest = 0.0
    renderer = ax.figure.canvas.get_renderer()
    for (x, y), at, text, weight in zip(points, xs, texts, weights):
        ax.plot([x, at], [y, rail], color=GRAY, lw=0.6, zorder=1)
        label = ax.text(at, rail, text, rotation=45, rotation_mode="anchor",
                        ha="left", va="bottom", fontsize=8.5, color=DARK,
                        fontweight=weight, clip_on=False)
        box = label.get_window_extent(renderer)
        tallest = max(tallest, box.height / ax.figure.dpi)
    return tallest


def date_axis(ax, start, stop):
    """Weekly date ticks between two dates."""
    ticks = np.arange(np.ceil(start), stop, 7)
    ax.set_xticks(ticks, [f"{mdates.num2date(t):%b %-d}" for t in ticks])
    ax.set_xlim(start, stop)


def signed(change) -> str:
    return f"{change:+.0%}".replace("-", "\u2212")


def steps_over(values, threshold) -> dict:
    """Index -> change from the row before, for changes of threshold or more."""
    return {i: values[i] / values[i - 1] - 1 for i in range(1, len(values))
            if abs(values[i] / values[i - 1] - 1) >= threshold}


def quail_plot(summaries, names, out):
    """Quail's query seconds and startup seconds at each merge date.

    Query moves under 3% and startup moves under 15% are within the
    spread of repeated runs, so only larger moves are marked.
    """
    rows = [summaries[n] for n in names if summaries[n]["backend"] == "quail"]
    t = [mdates.date2num(when(r)) for r in rows]
    query = np.array([seconds(r)[0] for r in rows])
    startup = np.array([seconds(r)[1] for r in rows])
    query_steps = steps_over(query, 0.03)
    startup_steps = steps_over(startup, 0.15)
    fig = plt.figure(figsize=(12, 7.4))
    fig.set_layout_engine("none")
    top = fig.add_axes((0.06, 0.31, 0.82, 0.64))
    low = fig.add_axes((0.06, 0.06, 0.82, 0.16))
    stop = t[-1] + 2
    date_axis(low, t[0] - 1.6, stop)
    top.set_xlim(low.get_xlim())
    top.set_xticks([])
    # the first row is the first engine with vLLM's kernels, on the date
    # of the second
    step_line(top, t, query, stop, BLUE, first_apart=True)
    step_line(low, t, startup, stop, DARK, first_apart=True)

    # the names sit in the empty band above every row but the first
    bottom = 20 * np.floor(query.min() / 20 - 1)
    rail = query[1:].max() + 0.08 * (query[1:].max() - bottom)
    top.set_ylim(bottom, rail)
    texts, weights = [], []
    for i, row in enumerate(rows):
        pr = row.get("pull_request")
        texts.append((f"#{pr} " if pr else "") + CHANGE[row["name"]])
        marked = i in query_steps or i in startup_steps
        weights.append("bold" if marked else "normal")
    # the first engine shares the date of #3, so #3's label names both
    texts[1] += f" (before: vLLM's, {query[0]:,.0f} s)"
    tallest = rail_labels(top, list(zip(t[1:], query[1:])), texts[1:], rail,
                          weights[1:])
    top.annotate(f"{query[0]:,.0f} s", (t[0], query[0]), xytext=(8, 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=8.5, color=DARK)
    height_in = top.get_position().height * fig.get_figheight()
    span = (rail - bottom) * height_in / (height_in - tallest - 0.05)
    top.set_ylim(bottom, max(bottom + span, query.max() + 10))
    top.set_yticks(np.arange(100 * np.ceil(bottom / 100), query.max(), 100))
    low.set_ylim(0, startup.max() * 1.12)
    low.set_yticks(np.arange(0, startup.max(), 20))
    for ax in (top, low):
        ax.yaxis.grid(True, color=LIGHT_GRAY, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel("seconds")
    for ax, values, steps in ((top, query, query_steps),
                              (low, startup, startup_steps)):
        for i, step in steps.items():
            middle = (values[i] + values[i - 1]) / 2
            # same date: the label goes left of the drop
            left = t[i] == t[i - 1]
            ax.annotate(signed(step), (t[i], middle),
                        xytext=(-6 if left else 8, 0),
                        textcoords="offset points",
                        ha="right" if left else "left", va="center",
                        fontsize=10, fontweight="bold",
                        color=GREEN if step < 0 else DARK)
    top.text(stop + 0.3, query[-1], f"query today:\n{query[-1]:,.0f} s",
             ha="left", va="center", fontsize=10, fontweight="bold",
             color=BLUE, clip_on=False)
    low.text(stop + 0.3, startup[-1], f"startup today:\n{startup[-1]:,.0f} s",
             ha="left", va="center", fontsize=10, fontweight="bold",
             color=DARK, clip_on=False)
    total_first, total_last = query[0] + startup[0], query[-1] + startup[-1]
    top.set_title(f"BIO-4 query time on Quail's code at each merge date "
                  f"(startup + query: {total_first:,.0f} s to "
                  f"{total_last:,.0f} s)")
    low.set_title("BIO-4 startup time (process start to engine ready)")
    fig.savefig(out)
    plt.close(fig)


def history(summaries, names, quail):
    """Rows, dates, query seconds, and total seconds of Quail or vLLM."""
    rows = [summaries[n] for n in names
            if (summaries[n]["backend"] == "quail") == quail]
    t = [mdates.date2num(when(r)) for r in rows]
    query = np.array([seconds(r)[0] for r in rows])
    total = query + np.array([seconds(r)[1] for r in rows])
    return rows, t, query, total


def step_line(ax, t, values, stop, color, first_apart, lw=2):
    """A step line with a point at each date.

    With first_apart, the first row shares the second row's date and is
    drawn as a hollow point joined to it by a dotted line.
    """
    skip = 1 if first_apart else 0
    ax.step(t[skip:] + [stop], list(values[skip:]) + [values[-1]],
            where="post", color=color, lw=lw)
    ax.scatter(t[skip:], values[skip:], s=24, color=color, edgecolor="white",
               lw=1, zorder=3)
    if first_apart:
        ax.plot([t[0], t[0]], [values[1], values[0]], color=color, lw=0.8,
                ls=":")
        ax.scatter(t[:1], values[:1], s=24, color="white", edgecolor=color,
                   lw=1.4, zorder=3)


def seconds_lines(ax, t, query, total, stop, color, first_apart):
    """Query line, startup band, and startup + query line as steps."""
    skip = 1 if first_apart else 0
    ax.fill_between(t[skip:] + [stop], list(query[skip:]) + [query[-1]],
                    list(total[skip:]) + [total[-1]], step="post",
                    color=LIGHT_GRAY, lw=0)
    step_line(ax, t, total, stop, DARK, first_apart, lw=1.2)
    step_line(ax, t, query, stop, color, first_apart)


def versus_plot(summaries, names, out):
    """Startup and query seconds of vLLM and Quail by date."""
    vllm, vt, vq, vtotal = history(summaries, names, quail=False)
    quail, qt, qq, qtotal = history(summaries, names, quail=True)
    fig = plt.figure(figsize=(12, 4.6))
    fig.set_layout_engine("none")
    ax = fig.add_axes((0.07, 0.09, 0.78, 0.8))
    stop = max(vt + qt) + 2.5
    date_axis(ax, vt[0] - 2, stop)
    seconds_lines(ax, vt, vq, vtotal, stop, ORANGE, first_apart=False)
    seconds_lines(ax, qt, qq, qtotal, stop, BLUE, first_apart=True)
    for row, x, y in zip(vllm, vt, vtotal):
        pr = row.get("pull_request")
        right = x == vt[-1]
        name = SHORT[row["name"]] + (f" #{pr}" if pr else "")
        if right:
            name = f"vLLM today ({name.removeprefix('vLLM ')})"
        text = f"{name}\n{y:,.0f} s ({when(row):%b %-d})"
        ax.annotate(text, (x, y), xytext=(-6 if right else 6, 8),
                    textcoords="offset points", ha="right" if right else "left",
                    va="bottom", fontsize=9.5, color=DARK)
    ax.annotate(f"Quail, first engine\n{qtotal[0]:,.0f} s (Aug 18)",
                (qt[0], qtotal[0]), xytext=(6, 8), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.5, color=DARK)
    ax.text(stop + 0.4, qtotal[-1],
            f"Quail today: {qtotal[-1]:,.0f} s\n"
            f"(code of {when(quail[-1]):%b %-d})",
            ha="left", va="center", fontsize=10, color=DARK, clip_on=False)
    arrow_x = stop - 1.2
    ax.annotate("", (arrow_x, qtotal[-1] + 60), (arrow_x, vtotal[-1] - 60),
                arrowprops={"arrowstyle": "<->", "color": DARK, "lw": 1})
    ax.text(arrow_x - 0.5, (qtotal[-1] + vtotal[-1]) / 2,
            f"{vtotal[-1] / qtotal[-1]:.1f}\u00d7 less time,\nstartup + query",
            ha="right", va="center", fontsize=10, fontweight="bold",
            color=GREEN)
    handles = [
        plt.Line2D([], [], color=ORANGE, lw=2, label="vLLM, query"),
        plt.Line2D([], [], color=BLUE, lw=2, label="Quail, query"),
        plt.Rectangle((0, 0), 1, 1, color=LIGHT_GRAY, label="startup"),
        plt.Line2D([], [], color=DARK, lw=1.2, label="startup + query"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.01, 0.42))
    ax.set_ylim(0, vtotal.max() * 1.12)
    ax.set_yticks(np.arange(0, vtotal.max(), 1000))
    ax.yaxis.grid(True, color=LIGHT_GRAY, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("seconds")
    ax.set_title("BIO-4 startup + query time: vLLM baselines and Quail by date")
    fig.savefig(out)
    plt.close(fig)


def token_plot(summaries, names, out):
    """Fresh tokens, recomputed KV tokens, and agreement by date."""
    vllm, vt, _, _ = history(summaries, names, quail=False)
    quail, qt, _, _ = history(summaries, names, quail=True)
    stop = max(vt + qt) + 2.5
    panels = (
        ("Fresh input tokens by date", "millions of tokens",
         lambda r: r["metrics"]["fresh_tokens"] / 1e6, "{:.1f} M"),
        ("Recomputed KV tokens by date", "millions of tokens",
         lambda r: r["metrics"]["regret_tokens"] / 1e6, "{:.1f} M"),
        ("Answer agreement with the Qwen3 32B reference labels by date",
         "percent",
         lambda r: 100 * agreement(r), "{:.1f}%"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True)
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.07, right=0.84, top=0.95, bottom=0.05,
                        hspace=0.32)
    streaming = next(k for k, r in enumerate(quail)
                     if r["name"] == "filter_join_streaming")
    for ax, (title, unit, value, fmt) in zip(axes, panels):
        date_axis(ax, vt[0] - 2, stop)
        v = np.array([value(r) for r in vllm])
        q = np.array([value(r) for r in quail])
        step_line(ax, vt, v, stop, ORANGE, first_apart=False)
        step_line(ax, qt, q, stop, BLUE, first_apart=True)
        low, high = min(v.min(), q.min()), max(v.max(), q.max())
        pad = 0.15 * (high - low)
        ax.set_ylim(max(0, low - pad), high + pad)
        for name, values, color in (("vLLM today", v, ORANGE),
                                    ("Quail today", q, BLUE)):
            ax.text(stop + 0.4, values[-1],
                    f"{name}: {fmt.format(values[-1])}", ha="left",
                    va="center", fontsize=10, fontweight="bold", color=color,
                    clip_on=False)
        i = streaming
        if abs(q[i] / q[i - 1] - 1) >= 0.03:
            step = f"{q[i] / q[i - 1] - 1:+.0%}".replace("-", "\u2212")
            ax.annotate(f"filter-join streaming #92: {step}", (qt[i], q[i]),
                        xytext=(-8, -4), textcoords="offset points",
                        ha="right", va="top", fontsize=9.5, color=GREEN,
                        fontweight="bold")
        if abs(v[-1] / v[-2] - 1) >= 0.03:
            ax.annotate("vLLM tokenizes whole prompt text (#167)",
                        (vt[-1], v[-1]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9.5, color=DARK)
        ax.annotate("Quail, first engine", (qt[0], q[0]), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9.5, color=DARK)
        ax.yaxis.grid(True, color=LIGHT_GRAY, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel(unit)
        ax.set_title(f"BIO-4 {title[0].lower()}{title[1:]}", fontsize=11)
    fig.savefig(out)
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
    versus_plot(summaries, names, OUT / "bio4_history_versus_vllm.pdf")
    quail_plot(summaries, names, OUT / "bio4_history_quail.pdf")
    token_plot(summaries, names, OUT / "bio4_history_tokens.pdf")
    print(f"wrote three figures to {OUT}")


if __name__ == "__main__":
    main()
