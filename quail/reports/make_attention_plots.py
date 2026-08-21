"""Attention-path figures (issue #24): path performance on filters
and joins, accuracy against stock vLLM, and the FlashInfer
comparison.

    uv run --with matplotlib python reports/make_attention_plots.py

Reads results/attention_paths.json,
results/join_attention_paths_*.json, results/accuracy_vs_stock.json,
and results/flashinfer_bench.json; writes three PNGs into
reports/plots/.
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
from plot_colors import BLUE, GRAY, GREEN, RED, DARK, ORANGE, TEAL

PATH_COLOR = {"split": BLUE, "merge_quant": ORANGE,
              "unified": GREEN, "unified_waves": GREEN}
PATH_LABEL = {"split": "split", "merge_quant": "merge_quant",
              "unified": "unified", "unified_waves": "unified (waves)"}


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


# ---------------------------------------------- figure 1: path speed

def fig_paths(tag="", model_label="Qwen3 4B fp8"):
    ap = load(f"attention_paths{tag}.json")
    joins = {label: load(f"join_attention_paths{tag}_{label}.json")
             for label in ("10x256", "1x2560", "100x256")}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 3.8),
        gridspec_kw={"width_ratios": [1, 1.35]})

    # left: the filter workload
    modes = ["split", "merge_quant", "unified"]
    y = range(len(modes))
    us = [mean(ap["runs"][m], "us_per_token") for m in modes]
    walls = [mean(ap["runs"][m], "wall") for m in modes]
    ax1.barh(list(y), us, height=0.55,
             color=[PATH_COLOR[m] for m in modes], edgecolor="white")
    for i, (u, w) in enumerate(zip(us, walls)):
        ax1.text(u + 0.008 * max(us), i,
                 f"{u:.2f} us/token   {w:.1f} s wall",
                 va="center", fontsize=9, color=DARK)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([PATH_LABEL[m] for m in modes])
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(us) * 1.6)
    ax1.set_xlabel("us per fresh token (lower is better)")
    ax1.set_title(
        f"Filters: {ap['n_docs']:,} documents, 5 stages\n"
        f"assignment: unified", loc="left")

    # right: the join workload across fan-out shapes
    shapes = ["10x256", "1x2560", "100x256"]
    shape_note = {"10x256": "10 anchors x 256 partners",
                  "1x2560": "1 anchor x 2,560 (high fan-out)",
                  "100x256": "100 anchors x 256 (multi-chunk)"}
    jmodes = ["split", "merge_quant", "unified_waves"]
    clip = 3.0 * max(mean(joins[s]["runs"]["split"], "us_per_token")
                     for s in shapes)
    bar_h = 0.24
    yticks, ylabels = [], []
    for si, shape in enumerate(shapes):
        base = si * (len(jmodes) + 1) * bar_h
        mq = mean(joins[shape]["runs"]["merge_quant"],
                  "us_per_token")
        for mi, m in enumerate(jmodes):
            u = mean(joins[shape]["runs"][m], "us_per_token")
            ypos = base + mi * bar_h
            shown = min(u, clip)
            ax2.barh(ypos, shown, height=bar_h * 0.82,
                     color=PATH_COLOR[m], edgecolor="white")
            if u > clip:
                label = (f"{u:.0f} us/token "
                         f"({u / mq:.0f}x merge_quant)")
                ax2.text(clip * 0.99, ypos, label, va="center",
                         ha="right", fontsize=8.5, color="white",
                         fontweight="bold")
            else:
                ax2.text(shown + 0.01 * clip, ypos, f"{u:.1f}",
                         va="center", fontsize=8.5, color=DARK)
        yticks.append(base + bar_h)
        ylabels.append(shape_note[shape])
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, clip)
    ax2.set_xlabel("us per fresh token (lower is better)")
    ax2.set_title("Joins: BioDEX reports x terms\n"
                  "assignment: merge_quant", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PATH_COLOR[m])
               for m in jmodes]
    ax2.legend(handles, [PATH_LABEL[m] for m in jmodes],
               loc="lower right")

    fig.suptitle(
        f"Attention paths by workload ({model_label}, one H100)",
        x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"attention_paths{tag}.png")
    print("wrote", OUT / f"attention_paths{tag}.png")


# ------------------------------------- figure 2: accuracy vs stock

def fig_accuracy(tag="", model_label="Qwen3 4B fp8"):
    acc = load(f"accuracy_vs_stock{tag}.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9))

    # left: planted-truth accuracy, filters and the join task
    entities = [("stock vLLM", GRAY), ("split", BLUE),
                ("merge_quant", ORANGE), ("unified", GREEN)]
    flag_acc = {
        "stock vLLM": (sum(acc["stock_flag_accuracy"].values())
                       / len(acc["stock_flag_accuracy"])),
        **{m: acc["filters"][m]["flag_accuracy"]
           for m in ("split", "merge_quant", "unified")}}
    key_acc = {
        "stock vLLM": acc["join"]["split"]["key_accuracy_stock"],
        "split": acc["join"]["split"]["key_accuracy_quail"],
        "merge_quant": acc["join"]["merge_quant"]["key_accuracy_quail"],
        "unified": None}

    bar_h = 0.19
    yticks, ylabels = [], []
    for gi, (gname, series) in enumerate(
            (("filters: planted flags", flag_acc),
             ("join: planted keys", key_acc))):
        base = gi * (len(entities) + 1.2) * bar_h
        for ei, (ent, color) in enumerate(entities):
            v = series[ent]
            if v is None:
                continue
            ypos = base + ei * bar_h
            ax1.barh(ypos, v * 100, height=bar_h * 0.82, color=color,
                     edgecolor="white")
            ax1.text(v * 100 + 1.2, ypos, f"{v * 100:.2f}%",
                     va="center", fontsize=8.5, color=DARK)
        yticks.append(base + 1.5 * bar_h)
        ylabels.append(gname)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 112)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("accuracy against planted ground truth (%)")
    ax1.set_title("Answer accuracy, both systems", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for _, c in entities]
    ax1.legend(handles, [e for e, _ in entities], ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, -0.22))

    # right: disagreement with stock, with the margin context
    rows = [
        ("filters\nsplit", acc["filters"]["split"], BLUE),
        ("filters\nmerge_quant", acc["filters"]["merge_quant"],
         ORANGE),
        ("filters\nunified", acc["filters"]["unified"], GREEN),
        ("join\nsplit", acc["join"]["split"], BLUE),
        ("join\nmerge_quant", acc["join"]["merge_quant"],
         ORANGE),
    ]
    y = range(len(rows))
    for i, (label, d, color) in enumerate(rows):
        rate = 100.0 * d["disagreements"] / d["compared"]
        dec = 100.0 * d["decisive_disagreements"] / d["compared"]
        ax2.barh(i, rate, height=0.55, color=color, edgecolor="white",
                 alpha=0.45)
        ax2.barh(i, dec, height=0.55, color=color, edgecolor="white")
        ax2.text(rate + 0.1, i,
                 f"{d['disagreements']}/{d['compared']}"
                 f"  ({rate:.2f}%),  {d['decisive_disagreements']} at "
                 f"decisive margin",
                 va="center", fontsize=8.5, color=DARK)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.invert_yaxis()
    max_rate = max(100.0 * d["disagreements"] / d["compared"]
                   for _, d, _ in rows)
    ax2.set_xlim(0, max(max_rate * 2.6, 3.0))
    ax2.set_xlabel("answers that differ from stock vLLM (%)")
    ax2.set_title(
        "Disagreement with stock (solid = at |margin| > 1)",
        loc="left")
    ctrl = acc["stock_self_control"]
    fm = acc["stock_filter_margins"]
    jm = acc["stock_join_margins"]
    if fm["p50"] == 0.0:
        margin_note = (
            "stock margins saturate at this model's confidence (the "
            "losing token drops out of vLLM's returned top-k "
            "logprobs), so disagreements are graded against planted "
            "truth instead - see the report.")
    else:
        margin_note = (
            f"stock margin medians: filters {fm['p50']:.2f} "
            f"({fm['under_1'] * 100:.0f}% under 1), join "
            f"{jm['p50']:.2f} ({jm['under_1'] * 100:.0f}% under 1): "
            f"join answers are barely decided, so numeric noise "
            f"flips more of them.")
    ax2.text(0.995, -0.32,
             f"stock's own order-shuffle control: {ctrl['flips']} flips "
             f"in {ctrl['of']} (filters) and "
             f"{acc['stock_join_control']['flips']} in "
             f"{acc['stock_join_control']['of']} (join).\n"
             + margin_note,
             transform=ax2.transAxes, fontsize=7.8, color="#6b6b66",
             ha="right", va="top")

    fig.suptitle(
        f"Accuracy against stock vLLM ({model_label}): identical "
        f"token streams, TRUE/FALSE constrained, temperature 0",
        x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(OUT / f"accuracy_vs_stock{tag}.png")
    print("wrote", OUT / f"accuracy_vs_stock{tag}.png")


# ------------------------------------------ figure 3: FlashInfer

def fig_flashinfer(tag="", geom_label="4B geometry: 32 query heads"):
    fi = load(f"flashinfer_bench{tag}.json")
    try:
        tuned = load(f"flashinfer_tuned{tag}.json")["shapes"]
    except FileNotFoundError:
        tuned = {}
    shapes = [
        ("filter_fresh", "filter, fresh chunk\n(~110k tokens)"),
        ("filter_cached", "filter, rewind chunk\n(~11k tokens)"),
        ("join", "join, 10 anchors\nx 26 suffixes"),
        ("join_fanout", "join, 1 anchor\nx 256 suffixes"),
    ]
    variants = [
        ("fa3_split_plus_quant", "FA3 split + quant", BLUE),
        ("fa3_merge_quant", "FA3 merge_quant", ORANGE),
        ("fa3_unified_plus_quant", "FA3 unified + quant", GREEN),
        ("fi_paged_causal_plus_quant", "FlashInfer paged causal",
         RED),
        ("fi_two_call_merge_plus_quant",
         "FlashInfer two-call + merge_state", TEAL),
        ("fi_cascade_plus_quant", "FlashInfer cascade", GRAY),
        ("fi_best_tuned", "FlashInfer best forced backend", DARK),
    ]

    def best_tuned(skey):
        # the fastest pure-FlashInfer variant across forced backends
        row = tuned.get(skey)
        if not row:
            return None
        pool = [(v, f"{grp} {be}")
                for grp in ("paged_causal", "two_call_merge_state")
                for be, v in row[grp].items()]
        return min(pool) if pool else None

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bar_h = 0.15
    yticks, ylabels = [], []
    for si, (skey, slabel) in enumerate(shapes):
        row = fi["shapes"][skey]
        vals = {**row["fa3"], **row["flashinfer"]}
        bt = best_tuned(skey)
        if bt is not None:
            vals["fi_best_tuned"] = bt[0]
        present = [(k, lab, c) for k, lab, c in variants if k in vals]
        base = si * (len(variants) + 1.4) * bar_h
        for vi, (k, lab, c) in enumerate(present):
            v = vals[k]
            ypos = base + vi * bar_h
            ax.barh(ypos, v, height=bar_h * 0.82, color=c,
                    edgecolor="white")
            note = f" ({bt[1]})" if k == "fi_best_tuned" else ""
            ax.text(v + 0.05, ypos, f"{v:.2f}{note}", va="center",
                    fontsize=8, color=DARK)
        yticks.append(base + (len(present) - 1) * bar_h / 2)
        ylabels.append(slabel)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("per-layer attention ms, including KV "
                  "scatter, merge, and FP8 quantization "
                  "(lower is better)")
    ax.set_title(f"FlashInfer 0.6.14 against FlashAttention-3 on "
                 f"the real chunk shapes ({geom_label})",
                 loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for _, _, c in variants]
    ax.legend(handles, [lab for _, lab, _ in variants],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"flashinfer_compare{tag}.png")
    print("wrote", OUT / f"flashinfer_compare{tag}.png")


if __name__ == "__main__":
    fig_paths()
    fig_accuracy()
    fig_flashinfer()
    # the 32B variants render once their result files exist
    for fn, args in (
            (fig_paths, ("_32b", "Qwen3 32B fp8")),
            (fig_accuracy, ("_32b", "Qwen3 32B fp8")),
            (fig_flashinfer,
             ("_64h", "32B geometry: 64 query heads"))):
        try:
            fn(*args)
        except FileNotFoundError as exc:
            print(f"skipped {args[0]}: {exc}")
