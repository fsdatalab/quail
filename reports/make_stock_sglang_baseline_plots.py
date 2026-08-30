"""Plot stock SGLang against stock vLLM on BIO-2 and IMDB-3 at sf=0.1.

Pull the three summaries from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_vllm.json
    modal volume get quail-results stock_sglang/2026-08-30_033502_05712d88/summary.json $W/stock_sglang_anchor_major.json
    modal volume get quail-results stock_sglang/2026-08-30_042550_e5f6d2e8/summary.json $W/stock_sglang_tiled.json
    uv run --with matplotlib python reports/make_stock_sglang_baseline_plots.py $W
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from quail.bench.evaluate import H100_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, GRAY, ORANGE  # noqa: E402

QUERY_IDS = ("BIO-2", "IMDB-3")
SYSTEMS = (
    ("stock vLLM\n(anchor-major)", GRAY),
    ("stock SGLang\n(anchor-major)", ORANGE),
    ("stock SGLang\n(tiled)", BLUE),
)


def load(path):
    with path.open() as source:
        return json.load(source)


def check(summary, baseline, memory_key, memory_fraction,
          join_submission=None):
    if (summary["baseline"] != baseline
            or summary["filter_submission"] != "stage-major"
            or summary["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or summary["sf"] != 0.1
            or summary["checkpoint"] != "pre-quantized FP8"
            or summary[memory_key] != memory_fraction
            or summary["max_num_seqs"] != 4096
            or summary["max_num_batched_tokens"] != 25_305
            or summary.get("join_submission") != join_submission):
        raise ValueError(f"unexpected {baseline} configuration")


def query_entries(summary):
    entries = {row["query"]: row for row in summary["results"][0]}
    missing = [qid for qid in QUERY_IDS if qid not in entries]
    if missing:
        raise ValueError(f"missing queries {missing}")
    return entries


def join_pairs(entry):
    return sum(step["n_pairs"] for step in entry["steps"]
               if step["kind"] == "join")


def metrics(entry):
    wall = entry["total_wall_s"]
    pairs = join_pairs(entry)
    return dict(
        wall_s=wall,
        pairs=pairs,
        pairs_per_s=pairs / wall,
        usd=wall / 3600.0 * H100_USD_PER_HOUR,
        answer_accuracy=entry["accuracy"]["answer_accuracy"]["accuracy"],
    )


def main():
    workdir = Path(sys.argv[1])
    vllm = load(workdir / "stock_vllm.json")
    anchor_major = load(workdir / "stock_sglang_anchor_major.json")
    tiled = load(workdir / "stock_sglang_tiled.json")
    check(vllm, "stock_vllm", "gpu_memory_utilization", 0.91)
    check(anchor_major, "stock_sglang", "mem_fraction_static", 0.78)
    check(tiled, "stock_sglang", "mem_fraction_static", 0.78,
          join_submission="suffix-major-tiled")

    entries = [query_entries(summary)
               for summary in (vllm, anchor_major, tiled)]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(wspace=0.3)
    for ax, qid in zip(axes, QUERY_IDS):
        rows = [metrics(system_entries[qid])
                for system_entries in entries]
        walls = [row["wall_s"] for row in rows]
        names = [name for name, _color in SYSTEMS]
        colors = [color for _name, color in SYSTEMS]
        bars = ax.bar(names, walls, color=colors, width=0.6)
        for bar, row in zip(bars, rows):
            ax.annotate(
                f"{row['wall_s']:,.1f} s\n"
                f"{row['pairs_per_s']:,.0f} pairs/s\n"
                f"${row['usd']:,.4f}/query",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9)
        ratio = walls[0] / walls[2]
        direction = "faster" if ratio >= 1 else "slower"
        factor = ratio if ratio >= 1 else 1 / ratio
        pair_counts = sorted({row["pairs"] for row in rows})
        if len(pair_counts) == 1:
            pairs_text = f"{pair_counts[0]:,} join pairs"
        else:
            pairs_text = (
                f"{min(pair_counts):,}-{max(pair_counts):,} join pairs")
        ax.set_title(
            f"{qid} — {pairs_text}\n"
            f"tiled stock SGLang {factor:.2f}x {direction} than "
            f"stock vLLM")
        ax.set_ylabel("seconds")
        ax.set_ylim(0, max(walls) * 1.4)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "stock_sglang_vs_stock_vllm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")

    for qid in QUERY_IDS:
        for (name, _color), system_entries in zip(SYSTEMS, entries):
            row = metrics(system_entries[qid])
            label = name.replace("\n", " ")
            print(f"{qid} {label}: {row['wall_s']:.2f} s, "
                  f"{row['pairs']:,} pairs, "
                  f"{row['pairs_per_s']:.1f} pairs/s, "
                  f"${row['usd']:.4f}/query, "
                  f"answer accuracy {row['answer_accuracy']:.2%}")


if __name__ == "__main__":
    main()
