"""Plot stock SGLang against stock vLLM on BIO-2 and IMDB-3 at sf=0.1.

Pull the two summaries from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_vllm.json
    modal volume get quail-results stock_sglang/2026-08-30_015521_465020a3/summary.json $W/stock_sglang.json
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
from plot_colors import BLUE, GRAY  # noqa: E402

QUERY_IDS = ("BIO-2", "IMDB-3")
SYSTEMS = (("stock vLLM", GRAY), ("stock SGLang", BLUE))


def load(path):
    with path.open() as source:
        return json.load(source)


def check(summary, baseline, memory_key, memory_fraction):
    if (summary["baseline"] != baseline
            or summary["filter_submission"] != "stage-major"
            or summary["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or summary["sf"] != 0.1
            or summary["checkpoint"] != "pre-quantized FP8"
            or summary[memory_key] != memory_fraction
            or summary["max_num_seqs"] != 4096
            or summary["max_num_batched_tokens"] != 25_305):
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
    sglang = load(workdir / "stock_sglang.json")
    check(vllm, "stock_vllm", "gpu_memory_utilization", 0.91)
    check(sglang, "stock_sglang", "mem_fraction_static", 0.78)

    vllm_entries = query_entries(vllm)
    sglang_entries = query_entries(sglang)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(wspace=0.35)
    for ax, qid in zip(axes, QUERY_IDS):
        rows = [metrics(vllm_entries[qid]), metrics(sglang_entries[qid])]
        walls = [row["wall_s"] for row in rows]
        colors = [color for _name, color in SYSTEMS]
        names = [name for name, _color in SYSTEMS]
        bars = ax.bar(names, walls, color=colors, width=0.5)
        for bar, row in zip(bars, rows):
            ax.annotate(
                f"{row['wall_s']:,.1f} s\n"
                f"{row['pairs_per_s']:,.0f} pairs/s\n"
                f"${row['usd']:,.4f}/query",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9)
        ratio = walls[0] / walls[1]
        direction = "faster" if ratio >= 1 else "slower"
        factor = ratio if ratio >= 1 else 1 / ratio
        ax.set_title(
            f"{qid} — {metrics(sglang_entries[qid])['pairs']:,} join "
            f"pairs\nstock SGLang {factor:.2f}x {direction}")
        ax.set_ylabel("seconds")
        ax.set_ylim(0, max(walls) * 1.35)
        ax.spines[["top", "right"]].set_visible(False)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "stock_sglang_vs_stock_vllm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")

    for qid in QUERY_IDS:
        for name, entries in (("stock vLLM", vllm_entries),
                              ("stock SGLang", sglang_entries)):
            row = metrics(entries[qid])
            print(f"{qid} {name}: {row['wall_s']:.2f} s, "
                  f"{row['pairs']:,} pairs, "
                  f"{row['pairs_per_s']:.1f} pairs/s, "
                  f"${row['usd']:.4f}/query, "
                  f"answer accuracy {row['answer_accuracy']:.2%}")


if __name__ == "__main__":
    main()
