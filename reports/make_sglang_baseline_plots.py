"""Plot the SGLang baseline against SoL, Quail, and the vLLM baselines.

Pull the five inputs from the quail-results volume, then pass the
work directory to this script:

    W=<workdir>
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
    modal volume get quail-results ablations/ringfix_bio2.json $W/quail_ringfix_bio2.json
    modal volume get quail-results ablations/ringfix_tokens_head_imdb3.json $W/quail_ringfix_imdb3.json
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_vllm.json
    modal volume get quail-results pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined_vllm.json
    modal volume get quail-results pipelined_sglang/2026-08-31_034457_9032b486/summary.json $W/pipelined_sglang.json
    uv run --with matplotlib python reports/make_sglang_baseline_plots.py $W
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
from plot_colors import BLUE, GRAY, ORANGE, RED, TEAL  # noqa: E402

QUERY_IDS = ("BIO-2", "IMDB-3")
MODEL = "qwen3-4b-fp8"


def load(path):
    with path.open() as source:
        return json.load(source)


def check(summary, baseline, memory_key, memory_fraction,
          filter_submission, join_submission=None,
          batched_tokens=25_305):
    if (summary["baseline"] != baseline
            or summary["filter_submission"] != filter_submission
            or summary["hf_name"] != "Qwen/Qwen3-4B-FP8"
            or summary["sf"] != 0.1
            or summary["checkpoint"] != "pre-quantized FP8"
            or summary[memory_key] != memory_fraction
            or summary["max_num_seqs"] != 4096
            or summary["max_num_batched_tokens"] != batched_tokens
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


def usd(wall_s):
    return wall_s / 3600.0 * H100_USD_PER_HOUR


def annotate(ax, bar, wall_s):
    ax.annotate(
        f"{wall_s:,.1f} s\n${usd(wall_s):,.4f}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center", va="bottom", fontsize=9)


def five_way_figure(sol, quail_walls, entries):
    systems = (
        ("SoL\nestimate", GRAY),
        ("Quail", BLUE),
        ("stock\nvLLM", ORANGE),
        ("pipelined\nvLLM", TEAL),
        ("pipelined\nSGLang", RED),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(wspace=0.3)
    for ax, qid in zip(axes, QUERY_IDS):
        walls = [
            sol["queries"][qid]["models"][MODEL]["sol_s"],
            quail_walls[qid],
            entries["stock_vllm"][qid]["total_wall_s"],
            entries["pipelined_vllm"][qid]["total_wall_s"],
            entries["pipelined_sglang"][qid]["total_wall_s"],
        ]
        names = [name for name, _color in systems]
        colors = [color for _name, color in systems]
        log = max(walls) / min(walls) > 10
        bars = ax.bar(names, walls, color=colors, width=0.62)
        for bar, wall in zip(bars, walls):
            annotate(ax, bar, wall)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("seconds (log scale)")
            ax.set_ylim(min(walls) * 0.5, max(walls) * 3.2)
        else:
            ax.set_ylabel("seconds")
            ax.set_ylim(0, max(walls) * 1.3)
        ratio = (entries["pipelined_vllm"][qid]["total_wall_s"]
                 / entries["pipelined_sglang"][qid]["total_wall_s"])
        direction = "faster" if ratio >= 1 else "slower"
        factor = ratio if ratio >= 1 else 1 / ratio
        pairs = join_pairs(entries["pipelined_sglang"][qid])
        ax.set_title(
            f"{qid} — {pairs:,} join pairs\n"
            f"pipelined SGLang {factor:.2f}x {direction} than "
            f"pipelined vLLM")
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    out_path = OUT / "sglang_baseline_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    workdir = Path(sys.argv[1])
    sol = load(workdir / "sol.json")
    quail_runs = [load(workdir / "quail_ringfix_bio2.json"),
                  load(workdir / "quail_ringfix_imdb3.json")]
    stock_vllm = load(workdir / "stock_vllm.json")
    pipelined_vllm = load(workdir / "pipelined_vllm.json")
    pipelined_sglang = load(workdir / "pipelined_sglang.json")

    if sol["scale_factor"] != 0.1:
        raise ValueError("unexpected SoL configuration")
    for run in quail_runs:
        if run["model"] != MODEL or run["sf"] != 0.1:
            raise ValueError("unexpected Quail configuration")
    quail_walls = {run["query"]: run["unprofiled"]["engine_wall_s"]
                   for run in quail_runs}
    if sorted(quail_walls) != sorted(QUERY_IDS):
        raise ValueError("Quail runs do not cover the two queries")
    check(stock_vllm, "stock_vllm", "gpu_memory_utilization", 0.91,
          "stage-major")
    check(pipelined_vllm, "pipelined_vllm", "gpu_memory_utilization",
          0.91, "pipelined")
    check(pipelined_sglang, "pipelined_sglang", "mem_fraction_static",
          0.76, "pipelined", join_submission="suffix-major-tiled",
          batched_tokens=25_296)

    entries = {
        "stock_vllm": query_entries(stock_vllm),
        "pipelined_vllm": query_entries(pipelined_vllm),
        "pipelined_sglang": query_entries(pipelined_sglang),
    }

    OUT.mkdir(exist_ok=True)
    five_way_figure(sol, quail_walls, entries)

    for qid in QUERY_IDS:
        print(f"{qid} SoL estimate: "
              f"{sol['queries'][qid]['models'][MODEL]['sol_s']:.2f} s")
        print(f"{qid} Quail: {quail_walls[qid]:.2f} s")
        for key in ("stock_vllm", "pipelined_vllm", "pipelined_sglang"):
            entry = entries[key][qid]
            wall = entry["total_wall_s"]
            pairs = join_pairs(entry)
            accuracy = entry["accuracy"]["answer_accuracy"]["accuracy"]
            print(f"{qid} {key}: {wall:.2f} s, {pairs:,} pairs, "
                  f"{pairs / wall:.1f} pairs/s, ${usd(wall):.4f}/query, "
                  f"answer accuracy {accuracy:.2%}")


if __name__ == "__main__":
    main()
