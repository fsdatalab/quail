"""Plot the SGLang baseline against SoL, Quail, and the vLLM baselines.

Pull the inputs from the quail-results volume, then pass the work
directory to this script:

    W=<workdir>
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
    modal volume get quail-results ablations/ringfix_bio2.json $W/quail_ringfix_bio2.json
    modal volume get quail-results benchmarks/quailb/runs/qb_20260831T062218Z_1192cd76/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/quail_agent.json
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_vllm.json
    modal volume get quail-results pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined_vllm.json
    modal volume get quail-results stock_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock_vllm_agent.json
    modal volume get quail-results pipelined_vllm/20260831T062218Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined_vllm_agent.json
    modal volume get quail-results pipelined_sglang/2026-08-31_034457_9032b486/summary.json $W/pipelined_sglang_clean.json
    modal volume get quail-results pipelined_sglang/2026-08-31_213347_9f85216d/summary.json $W/pipelined_sglang_agent.json
    uv run --with matplotlib python reports/make_sglang_baseline_plots.py $W

The SoL file does not cover the agent queries, so the AGENT-1 panel
has no SoL bar.
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

QUERY_IDS = ("BIO-2", "AGENT-1")
MODEL = "qwen3-4b-fp8"
SYSTEM_COLORS = {
    "SoL\nestimate": GRAY,
    "Quail": BLUE,
    "stock\nvLLM": ORANGE,
    "pipelined\nvLLM": TEAL,
    "pipelined\nSGLang": RED,
}


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


def entry_for(summary, qid):
    entries = {row["query"]: row for row in summary["results"][0]}
    if qid not in entries:
        raise ValueError(f"summary is missing {qid}")
    return entries[qid]


def quail_family_wall(family_run, qid):
    if family_run["model"] != MODEL or family_run["sf"] != 0.1:
        raise ValueError("unexpected Quail family run configuration")
    for row in family_run["passes"]["single"]["queries"]:
        if row["query"] == qid:
            return row["wall_s"]
    raise ValueError(f"Quail family run is missing {qid}")


def join_pairs(entry):
    return sum(step["n_pairs"] for step in entry["steps"]
               if step["kind"] == "join")


def filter_documents(entry):
    return entry["steps"][0]["n_in"]


def usd(wall_s):
    return wall_s / 3600.0 * H100_USD_PER_HOUR


def annotate(ax, bar, wall_s):
    ax.annotate(
        f"{wall_s:,.1f} s\n${usd(wall_s):,.4f}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center", va="bottom", fontsize=9)


def figure(panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(11.5, 5.0))
    if len(panels) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.3)
    for ax, (qid, subtitle, walls) in zip(axes, panels):
        names = list(walls)
        values = [walls[name] for name in names]
        colors = [SYSTEM_COLORS[name] for name in names]
        log = max(values) / min(values) > 10
        bars = ax.bar(names, values, color=colors, width=0.62)
        for bar, wall in zip(bars, values):
            annotate(ax, bar, wall)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("seconds (log scale)")
            ax.set_ylim(min(values) * 0.5, max(values) * 3.2)
        else:
            ax.set_ylabel("seconds")
            ax.set_ylim(0, max(values) * 1.3)
        ratio = walls["pipelined\nvLLM"] / walls["pipelined\nSGLang"]
        direction = "faster" if ratio >= 1 else "slower"
        factor = ratio if ratio >= 1 else 1 / ratio
        ax.set_title(
            f"{qid} — {subtitle}\n"
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
    quail_runs = [load(workdir / "quail_ringfix_bio2.json")]
    quail_agent = load(workdir / "quail_agent.json")
    stock_vllm = load(workdir / "stock_vllm.json")
    pipelined_vllm = load(workdir / "pipelined_vllm.json")
    stock_vllm_agent = load(workdir / "stock_vllm_agent.json")
    pipelined_vllm_agent = load(workdir / "pipelined_vllm_agent.json")
    pipelined_sglang_clean = load(workdir / "pipelined_sglang_clean.json")
    pipelined_sglang_agent = load(workdir / "pipelined_sglang_agent.json")

    if sol["scale_factor"] != 0.1:
        raise ValueError("unexpected SoL configuration")
    for run in quail_runs:
        if run["model"] != MODEL or run["sf"] != 0.1:
            raise ValueError("unexpected Quail configuration")
    quail_walls = {run["query"]: run["unprofiled"]["engine_wall_s"]
                   for run in quail_runs}
    for summary in (stock_vllm, stock_vllm_agent):
        check(summary, "stock_vllm", "gpu_memory_utilization", 0.91,
              "stage-major")
    for summary in (pipelined_vllm, pipelined_vllm_agent):
        check(summary, "pipelined_vllm", "gpu_memory_utilization",
              0.91, "pipelined")
    for summary in (pipelined_sglang_clean, pipelined_sglang_agent):
        check(summary, "pipelined_sglang", "mem_fraction_static",
              0.76, "pipelined", join_submission="suffix-major-tiled",
              batched_tokens=25_296)

    sources = {
        "BIO-2": {
            "SoL\nestimate":
                sol["queries"]["BIO-2"]["models"][MODEL]["sol_s"],
            "Quail": quail_walls["BIO-2"],
            "stock\nvLLM": entry_for(stock_vllm, "BIO-2"),
            "pipelined\nvLLM": entry_for(pipelined_vllm, "BIO-2"),
            "pipelined\nSGLang": entry_for(pipelined_sglang_clean, "BIO-2"),
        },
        "AGENT-1": {
            "Quail": quail_family_wall(quail_agent, "AGENT-1"),
            "stock\nvLLM": entry_for(stock_vllm_agent, "AGENT-1"),
            "pipelined\nvLLM":
                entry_for(pipelined_vllm_agent, "AGENT-1"),
            "pipelined\nSGLang": entry_for(pipelined_sglang_agent, "AGENT-1"),
        },
    }

    panels = []
    for qid in QUERY_IDS:
        walls = {}
        for system, value in sources[qid].items():
            walls[system] = (value if isinstance(value, (int, float))
                             else value["total_wall_s"])
        sglang_entry = sources[qid]["pipelined\nSGLang"]
        pairs = join_pairs(sglang_entry)
        subtitle = (f"{pairs:,} join pairs" if pairs
                    else f"{filter_documents(sglang_entry):,} documents")
        panels.append((qid, subtitle, walls))

    OUT.mkdir(exist_ok=True)
    figure(panels)

    for qid, subtitle, walls in panels:
        for system, wall in walls.items():
            name = system.replace("\n", " ")
            line = (f"{qid} {name}: {wall:,.2f} s, "
                    f"${usd(wall):.4f}/query")
            entry = sources[qid][system]
            if isinstance(entry, dict):
                pairs = join_pairs(entry)
                rate = ((pairs or filter_documents(entry)) / wall)
                unit = "pairs/s" if pairs else "documents/s"
                accuracy = (entry["accuracy"]["answer_accuracy"]
                            ["accuracy"])
                line += (f", {rate:,.1f} {unit}, "
                         f"answer accuracy {accuracy:.2%}")
            print(line)


if __name__ == "__main__":
    main()
