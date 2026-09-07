r"""Plot query time before and after the engine image upgrade.

Compares the six-query run on vLLM 0.28.0, SGLang 0.5.19, and CUDA
13.3.1 with the same queries from the full run on vLLM 0.26.0, SGLang
0.5.18, and CUDA 13.0.1. Pull both runs' manifests and per method suite
files into two work directories:

    for pair in "OLD 20260905T021527Z" "NEW 20260906T011248Z"; do
      set -- $pair; D=<workdir>/$1; mkdir -p $D
      R=benchmarks/quailb/family-runs/$2-quailb-sf0.1-lf1-qwen3-4b-fp8-families
      modal volume get quail-results $R/manifest.json $D/manifest.json
      for m in quail stock_vllm pipelined_vllm pipelined_sglang; do
        p=$(python3 -c "import json; m = json.load(open('$D/manifest.json')); \
          print(m['result_volume_paths']['$m'].removeprefix('/results/'))")
        modal volume get quail-results $p $D/$m.json
      done
    done
    uv run --with matplotlib python reports/make_engine_upgrade_plots.py \
      <workdir>/OLD <workdir>/NEW

The queries compared are the ones the new run contains. The script
prints the per query table and writes plots/engine_upgrade_check.png.
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
from plot_colors import BLUE, GRAY  # noqa: E402

METHODS = (
    ("Quail", "quail"),
    ("Stock vLLM", "stock_vllm"),
    ("Pipelined vLLM", "pipelined_vllm"),
    ("Pipelined SGLang", "pipelined_sglang"),
)


def load_rows(workdir, key):
    with (workdir / f"{key}.json").open() as source:
        suite = json.load(source)
    return {row["query"]: row for row in suite["passes"]["single"]["queries"]}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} OLD_WORKDIR NEW_WORKDIR")
    old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
    with (new_dir / "manifest.json").open() as source:
        queries = list(json.load(source)["query_ids"])
    runs = {label: (load_rows(old_dir, key), load_rows(new_dir, key))
            for label, key in METHODS}

    print("| Method | Query | Old images (s) | New images (s) | Change | "
          "Boot, new | Accuracy old | Accuracy new |")
    print("|---|---|---:|---:|---:|---|---:|---:|")
    for label, (old, new) in runs.items():
        for query in queries:
            o, n = old[query], new[query]
            print(f"| {label} | {query} | {o['wall_s']:.2f} | {n['wall_s']:.2f} "
                  f"| {100 * (n['wall_s'] / o['wall_s'] - 1):+.1f}% "
                  f"| {n['boot_kind']} {n['boot_s']:.0f} s "
                  f"| {o['accuracy']['answer_accuracy']['accuracy']:.4f} "
                  f"| {n['accuracy']['answer_accuracy']['accuracy']:.4f} |")
        total_old = sum(old[q]["wall_s"] for q in queries)
        total_new = sum(new[q]["wall_s"] for q in queries)
        print(f"| {label} | total | {total_old:.2f} | {total_new:.2f} "
              f"| {100 * (total_new / total_old - 1):+.1f}% | | | |")

    fig, axes = plt.subplots(1, len(METHODS), figsize=(18, 5.2), sharey=False)
    width = 0.38
    x = np.arange(len(queries))
    for ax, (label, (old, new)) in zip(axes, runs.items()):
        old_values = [old[q]["wall_s"] for q in queries]
        new_values = [new[q]["wall_s"] for q in queries]
        ax.bar(x - width / 2, old_values, width, color=GRAY,
               label="vLLM 0.26.0 / SGLang 0.5.18 / CUDA 13.0.1")
        ax.bar(x + width / 2, new_values, width, color=BLUE,
               label="vLLM 0.28.0 / SGLang 0.5.19 / CUDA 13.3.1")
        for index, (o, n) in enumerate(zip(old_values, new_values)):
            ax.annotate(f"{100 * (n / o - 1):+.0f}%",
                        (index + width / 2, n), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, queries, rotation=30, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("seconds per query (log scale)")
        ax.set_title(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Query time before and after the engine image upgrade, "
                 "Qwen3 4B fp8, one H100")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    OUT.mkdir(exist_ok=True)
    output = OUT / "engine_upgrade_check.png"
    fig.savefig(output, dpi=300)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
