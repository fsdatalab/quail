"""Plot the BIO-3 vLLM join profile from quail-results volume files.

    W=/tmp/quail-bio3-vllm-profile
    mkdir -p "$W/join-0"
    RUN=ablations/vllm-join-profile-20260907T013304Z
    uv run modal volume get quail-results "$RUN/result.json" "$W/result.json"
    for trace in worker.trace.json.gz driver.trace.json.gz; do
      uv run modal volume get quail-results "$RUN/join-0/$trace" "$W/join-0/$trace"
    done
    uv run modal volume get quail-results \
      benchmarks/quailb/runs/qb_20260905T024836Z_a36d4647/single/BIO-3.json \
      "$W/pipelined_vllm-raw.json"
    uv run modal volume get quail-results \
      benchmarks/quailb/runs/qb_20260905T021548Z_43ca0948/single/BIO-3.json \
      "$W/quail-raw.json"
    uv run --with matplotlib --with ijson python reports/make_bio3_join_profile_plots.py "$W"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.sglang_profile_analysis import binned_activity, read_trace
from plot_colors import BLUE, GRAY
from quail.bench.evaluate import H100_USD_PER_HOUR


HERE = Path(__file__).resolve().parent
SCOPES = [
    ("vllm.scheduler.schedule", "Choose next batch"),
    ("vllm.scheduler.add_request", "Add request"),
    ("vllm.scheduler.update_from_output", "Process model answers"),
]


def load_profile(root):
    """Read the worker trace within the driver's join interval."""
    result = json.loads((root / "result.json").read_text())
    assert result["query"] == "BIO-3"
    assert result["methods"] == ["pipelined_vllm"]
    assert len(result["joins"]) == 1
    driver = read_trace(root / "join-0/driver.trace.json.gz")
    start, end = driver["scopes"]["quail.join-0"][0]
    window = tuple(driver["base_ns"] + round(value * 1000) for value in (start, end))
    trace = read_trace(root / "join-0/worker.trace.json.gz", window)
    assert trace["gpu_busy_us"] > 0
    assert trace["scope_us"].get("vllm.scheduler.schedule", 0) > 0
    return result, trace


def plot_profile(trace):
    """Plot GPU activity and scheduling time during the join."""
    plt.style.use(HERE / "quail.mplstyle")
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    figure.set_layout_engine("none")
    timeline, cpu = axes
    bins = binned_activity(
        trace["gpu_busy_intervals"], trace["start_us"], trace["end_us"], 2_000_000,
    )
    timeline.bar(
        [(left - trace["start_us"]) / 1e6 for left, _, _ in bins],
        [100 * occupied / (right - left) for left, right, occupied in bins],
        width=[(right - left) / 1e6 for left, right, _ in bins],
        align="edge", color=BLUE,
    )
    share = 100 * trace["gpu_busy_us"] / trace["capture_us"]
    timeline.set(
        title=f"BIO-3 join: GPU operations cover {share:.1f}% of the interval",
        xlabel="seconds", ylabel="percent", ylim=(0, 105),
        xlim=(0, trace["capture_us"] / 1e6),
    )
    values = [trace["scope_us"][name] / 1e6 for name, _ in SCOPES]
    idle = [trace["scope_gpu_idle_us"][name] / 1e6 for name, _ in SCOPES]
    cpu.barh([i - 0.18 for i in range(len(SCOPES))], values, height=0.32,
             color=GRAY, label="All elapsed time")
    cpu.barh([i + 0.18 for i in range(len(SCOPES))], idle, height=0.32,
             color=BLUE, label="While GPU idle")
    for offset, numbers in [(-0.18, values), (0.18, idle)]:
        for index, value in enumerate(numbers):
            cpu.text(value + max(values) * 0.015, index + offset,
                     f"{value:.2f}", va="center", fontsize=10)
    cpu.set_yticks(range(len(SCOPES)), [label for _, label in SCOPES])
    cpu.set(xlabel="seconds", title="BIO-3 join: elapsed scheduler operations",
            xlim=(0, max(values) * 1.15))
    cpu.invert_yaxis()
    cpu.legend(loc="lower right", frameon=False)
    figure.suptitle("BIO-3, pipelined vLLM, Qwen3 4B FP8, one H100", fontsize=16, y=0.98)
    figure.text(0.18, 0.025,
                "GPU activity counts elapsed kernels and transfers once, in 2-second bins.\n"
                "CPU intervals can overlap GPU operations. Profiling overhead remains; trace export is excluded.",
                fontsize=10, linespacing=1.5)
    figure.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.13, hspace=0.55)
    destination = HERE / "plots/bio3_join_profile"
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def summarize(root, result, trace):
    """Print trace measurements and comparisons with saved results."""
    current = result["suites"]["pipelined_vllm"]["passes"]["single"]["queries"][0]
    join = result["joins"][0]
    print(json.dumps({
        "source": result["result_volume_path"],
        "join": join,
        "capture_s": trace["capture_us"] / 1e6,
        "gpu_busy_s": trace["gpu_busy_us"] / 1e6,
        "gpu_active_percent": 100 * trace["gpu_busy_us"] / trace["capture_us"],
        "first_gpu_s": (trace["gpu_busy_intervals"][0][0] - trace["start_us"]) / 1e6,
        "scope_s": {k: v / 1e6 for k, v in trace["scope_us"].items()},
        "scope_gpu_idle_s": {k: v / 1e6 for k, v in trace["scope_gpu_idle_us"].items()},
        "largest_gaps": sorted(
            [(left - trace["start_us"]) / 1e6, (right - left) / 1e6]
            for left, right in trace["gpu_gap_intervals"] if right - left > 1e6
        ),
        "top_kernels": sorted(trace["kernels"].items(), key=lambda item: item[1]["us"], reverse=True)[:12],
        "top_cpu_ops": sorted(trace["cpu_ops"].items(), key=lambda item: item[1]["us"], reverse=True)[:15],
        "current": current,
    }, indent=2))
    for method in ("quail", "pipelined_vllm"):
        raw = json.loads((root / f"{method}-raw.json").read_text())
        previous = raw["engine_report"]
        pairs = sum(stage.get("tuples", 0) for stage in previous["stages"])
        print(method, json.dumps({
            "runtime_s": previous["wall_s"], "pairs": pairs,
            "pairs_per_second": pairs / previous["wall_s"],
            "gpu_cost_usd": previous["wall_s"] / 3600 * H100_USD_PER_HOUR,
            "fresh_tokens": previous["fresh_tokens"],
            "regret_tokens": previous["regret_tokens"],
            "answer_accuracy": raw["accuracy"]["answer_accuracy"]["accuracy"],
            "output_accuracy": raw["accuracy"]["output_accuracy"],
            "same_accuracy_as_profile": raw["accuracy"] == current["accuracy"],
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    root = parser.parse_args().workdir
    result, trace = load_profile(root)
    plot_profile(trace)
    summarize(root, result, trace)
