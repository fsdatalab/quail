"""Compare SGLang and vLLM join profiles from saved PyTorch traces.

Download the profile and its unprofiled reference from quail-results:

    W=/tmp/quail-join-profiles; mkdir -p "$W/sglang"
    RUN_DIRECTORY=ablations/sglang-join-profile-20260906T225320Z
    uv run modal volume get quail-results "$RUN_DIRECTORY/result.json" "$W/sglang/result.json"
    for join in 0 1 2; do
      mkdir -p "$W/sglang/join-$join"
      for trace in "join-$join-TP-0.trace.json.gz" driver.trace.json.gz; do
        uv run modal volume get quail-results \
          "$RUN_DIRECTORY/join-$join/$trace" "$W/sglang/join-$join/$trace"
      done
    done
    uv run modal volume get quail-results \
      benchmarks/quailb/families/20260906T222629Z-sglang-suffix-major/fever-sglang-process.json \
      "$W/sglang/baseline.json"
    mkdir -p "$W/vllm" "$W/sglang-input"
    VLLM_RUN=ablations/vllm-join-profile-20260906T233246Z
    INPUT_RUN=ablations/sglang-input-profile-20260906T233056Z
    uv run modal volume get quail-results "$VLLM_RUN/result.json" "$W/vllm/result.json"
    uv run modal volume get quail-results "$INPUT_RUN/result.json" "$W/sglang-input/result.json"
    for join in 0 1 2; do
      mkdir -p "$W/vllm/join-$join" "$W/sglang-input/join-$join"
      for trace in worker.trace.json.gz driver.trace.json.gz; do
        uv run modal volume get quail-results \
          "$VLLM_RUN/join-$join/$trace" "$W/vllm/join-$join/$trace"
      done
      for trace in "join-$join-TP-0.trace.json.gz" driver.trace.json.gz; do
        uv run modal volume get quail-results \
          "$INPUT_RUN/join-$join/$trace" "$W/sglang-input/join-$join/$trace"
      done
    done
    uv run modal volume get quail-results \
      benchmarks/quailb/runs/qb_20260906T211500Z_37db8090/20260906T211500Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_vllm.json \
      "$W/vllm/baseline.json"
    cp "$W/sglang/baseline.json" "$W/sglang-input/baseline.json"
    uv run modal volume get quail-results "$INPUT_RUN/join-0/input-preparation.pstats" \
      "$W/sglang-input/join-0/input-preparation.pstats"
    uv run --with matplotlib --with ijson python reports/make_join_profile_plots.py "$W"

CPU scope durations can overlap GPU work. Nested CPU scopes are not additive.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.sglang_profile_analysis import binned_activity, input_preparation_breakdown, read_trace
from plot_colors import BLUE, GRAY


HERE = Path(__file__).resolve().parent
SCOPES = [
    ("scheduler.get_next_batch_to_run", "scheduler.get_next_batch_to_run", BLUE),
    ("scheduler.process_batch_result", "scheduler.process_batch_result", BLUE),
    ("scheduler.recv_requests", "scheduler.recv_requests", GRAY),
    ("scheduler.process_input_requests", "scheduler.process_input_requests", GRAY),
    ("scheduler.run_batch", "scheduler.run_batch", GRAY),
]


def load_profiles(root, backend="sglang"):
    """Read the saved experiment and each scheduler and driver trace."""
    result = json.loads((root / "result.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    assert result["profiled"]
    assert result["model_info"]["model_path"] == "Qwen/Qwen3-4B-FP8"
    assert result["methods"] == [f"pipelined_{backend}"]
    assert len(result["joins"]) == 3
    profiles = []
    for join in result["joins"]:
        number = join["join"]
        driver = read_trace(root / f"join-{number}" / "driver.trace.json.gz")
        start, end = driver["scopes"][f"quail.join-{number}"][0]
        window = tuple(driver["base_ns"] + round(value * 1000) for value in (start, end))
        assert abs(window[0] - join["started_unix_ns"]) < 100_000_000
        assert abs(window[1] - join["finished_unix_ns"]) < 100_000_000
        filename = f"join-{number}-TP-0.trace.json.gz" if backend == "sglang" else "worker.trace.json.gz"
        scheduler = read_trace(root / f"join-{number}" / filename, window)
        assert scheduler["gpu_busy_us"] > 0
        scope = "scheduler.get_next_batch_to_run" if backend == "sglang" else "vllm.scheduler.schedule"
        if not result.get("input_detail"):
            assert scheduler["scope_us"].get(scope, 0) > 0
        assert driver["scope_us"].get(f"quail.join-{number}", 0) > 0
        profiles.append((join, scheduler, driver))
    return result, baseline, profiles


def plot_profiles(profiles):
    """Plot GPU activity over time and elapsed CPU scopes for each join."""
    plt.style.use(HERE / "quail.mplstyle")
    figure, axes = plt.subplots(3, 2, figsize=(15, 13))
    figure.set_layout_engine("none")
    for (join, scheduler, _), (timeline, cpu) in zip(profiles, axes):
        number = join["join"] + 1
        bins = binned_activity(
            scheduler["gpu_busy_intervals"], scheduler["start_us"], scheduler["end_us"], 500_000,
        )
        times = [(left - scheduler["start_us"]) / 1e6 for left, _, _ in bins]
        heights = [100 * occupied / (right - left) for left, right, occupied in bins]
        widths = [(right - left) / 1e6 for left, right, _ in bins]
        timeline.bar(times, heights, width=widths, align="edge", color=BLUE)
        share = 100 * scheduler["gpu_busy_us"] / scheduler["capture_us"]
        timeline.set_title(f"FEV-9 join {number}: GPU activity, {share:.1f}% of join")
        timeline.set_xlabel("seconds")
        timeline.set_ylabel("percent")
        timeline.set_ylim(0, 105)
        timeline.set_yticks([0, 25, 50, 75, 100])
        selected = [(name, label, color) for name, label, color in SCOPES
                    if name in scheduler["scope_us"]]
        values = [scheduler["scope_us"][name] / 1e6 for name, _, _ in selected]
        cpu.barh(range(len(selected)), values, color=[color for _, _, color in selected])
        cpu.set_yticks(range(len(selected)), [label for _, label, _ in selected])
        cpu.invert_yaxis()
        cpu.set_xlabel("seconds")
        cpu.set_title(f"FEV-9 join {number}: CPU scopes")
        cpu.set_xlim(0, max(values) * 1.18)
        for index, value in enumerate(values):
            cpu.text(value + max(values) * 0.02, index, f"{value:.2f}", va="center", fontsize=10)
    figure.suptitle("FEV-9, SGLang PyTorch profile, Qwen3 4B FP8, one H100", fontsize=17, y=0.975)
    figure.text(
        0.07, 0.025,
        "GPU activity is the union of kernels and transfers in each 0.5-second bin. It is not hardware occupancy.\n"
        "CPU scopes measure elapsed time and may overlap GPU work.\n"
        "Join intervals exclude trace export. Profiling overhead remains; benchmark measurements are unchanged.",
        fontsize=10, linespacing=1.5,
    )
    figure.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.13, hspace=0.55, wspace=0.70)
    destination = HERE / "plots" / "sglang_join_profile.png"
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def summarize(result, baseline, profiles):
    """Print elapsed trace measurements and saved-answer comparisons."""
    print(f"Source: {result['result_volume_path']}")
    print(f"Model: {result['model_info']}")
    for join, scheduler, driver in profiles:
        print(json.dumps({
            "method": result["methods"][0],
            "join": join["join"], "generate_wall_s": join["generate_wall_s"],
            "capture_s": scheduler["capture_us"] / 1e6,
            "gpu_busy_s": scheduler["gpu_busy_us"] / 1e6,
            "initial_gpu_idle_s": (scheduler["gpu_busy_intervals"][0][0] - scheduler["start_us"]) / 1e6,
            "largest_gpu_gaps_s": sorted(
                ((left - scheduler["start_us"]) / 1e6, (right - left) / 1e6)
                for left, right in scheduler["gpu_gap_intervals"]
                if right - left > 500_000
            ),
            "scope_s": {k: v / 1e6 for k, v in scheduler["scope_us"].items()},
            "scope_gpu_idle_s": {k: v / 1e6 for k, v in scheduler["scope_gpu_idle_us"].items()},
            "cpu_process_times": join.get("process_cpu"),
            "driver_scope_s": {k: v / 1e6 for k, v in driver["scope_us"].items()},
            "top_kernels": sorted(scheduler["kernels"].items(), key=lambda kv: kv[1]["us"], reverse=True)[:10],
            "top_cpu_ops": sorted(scheduler["cpu_ops"].items(), key=lambda kv: kv[1]["us"], reverse=True)[:15],
        }))
    method = result["methods"][0]
    current = result["suites"][method]["passes"]["single"]["queries"][0]
    previous_suite = baseline.get("suites", {}).get(method, baseline)
    previous = previous_suite["passes"]["single"]["queries"][0]
    print(f"Same accuracy counters: {current['accuracy'] == previous['accuracy']}")
    for name in ("fresh_tokens", "cached_tokens", "regret_tokens", "rows"):
        print(f"Same {name}: {current[name] == previous[name]}")


def plot_comparison(sglang_profiles, vllm_profiles):
    """Compare GPU activity using identical time and percentage scales."""
    plt.style.use(HERE / "quail.mplstyle")
    figure, axes = plt.subplots(3, 2, figsize=(14, 12))
    figure.set_layout_engine("none")
    groups = [("SGLang", sglang_profiles, BLUE), ("vLLM", vllm_profiles, GRAY)]
    limit = max(trace["capture_us"] / 1e6 for _, profiles, _ in groups for _, trace, _ in profiles)
    for column, (name, profiles, color) in enumerate(groups):
        for row, (join, trace, _) in enumerate(profiles):
            axis = axes[row, column]
            bins = binned_activity(trace["gpu_busy_intervals"], trace["start_us"], trace["end_us"], 500_000)
            times = [(left - trace["start_us"]) / 1e6 for left, _, _ in bins]
            heights = [100 * occupied / (right - left) for left, right, occupied in bins]
            widths = [(right - left) / 1e6 for left, right, _ in bins]
            axis.bar(times, heights, width=widths, align="edge", color=color)
            share = 100 * trace["gpu_busy_us"] / trace["capture_us"]
            axis.set_title(f"FEV-9 join {join['join'] + 1}, {name}, GPU active {share:.1f}%")
            axis.set(xlabel="seconds", ylabel="percent", xlim=(0, limit), ylim=(0, 105))
            axis.set_yticks([0, 25, 50, 75, 100])
    figure.suptitle("FEV-9 join profiles, Qwen3 4B FP8, one H100 per engine", fontsize=16, y=0.975)
    figure.text(0.07, 0.025,
                "Each bar shows time covered by GPU kernels or transfers within 0.5 seconds. Overlapping GPU work counts once.\n"
                "The traces include profiling overhead and exclude trace export. Each engine evaluates its own filter survivors.",
                fontsize=10, linespacing=1.5)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.13, hspace=0.55, wspace=0.25)
    path = HERE / "plots" / "join_profile_comparison.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def main(workdir):
    """Read saved traces and create the matching join figures."""
    root = Path(workdir)
    sg = load_profiles(root / "sglang")
    vl = load_profiles(root / "vllm", "vllm")
    print(f"Figure: {plot_profiles(sg[2])}")
    print(f"Figure: {plot_comparison(sg[2], vl[2])}")
    summarize(*sg)
    summarize(*vl)
    inputs = load_profiles(root / "sglang-input")
    for join, trace, _ in inputs[2]:
        print(json.dumps({"input_join": join["join"], **input_preparation_breakdown(join, trace)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    main(parser.parse_args().workdir)
