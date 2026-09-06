"""Plot SGLang join profiles from saved PyTorch traces without inference.

Download the profile and its unprofiled reference from quail-results:

    W=/tmp/quail-sglang-profile; mkdir -p "$W"
    RUN_DIRECTORY=ablations/sglang-join-profile-20260906T225320Z
    uv run modal volume get quail-results "$RUN_DIRECTORY/result.json" "$W/result.json"
    for join in 0 1 2; do
      mkdir -p "$W/join-$join"
      for trace in "join-$join-TP-0.trace.json.gz" driver.trace.json.gz; do
        uv run modal volume get quail-results \
          "$RUN_DIRECTORY/join-$join/$trace" "$W/join-$join/$trace"
      done
    done
    uv run modal volume get quail-results \
      benchmarks/quailb/families/20260906T222629Z-sglang-suffix-major/fever-sglang-process.json \
      "$W/baseline.json"
    uv run --with matplotlib --with ijson python reports/make_sglang_join_profile_plots.py "$W"

CPU scope durations can overlap GPU work. Nested CPU scopes are not additive.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.sglang_profile_analysis import binned_activity, read_trace
from plot_colors import BLUE, GRAY


HERE = Path(__file__).resolve().parent
SCOPES = [
    ("scheduler.get_next_batch_to_run", "Choose next batch", BLUE),
    ("scheduler.process_batch_result", "Process model answers", BLUE),
    ("scheduler.recv_requests", "Poll for requests", GRAY),
    ("scheduler.process_input_requests", "Process incoming requests", GRAY),
    ("scheduler.run_batch", "Submit model computation", GRAY),
]


def load_profiles(root):
    """Read the saved experiment and each scheduler and driver trace."""
    result = json.loads((root / "result.json").read_text())
    baseline = json.loads((root / "baseline.json").read_text())
    assert result["profiled"]
    assert result["model_info"]["model_path"] == "Qwen/Qwen3-4B-FP8"
    assert result["methods"] == ["pipelined_sglang"]
    assert len(result["joins"]) == 3
    profiles = []
    for join in result["joins"]:
        number = join["join"]
        driver = read_trace(root / f"join-{number}" / "driver.trace.json.gz")
        start, end = driver["scopes"][f"quail.join-{number}"][0]
        window = tuple(driver["base_ns"] + round(value * 1000) for value in (start, end))
        assert abs(window[0] - join["started_unix_ns"]) < 100_000_000
        assert abs(window[1] - join["finished_unix_ns"]) < 100_000_000
        scheduler = read_trace(
            root / f"join-{number}" / f"join-{number}-TP-0.trace.json.gz", window,
        )
        assert scheduler["gpu_busy_us"] > 0
        assert scheduler["scope_us"].get("scheduler.get_next_batch_to_run", 0) > 0
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


def main(workdir):
    """Create the diagnostic figure and print trace summaries."""
    result, baseline, profiles = load_profiles(Path(workdir))
    destination = plot_profiles(profiles)
    print(f"Figure: {destination}")
    print(f"Source: {result['result_volume_path']}")
    print(f"Model: {result['model_info']}")
    for join, scheduler, driver in profiles:
        print(json.dumps({
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
            "cpu_process_times": join["process_cpu"],
            "driver_scope_s": {k: v / 1e6 for k, v in driver["scope_us"].items()},
            "top_kernels": sorted(scheduler["kernels"].items(), key=lambda kv: kv[1]["us"], reverse=True)[:10],
            "top_cpu_ops": sorted(scheduler["cpu_ops"].items(), key=lambda kv: kv[1]["us"], reverse=True)[:15],
        }))
    current = result["suites"]["pipelined_sglang"]["passes"]["single"]["queries"][0]
    previous = baseline["suites"]["pipelined_sglang"]["passes"]["single"]["queries"][0]
    print(f"Same accuracy counters: {current['accuracy'] == previous['accuracy']}")
    assert current["accuracy"] == previous["accuracy"]
    for name in ("fresh_tokens", "cached_tokens", "regret_tokens", "rows"):
        print(f"Same {name}: {current[name] == previous[name]}")
        assert current[name] == previous[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    main(parser.parse_args().workdir)
