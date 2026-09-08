"""Save GPU activity and a midpoint CPU window from a downloaded trace.

    uv run --with ijson python -m experiments.analyze_blog_profiles \
      /tmp/quail-blog-profiles/pipelined_vllm/AGENT-1 \
      /results/ablations/vllm-join-profile-20260907T214358Z/result.json \
      ablations/vllm-join-profile-20260907T214358Z/blog-analysis.json

The workdir contains worker.trace.json.gz and phase.json or result.json.
Derived data is uploaded to quail-results before plots read it.
"""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from experiments.profile_cpu_timeline import read_cpu_window
from experiments.profile_gpu_timeline import read_gpu_intervals
from quail_bench.labels import ModalVolumeFiles


def main_thread(path):
    """Select the CPU thread with the most recorded CUDA API calls."""
    import ijson

    counts = Counter()
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") == "X" and event.get("cat") == "cuda_runtime":
                counts[event["tid"]] += 1
    if not counts:
        raise ValueError(f"No CUDA API calls in {path}")
    return counts.most_common(1)[0][0]


def analyze(directory, source):
    """Compute exact GPU activity and a five-second midpoint CPU window."""
    if (directory / "phase.json").exists():
        phase = json.loads((directory / "phase.json").read_text())
    else:
        result = json.loads((directory / "result.json").read_text())
        kind = "filter" if "filters" in result else "join"
        phase = {**result[f"{kind}s"][0], "query": result["query"],
                 "method": "pipelined_vllm", "phase": kind,
                 "model_info": result["model_info"]}
    path = directory / "worker.trace.json.gz"
    bounds = [phase["started_unix_ns"], phase["finished_unix_ns"]]
    duration = (bounds[1] - bounds[0]) / 1e9
    gpu = read_gpu_intervals(path, bounds)
    if not gpu:
        raise ValueError(f"No GPU events inside {bounds} in {path}")
    start = max(0, math.floor(duration / 2 - 2.5))
    end = min(duration, start + 5)
    window = read_cpu_window(path, bounds, main_thread(path), start, end)
    window["gpu"] = [(max(start, left / 1e6), min(end, right / 1e6))
                     for left, right in gpu if right / 1e6 > start and left / 1e6 < end]
    summary = {
        "source": source, "phase": phase, "duration_s": duration,
        "gpu_active_s": sum(right - left for left, right in gpu) / 1e6,
        "first_gpu_s": gpu[0][0] / 1e6,
        "window_gpu_active_s": sum(right - left for left, right in window["gpu"]),
        "window": window,
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("source")
    parser.add_argument("output_volume_path")
    args = parser.parse_args()
    summary = analyze(args.workdir, args.source)
    ModalVolumeFiles().write_json(args.output_volume_path, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "window"}))
