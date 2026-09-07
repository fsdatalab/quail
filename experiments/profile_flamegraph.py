"""Aggregate nested CPU intervals from a PyTorch trace."""

import gzip
import sys
from bisect import bisect_right

from experiments.sglang_profile_analysis import merge_intervals


def aggregate_intervals(events, duration_us, gpu_intervals=()):
    """Aggregate nested CPU intervals and their overlap with GPU activity."""
    gpu = merge_intervals(gpu_intervals)
    starts = [start for start, _ in gpu]
    cumulative = [0.0]
    for start, end in gpu:
        cumulative.append(cumulative[-1] + end - start)

    def gpu_before(timestamp):
        index = bisect_right(starts, timestamp) - 1
        if index < 0:
            return 0.0
        start, end = gpu[index]
        return cumulative[index] + min(timestamp, end) - start

    def overlap(start, end):
        return gpu_before(end) - gpu_before(start)

    root = {"name": "Worker main thread", "us": duration_us,
            "gpu_us": overlap(0, duration_us), "calls": 1, "children": {}}
    stack = []
    for start, end, name in sorted(events, key=lambda event: (event[0], -event[1])):
        if end <= start:
            continue
        while stack and start >= stack[-1][0]:
            stack.pop()
        if stack and end > stack[-1][0]:
            if end - stack[-1][0] > 0.01:
                raise ValueError(f"CPU intervals cross rather than nest: {name}")
            end = stack[-1][0]
        parent = stack[-1][1] if stack else root
        node = parent["children"].setdefault(
            name, {"name": name, "us": 0, "gpu_us": 0, "calls": 0, "children": {}},
        )
        node["us"] += end - start
        node["gpu_us"] += overlap(start, end)
        node["calls"] += 1
        stack.append((end, node))

    def finish(node):
        children = sorted(node["children"].values(), key=lambda child: child["us"], reverse=True)
        self_us = node["us"] - sum(child["us"] for child in children)
        self_gpu_us = node["gpu_us"] - sum(child["gpu_us"] for child in children)
        if self_us < -0.01:
            raise ValueError(f"Nested durations exceed their parent: {node['name']}")
        if not -0.01 <= self_gpu_us <= self_us + 0.01:
            raise ValueError(f"GPU overlap exceeds CPU self time: {node['name']}")
        return {
            "name": node["name"], "seconds": node["us"] / 1e6,
            "self_seconds": max(0, self_us) / 1e6, "calls": node["calls"],
            "gpu_seconds": node["gpu_us"] / 1e6,
            "gpu_idle_seconds": max(0, node["us"] - node["gpu_us"]) / 1e6,
            "self_gpu_seconds": max(0, self_gpu_us) / 1e6,
            "children": [finish(child) for child in children],
        }

    result = finish(root)
    result["children"].append({
        "name": "[no recorded CPU operation]", "seconds": result["self_seconds"],
        "self_seconds": result["self_seconds"], "calls": 0, "children": [],
        "gpu_seconds": result["self_gpu_seconds"],
        "gpu_idle_seconds": result["self_seconds"] - result["self_gpu_seconds"],
        "self_gpu_seconds": result["self_gpu_seconds"],
    })
    result["self_seconds"] = 0
    result["self_gpu_seconds"] = 0
    result["children"].sort(key=lambda child: child["seconds"], reverse=True)
    return result


def read_cpu_flamegraph(path, window_unix_ns, thread_id):
    """Read recorded operations on the worker's scheduler thread."""
    import ijson

    with gzip.open(path, "rb") as stream:
        base_ns = next(ijson.items(stream, "baseTimeNanoseconds"))
    left, right = ((value - base_ns) / 1000 for value in window_unix_ns)
    events = []
    gpu = []
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") != "X" or "dur" not in event:
                continue
            category = event.get("cat")
            name = event.get("name", "")
            start = max(left, event["ts"])
            end = min(right, event["ts"] + event["dur"])
            if end <= start:
                continue
            if category in ("kernel", "gpu_memcpy", "gpu_memset"):
                gpu.append((start - left, end - left))
                continue
            if event.get("tid") != thread_id:
                continue
            if category not in ("cpu_op", "cuda_runtime") and not (
                category == "user_annotation" and name.startswith("vllm.scheduler.")
            ):
                continue
            events.append((start - left, end - left, sys.intern(name)))
    return aggregate_intervals(events, right - left, gpu)
