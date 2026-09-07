"""Aggregate nested CPU intervals from a PyTorch trace."""

import gzip
import sys


def aggregate_intervals(events, duration_us):
    """Aggregate nested intervals by their recorded operation names."""
    root = {"name": "Worker main thread", "us": duration_us, "calls": 1, "children": {}}
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
            name, {"name": name, "us": 0, "calls": 0, "children": {}},
        )
        node["us"] += end - start
        node["calls"] += 1
        stack.append((end, node))

    def finish(node):
        children = sorted(node["children"].values(), key=lambda child: child["us"], reverse=True)
        self_us = node["us"] - sum(child["us"] for child in children)
        if self_us < -0.01:
            raise ValueError(f"Nested durations exceed their parent: {node['name']}")
        return {
            "name": node["name"], "seconds": node["us"] / 1e6,
            "self_seconds": max(0, self_us) / 1e6, "calls": node["calls"],
            "children": [finish(child) for child in children],
        }

    result = finish(root)
    result["children"].append({
        "name": "[no recorded CPU operation]", "seconds": result["self_seconds"],
        "self_seconds": result["self_seconds"], "calls": 0, "children": [],
    })
    result["self_seconds"] = 0
    result["children"].sort(key=lambda child: child["seconds"], reverse=True)
    return result


def read_cpu_flamegraph(path, window_unix_ns, thread_id):
    """Read recorded operations on the worker's scheduler thread."""
    import ijson

    with gzip.open(path, "rb") as stream:
        base_ns = next(ijson.items(stream, "baseTimeNanoseconds"))
    left, right = ((value - base_ns) / 1000 for value in window_unix_ns)
    events = []
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") != "X" or event.get("tid") != thread_id:
                continue
            category = event.get("cat")
            name = event.get("name", "")
            if category not in ("cpu_op", "cuda_runtime") and not (
                category == "user_annotation" and name.startswith("vllm.scheduler.")
            ):
                continue
            start = max(left, event["ts"])
            end = min(right, event["ts"] + event["dur"])
            if end > start:
                events.append((start, end, sys.intern(name)))
    return aggregate_intervals(events, right - left)
