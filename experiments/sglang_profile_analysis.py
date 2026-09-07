"""Read PyTorch traces and measure elapsed GPU activity without double counting."""

from __future__ import annotations

import gzip
from collections import defaultdict


def merge_intervals(intervals):
    """Return the union of elapsed-time intervals."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals):
    """Return elapsed microseconds covered by at least one interval."""
    return sum(end - start for start, end in merge_intervals(intervals))


def clip_interval(interval, window):
    """Return the intersection of an interval and a measurement window."""
    start, end = max(interval[0], window[0]), min(interval[1], window[1])
    return (start, end) if end > start else None


def intersect_intervals(left, right):
    """Return elapsed intervals covered by both sets."""
    left, right = merge_intervals(left), merge_intervals(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        overlap = clip_interval(left[i], right[j])
        if overlap is not None:
            result.append(overlap)
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def read_trace(path, window_unix_ns=None):
    """Summarize CPU scopes, GPU operations, and the capture interval."""
    import ijson

    with gzip.open(path, "rb") as stream:
        base_ns = next(ijson.items(stream, "baseTimeNanoseconds"))
    window = None if window_unix_ns is None else tuple(
        (value - base_ns) / 1000 for value in window_unix_ns
    )
    kernels = []
    transfers = []
    scopes = defaultdict(list)
    kernel_totals = defaultdict(lambda: {"count": 0, "us": 0.0})
    cpu_totals = defaultdict(lambda: {"count": 0, "us": 0.0})
    capture = []
    categories = defaultdict(int)
    event_count = 0
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            event_count += 1
            if event.get("ph") != "X" or "dur" not in event:
                continue
            category = event.get("cat", "")
            categories[category] += 1
            name = event.get("name", "")
            start = float(event["ts"])
            duration = float(event["dur"])
            interval = (start, start + duration)
            if category == "Trace" and "PyTorch Profiler" in name:
                capture.append(interval)
                continue
            if window is not None:
                interval = clip_interval(interval, window)
                if interval is None:
                    continue
                duration = interval[1] - interval[0]
            if category == "kernel":
                kernels.append(interval)
                kernel_totals[name]["count"] += 1
                kernel_totals[name]["us"] += duration
            elif category in ("gpu_memcpy", "gpu_memset"):
                transfers.append(interval)
            elif category in ("cpu_op", "user_annotation", "cuda_runtime"):
                cpu_totals[name]["count"] += 1
                cpu_totals[name]["us"] += duration
                if name.startswith(("scheduler.", "sglang.", "vllm.", "quail.join-")):
                    scopes[name].append(interval)
    if len(capture) != 1:
        raise ValueError(f"expected one PyTorch capture interval in {path}: {capture}")
    capture_start, capture_end = capture[0]
    start, end = window or capture[0]
    if start < capture_start - 1 or end > capture_end + 1:
        raise ValueError(f"measurement window falls outside the capture in {path}")
    busy = merge_intervals(kernels + transfers)
    if busy and (busy[0][0] < start - 1 or busy[-1][1] > end + 1):
        raise ValueError(f"GPU events fall outside the capture interval in {path}")
    gaps = []
    previous = start
    for left, right in busy:
        if left > previous:
            gaps.append((previous, left))
        previous = right
    if previous < end:
        gaps.append((previous, end))
    return {
        "path": str(path), "events": event_count, "categories": dict(categories),
        "base_ns": base_ns, "raw_capture_us": capture_end - capture_start,
        "start_us": start, "end_us": end, "capture_us": end - start,
        "gpu_busy_us": interval_duration(busy),
        "gpu_kernel_us": interval_duration(kernels),
        "gpu_transfer_us": interval_duration(transfers),
        "gpu_busy_intervals": busy, "gpu_gap_intervals": gaps,
        "scopes": dict(scopes),
        "scope_us": {name: interval_duration(spans) for name, spans in scopes.items()},
        "scope_gpu_idle_us": {
            name: interval_duration(intersect_intervals(spans, gaps))
            for name, spans in scopes.items()
        },
        "kernels": dict(kernel_totals), "cpu_ops": dict(cpu_totals),
    }


def binned_activity(intervals, start, end, width_us):
    """Return bin boundaries and occupied microseconds for a timeline."""
    merged = merge_intervals(intervals)
    bins = []
    left = start
    index = 0
    while left < end:
        right = min(end, left + width_us)
        while index < len(merged) and merged[index][1] <= left:
            index += 1
        occupied = 0.0
        for event_start, event_end in merged[index:]:
            if event_start >= right:
                break
            occupied += max(0, min(right, event_end) - max(left, event_start))
        bins.append((left, right, occupied))
        left = right
    return bins


def input_preparation_breakdown(join, trace):
    """Measure preparation stages before the first GPU operation."""
    window = (trace["start_us"], trace["gpu_busy_intervals"][0][0])
    names = {
        "sglang.input.normalize_batch_and_arguments": "normalize_s",
        "sglang.input._batch_tokenize_and_process": "prepare_s",
        "sglang.input._send_batch_request": "send_s",
    }
    spans = defaultdict(list)
    for event in join["input_preparation"]["intervals"]:
        if event["name"] not in names:
            continue
        interval = tuple(
            (event[key] - trace["base_ns"]) / 1000
            for key in ("start_unix_ns", "end_unix_ns")
        )
        clipped = clip_interval(interval, window)
        if clipped is not None:
            spans[names[event["name"]]].append(clipped)
    result = {name: interval_duration(spans[name]) / 1e6 for name in names.values()}
    result["first_gpu_s"] = (window[1] - window[0]) / 1e6
    result["other_s"] = result["first_gpu_s"] - sum(result[name] for name in names.values())
    if result["other_s"] < -1e-6:
        raise ValueError("input preparation stages overlap")
    return result
