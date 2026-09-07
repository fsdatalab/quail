"""Extract chronological CPU operations from a saved PyTorch trace."""

import gzip


def nest_intervals(events, start, end):
    """Assign nesting depths and fill gaps outside recorded CPU operations."""
    result = []
    stack = []
    position = start
    for left, right, name in sorted(events, key=lambda event: (event[0], -event[1])):
        left, right = max(start, left), min(end, right)
        if right <= left:
            continue
        while stack and left >= stack[-1]:
            stack.pop()
        if stack and right > stack[-1]:
            if right - stack[-1] > 1e-8:
                raise ValueError(f"CPU intervals cross rather than nest: {name}")
            right = stack[-1]
        if not stack:
            if left > position:
                result.append([position, left, 0, "[no recorded CPU operation]"])
            position = right
        result.append([left, right, len(stack), name])
        stack.append(right)
    if position < end:
        result.append([position, end, 0, "[no recorded CPU operation]"])
    return result


def read_cpu_window(path, join_window_unix_ns, thread_id, start, end):
    """Read nested CPU operations in seconds relative to the join start."""
    import ijson

    with gzip.open(path, "rb") as stream:
        base_ns = next(ijson.items(stream, "baseTimeNanoseconds"))
    join_start_us = (join_window_unix_ns[0] - base_ns) / 1000
    duration = (join_window_unix_ns[1] - join_window_unix_ns[0]) / 1e9
    if not 0 <= start < end <= duration:
        raise ValueError("CPU window must be inside the join")
    events = []
    with gzip.open(path, "rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("ph") != "X" or event.get("tid") != thread_id or "dur" not in event:
                continue
            category, name = event.get("cat"), event.get("name", "")
            if category not in ("cpu_op", "cuda_runtime") and not (
                category == "user_annotation" and name.startswith("vllm.scheduler.")
            ):
                continue
            left = (event["ts"] - join_start_us) / 1e6
            right = (event["ts"] + event["dur"] - join_start_us) / 1e6
            if right > start and left < end:
                events.append((left, right, name))
    return {"start": start, "end": end, "thread_id": thread_id,
            "cpu": nest_intervals(events, start, end)}
