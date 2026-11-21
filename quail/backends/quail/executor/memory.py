"""GPU memory snapshots shared by local and distributed execution."""

def memory_report(gpu) -> dict:
    """Collect one worker's cumulative query memory measurements."""
    arena = gpu.arena
    peak = int(gpu.torch.cuda.max_memory_allocated())
    report = {
        "memory_trace": list(getattr(arena, "memory_trace", ())),
        "memory_trace_started": getattr(arena, "memory_trace_started", 0.0),
        "memory_events": [list(event) for event in getattr(arena, "events", ())],
        "memory_capacity": {"pages": arena.num_pages,
            "slots": getattr(arena, "num_state_slots", 0)},
        "peak_gib": peak / 2**30,
    }
    if hasattr(gpu, "pool_bytes"):
        weights_bytes, pools = gpu.weights_bytes, gpu.pool_bytes
        other = peak - weights_bytes - pools["kv"] - pools["state"]
        report["memory_bytes"] = {
            "weights": weights_bytes, "kv": pools["kv"],
            "state": pools["state"], "other": other, "peak": peak}
    return report
