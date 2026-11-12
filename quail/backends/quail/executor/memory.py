"""GPU memory snapshots shared by local and distributed execution."""

def memory_breakdown(weights_bytes: int, pools: dict, peak_bytes: int) -> dict:
    """Split the peak allocated bytes into weights, KV, state, and the rest.

    Args:
        weights_bytes: Bytes allocated after the model loaded.
        pools: The pool_bytes of the arena.
        peak_bytes: The process's peak allocated bytes, which includes
            kernel warm-up and every query run so far.

    Returns:
        The four numbers and the peak. "other" is the peak minus the
        three pools: activations, temporary copies, and warm-up.
    """
    other = peak_bytes - weights_bytes - pools["kv"] - pools["state"]
    return {"weights": weights_bytes, "kv": pools["kv"],
            "state": pools["state"], "other": other, "peak": peak_bytes}


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
        report["memory_bytes"] = memory_breakdown(
            gpu.weights_bytes, gpu.pool_bytes, peak)
    return report
