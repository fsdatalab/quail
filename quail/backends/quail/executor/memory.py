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
        "memory_capacity": {"pages": arena.n_pages,
                            "slots": getattr(arena, "n_state_slots", 0)},
        "peak_gib": peak / 2**30,
    }
    if hasattr(gpu, "pool_bytes"):
        report["memory_bytes"] = memory_breakdown(
            gpu.weights_bytes, gpu.pool_bytes, peak)
    return report


def merge_memory_reports(workers) -> dict:
    """Combine latest worker snapshots without counting earlier rounds twice.

    Capacity and byte totals are sums across devices. Peak bytes are the
    sum of each device's peak, which need not occur at the same time.
    Trace occupancy sums the latest sample from each device; chunk tokens
    and groups describe the current event. Raw samples retain worker IDs
    in memory_workers.
    """
    workers = [dict(worker) for worker in workers]
    records = sorted((row[0], worker, row)
                     for worker, report in enumerate(workers)
                     for row in report.get("memory_trace", ()))
    latest, trace = {}, []
    for stamp, worker, row in records:
        latest[worker] = row
        trace.append([stamp,
                      *(sum(r[i] for r in latest.values()) for i in range(1, 7)),
                      row[7], row[8], sum(r[9] for r in latest.values())])
    events = [[stamp, kind, {**detail, "worker": worker}]
              for worker, report in enumerate(workers)
              for stamp, kind, detail in report.get("memory_events", ())]
    events.sort(key=lambda event: (event[0], event[2]["worker"]))
    result = {
        "memory_workers": workers,
        "memory_trace": trace,
        "memory_trace_started": min((r["memory_trace_started"] for r in workers
                                     if "memory_trace_started" in r), default=0.0),
        "memory_events": events,
        "peak_gib": max((r.get("peak_gib", 0.0) for r in workers), default=0.0),
    }
    for field in ("memory_capacity", "memory_bytes"):
        keys = {key for report in workers for key in report.get(field, {})}
        result[field] = {key: sum(r.get(field, {}).get(key, 0) for r in workers)
                         for key in keys}
    return result
