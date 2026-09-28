"""Trace one QUAIL-B query's CPU and GPU activity on Modal.

Runs the query through the bench in-process with torch.profiler
recording the physical graph's execution, then prints GPU busy time
against the graph's wall time, the CUDA runtime calls that block the
host, the top CUDA kernels, and the top host operations. Model boot
and the kernel warm-up run before the recording starts. The Chrome
trace goes to the results volume.

    uv run modal run --detach experiments/cells/trace_query.py::trace
        --query BIO-5 --label-scoring trie_paths
"""

from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    image,
    results_vol,
)

TRACE_DIR = "/results/ablations/traces"
BLOCKING = ("cudaStreamSynchronize", "cudaEventSynchronize",
            "cudaDeviceSynchronize", "cudaMalloc", "cudaFree",
            "cudaHostAlloc", "cudaFreeHost", "cudaMemcpy")


def _busy_seconds(events, device_type):
    """Seconds covered by the union of the device's kernel intervals."""
    spans = sorted((event.time_range.start, event.time_range.end)
                   for event in events if event.device_type == device_type)
    busy = 0
    end = None
    for start, stop in spans:
        if end is None or start > end:
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    return busy / 1e6


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def trace_query(query_id: str, sf: float, collection_id: str,
                label_scoring: str, top: int) -> str:
    """Run the query with the graph execution traced; returns a summary."""
    import gzip
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    from torch.profiler import ProfilerActivity, profile

    from quail import EngineConfig
    from quail.backends.quail import worker
    from quail.bench.quailb import run_suite

    config = EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                          device="h100-sxm", label_scoring=label_scoring,
                          gpu_timing=True)
    output = Path(tempfile.mkdtemp()) / "run"
    traces = []
    # the worker binds the graph executor by name at import
    original = worker.execute_single_graph

    def traced(state, settings, graph):
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            report = original(state, settings, graph)
            torch.cuda.synchronize()
        traces.append((prof, report))
        return report

    worker.execute_single_graph = traced
    try:
        suite = run_suite([query_id], sf=sf, config=config,
                          data_dir=Path(DATA_DIR) / f"sf{sf}",
                          ground_truth_collection=collection_id or None,
                          output_dir=output)
    finally:
        worker.execute_single_graph = original
    query = next(q for q in suite["queries"] if q["id"] == query_id)
    measures = query["measurements"]
    nodes = {k: (v.get("gpu_s"), v.get("pack_s"), v.get("chunks"))
             for k, v in measures["node_metrics"].items()
             if k.startswith("ai-")}
    lines = [f"{query_id}: wall {measures['wall_s']:.2f} s, model "
             f"{measures['model_wall_s']:.2f} s, per node (gpu_s, pack_s, "
             f"chunks) {nodes}"]
    Path(TRACE_DIR).mkdir(parents=True, exist_ok=True)
    for index, (prof, report) in enumerate(traces):
        events = prof.events()
        cpu_span = (max(e.time_range.end for e in events)
                    - min(e.time_range.start for e in events)) / 1e6
        gpu_busy = _busy_seconds(events, torch.autograd.DeviceType.CUDA)
        lines.append(f"\n== graph {index}: traced span {cpu_span:.2f} s, "
                     f"GPU busy {gpu_busy:.2f} s, "
                     f"GPU idle {cpu_span - gpu_busy:.2f} s")
        averages = prof.key_averages()
        blocking = [(a.key, a.count, a.self_cpu_time_total / 1e6)
                    for a in averages if a.key.startswith(BLOCKING)]
        lines.append("blocking CUDA runtime calls (name, calls, host s):")
        for key, count, seconds in sorted(blocking, key=lambda x: -x[2]):
            lines.append(f"  {key:28} {count:7} {seconds:8.3f}")
        lines.append(f"top {top} by CUDA time:")
        lines.append(averages.table(sort_by="cuda_time_total", row_limit=top))
        lines.append(f"top {top} by self CPU time:")
        lines.append(averages.table(sort_by="self_cpu_time_total",
                                    row_limit=top))
        path = Path(TRACE_DIR) / f"{query_id}-{label_scoring}-{index}.json"
        prof.export_chrome_trace(str(path))
        with open(path, "rb") as source, gzip.open(f"{path}.gz", "wb") as out:
            shutil.copyfileobj(source, out)
        path.unlink()
        lines.append(f"trace at {path}.gz")
    results_vol.commit()
    return "\n".join(lines)


@app.local_entrypoint()
def trace(query: str = "BIO-5", sf: float = 0.1,
          label_scoring: str = "trie_paths", top: int = 25):
    data = ensure_data.spawn(sf, [query], "")
    print(f"function call id: {data.object_id} (data)", flush=True)
    collection = data.get()
    call = trace_query.spawn(query, sf, collection, label_scoring, top)
    print(f"function call id: {call.object_id} (trace {query})", flush=True)
    print(call.get(), flush=True)
