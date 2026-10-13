"""Compare eager and captured extraction forwards on one H100.

    uv run modal run --detach -m experiments.compare_extract_cuda_graphs \
        2>&1 | tee /tmp/compare_extract_cuda_graphs.log

Results and traces live in the quail-results volume. Retrieve the return
value with modal.FunctionCall.from_id using the printed call ID.
"""

import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import modal

from quail.deploy.images import gpu_image

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
    "/results": modal.Volume.from_name("quail-results"),
}


def decode_trace_metrics(events):
    """Measure the short-forward tail and idle gaps inside its forwards."""
    from experiments.profile_extract import _merge

    gpu = [event for event in events if event.get("ph") == "X"
           and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    launches = [event for event in events
                if event.get("name") == "cudaGraphLaunch"]
    windows = []
    if launches:
        for launch in launches:
            own = [event for event in gpu if event["args"].get("correlation")
                   == launch["args"]["correlation"]]
            windows.append((min(e["ts"] for e in own),
                            max(e["ts"] + e["dur"] for e in own)))
    else:
        launch_times = {event["args"]["correlation"]: event["ts"]
                        for event in events
                        if event.get("cat") in ("cuda_runtime", "cuda_driver")
                        and "correlation" in event.get("args", {})}
        starts = sorted(e["ts"] for e in events
                        if e.get("name") == "aten::embedding")
        ends = sorted(e["ts"] for e in events
                      if e.get("name") == "aten::linear")
        for start, end in zip(starts, ends):
            own = [e for e in gpu if start <= launch_times.get(
                e["args"].get("correlation"), -1) < end]
            if any("qk_norm_rope" in e["name"]
                   and e["args"]["grid"][0] <= 512 for e in own):
                windows.append((min(e["ts"] for e in own),
                                max(e["ts"] + e["dur"] for e in own)))

    def active(start, end):
        return sum(b - a for a, b in _merge([
            (max(start, e["ts"]), min(end, e["ts"] + e["dur"])) for e in gpu]))

    first = min(a for a, _ in windows)
    last = max(e["ts"] + e["dur"] for e in gpu)
    duration = last - first
    idle = duration - active(first, last)
    forward_idle = sum(b - a - active(a, b) for a, b in windows)
    return dict(forwards=len(windows), wall_ms=duration / 1000,
                gpu_idle_ms=idle / 1000, gpu_idle_fraction=idle / duration,
                forward_idle_ms=forward_idle / 1000,
                outside_forward_idle_ms=(idle - forward_idle) / 1000)


@app.function(image=gpu_image(), gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def compare(run: str):
    """Check answers and spans, then measure alternating warmed queries."""
    import pyarrow.parquet as pq
    import torch

    import quail
    from bench import quailb
    from experiments.compare_extract_tokenization import _trace_metrics
    from quail.backends.quail.executor import stages
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b import get_query
    from quail_b.data import build_sets
    from quail_b.minimum import token_metrics

    out = Path(f"/results/ablations/extract-cuda-graphs-{run}")
    out.mkdir(parents=True)
    data = build_sets("/root/.cache/huggingface/quail-b-tables", 0.1)
    tables = {name: pq.read_table(data / f"{name}.parquet")
              for name in ("contracts", "clause_types")}
    VOLUMES["/root/.cache/huggingface"].commit()
    spec = get_query("CUAD-3")
    session = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"))
    for name, table in tables.items():
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    configuration = "eager"
    validate = False
    managers = []
    checked = 0
    original = stages.forward_chunk

    def forward(pipeline, arena, chunk):
        nonlocal checked
        graphs = pipeline.decode_graphs
        if graphs not in managers:
            managers.append(graphs)
        if validate and graphs.graphs and graphs.supports(chunk):
            expected = pipeline.forward_chunk(chunk)
            actual = graphs.forward(chunk)
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.02)
            checked += 1
        previous = graphs.enabled
        graphs.enabled = configuration == "graphs"
        try:
            return original(pipeline, arena, chunk)
        finally:
            graphs.enabled = previous

    stages.forward_chunk = forward
    reference = None
    fresh = None
    stores = {}

    def execute(mode, name):
        nonlocal configuration, reference, fresh
        configuration = mode
        for manager in managers:
            manager.replays = manager.fallbacks = 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function("query_execution"):
            query = quailb.build_query(session, spec)
            result = query.run()
            rows = result.collect()
            torch.cuda.synchronize()
        startup = result.report.get("boot_s") or 0.0
        wall = time.perf_counter() - started - startup
        if reference is None:
            reference, fresh = rows, result.report["fresh_tokens"]
        if not rows.equals(reference) or fresh != result.report["fresh_tokens"]:
            raise AssertionError(f"{name}: answers, spans, or token counts changed")
        pq.write_table(rows, out / f"{name}.parquet")
        output = quailb.run_output(result, spec.info, tables)
        output.prompt_pieces = quailb.prompt_pieces(
            query, spec.info, quailb.join_anchors(result))
        metrics = token_metrics(spec, output, tables, stores)
        record = dict(configuration=mode, name=name, query_s=wall,
                      startup_s=startup, **metrics,
                      input_tokens_per_second=metrics["input_tokens"] / wall,
                      usd_per_query=wall / 3600 * H100_USD_PER_HOUR,
                      graph_replays=sum(m.replays for m in managers),
                      eager_forwards=sum(m.fallbacks for m in managers))
        print(json.dumps(record), flush=True)
        return record

    try:
        warmups = [execute("eager", "warmup-eager")]
        validate = True
        warmups.append(execute("graphs", "warmup-graphs"))
        validate = False
        if checked == 0:
            raise AssertionError("no graph forwards checked")
        records = [execute(mode, f"{mode}-{repeat}")
                   for repeat in range(3) for mode in ("eager", "graphs")]
        traces = {}
        decode_traces = {}
        for mode in ("eager", "graphs"):
            with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                execute(mode, f"trace-{mode}")
            path = out / f"{mode}.chrome.json.gz"
            profiler.export_chrome_trace(str(path))
            traces[mode] = _trace_metrics(path)
            with gzip.open(path, "rt") as stream:
                decode_traces[mode] = decode_trace_metrics(
                    json.load(stream)["traceEvents"])
            VOLUMES["/results"].commit()
        medians = {mode: {
            key: statistics.median(r[key] for r in records
                                   if r["configuration"] == mode)
            for key in ("query_s", "input_tokens_per_second", "usd_per_query")}
            for mode in ("eager", "graphs")}
        source = Path("/root/quail/backends/quail/executor/cuda_graphs.py")
        summary = dict(
            run=run, model="qwen3-4b-fp8", device="h100-sxm", gpus=1,
            query="CUAD-3", scale_factor=0.1, documents=len(tables["contracts"]),
            timing="query building through collection; startup excluded",
            warmups=warmups, records=records, medians=medians, traces=traces,
            decode_traces=decode_traces,
            answers_and_spans_equal=True, checked_forwards=checked,
            capture_s=sum(m.capture_s for m in managers),
            graph_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved())
        (out / "CUAD-3.json").write_text(json.dumps(summary, indent=2))
        VOLUMES["/results"].commit()
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        stages.forward_chunk = original
        session.close()


@app.local_entrypoint()
def main():
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = compare.spawn(run)
    print(f"function call id {call.object_id}", flush=True)
    print(f"results /results/ablations/extract-cuda-graphs-{run}/", flush=True)
