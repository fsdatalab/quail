"""Compare extraction packing on one H100 using the saved CUAD corpus.

    uv run modal run --detach -m experiments.compare_extract_packing \
        2>&1 | tee /tmp/compare_extract_packing.log

Results and traces are saved on quail-results. Fetch the returned summary
with modal.FunctionCall.from_id using the printed function call ID.
"""

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
    "/results": modal.Volume.from_name("quail-results"),
}


@app.function(image=gpu_image(), gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def compare(run: str, baseline: str, source: str, revision: str):
    """Measure alternating warmed queries and save one trace per configuration."""
    import pyarrow.parquet as pq
    import torch

    import quail
    from experiments.compare_extract_tokenization import _load_source, _trace_metrics
    from quail.backends.quail.executor import loop, pack
    from quail.bench import quailb
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b import get_query
    from quail_b.minimum import token_metrics

    out = Path(f"/results/ablations/extract-packing-{run}")
    out.mkdir(parents=True)
    (out / "baseline_pack.py").write_text(source)
    current_source = Path(pack.__file__).read_bytes()
    (out / "current_pack.py").write_bytes(current_source)
    tables = {name: pq.read_table(
        f"/results/quailb_data/sf1.0/{name}.parquet")
        for name in ("contracts", "clause_types")}
    spec = get_query("CUAD-3")
    session = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"))
    for name, table in tables.items():
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    original_next = pack.JoinAdmission.next_chunk
    methods = {
        "before": _load_source("baseline_pack", source).JoinAdmission.next_chunk,
        "after": original_next,
    }
    original_forward = loop._forward
    reference = None
    stores = {}
    chunks = []
    budgets = []
    managers = []

    def forward(pipeline, arena, chunk):
        chunks.append(int(chunk.tokens))
        graphs = pipeline.decode_graphs
        if graphs not in managers:
            managers.append(graphs)
        return original_forward(pipeline, arena, chunk)

    loop._forward = forward

    def execute(mode, name):
        nonlocal reference

        def next_chunk(admission, free_pages):
            groups = methods[mode](admission, free_pages)
            if groups:
                budgets.append(admission.chunk_budget)
            return groups

        pack.JoinAdmission.next_chunk = next_chunk
        chunks.clear()
        budgets.clear()
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
            reference = rows
        pq.write_table(rows, out / f"{name}.parquet")
        output = quailb.run_output(result, spec.info, tables)
        output.prompt_pieces = quailb.prompt_pieces(
            query, spec.info, quailb.join_anchors(result))
        metrics = token_metrics(spec, output, tables, stores)
        record = dict(
            configuration=mode, name=name, query_s=wall, startup_s=startup,
            **metrics, input_tokens_per_second=metrics["input_tokens"] / wall,
            usd_per_query=wall / 3600 * H100_USD_PER_HOUR,
            answers_and_spans_equal=rows.equals(reference),
            forwards=len(chunks), forward_tokens=list(chunks),
            admission_budgets=list(budgets),
            graph_replays=sum(m.replays for m in managers),
            eager_forwards=sum(m.fallbacks for m in managers))
        (out / f"{name}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("forward_tokens", "admission_budgets")}),
              flush=True)
        VOLUMES["/results"].commit()
        return record

    try:
        warmups = [execute(mode, f"warmup-{mode}") for mode in methods]
        records = []
        for repeat in range(3):
            order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
            records.extend(execute(mode, f"{mode}-{repeat}") for mode in order)
        traces = {}
        for mode in methods:
            with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                execute(mode, f"trace-{mode}")
            path = out / f"{mode}.chrome.json.gz"
            profiler.export_chrome_trace(str(path))
            traces[mode] = _trace_metrics(path)
            print(json.dumps({"configuration": mode, "trace": traces[mode]}),
                  flush=True)
            VOLUMES["/results"].commit()
        medians = {mode: {
            key: statistics.median(r[key] for r in records
                                   if r["configuration"] == mode)
            for key in ("query_s", "input_tokens_per_second", "usd_per_query",
                        "regret_tokens", "forwards")}
            for mode in methods}
        summary = dict(
            run=run, query="CUAD-3", model="qwen3-4b-fp8", device="h100-sxm",
            gpus=1, scale_factor=1.0, documents=len(tables["contracts"]),
            baseline_revision=baseline, current_revision=revision,
            baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            current_source_sha256=hashlib.sha256(current_source).hexdigest(),
            timing="query building through collection; startup excluded",
            warmups=warmups, records=records, medians=medians, traces=traces,
            answers_and_spans_equal=all(r["answers_and_spans_equal"]
                                        for r in records),
            result_path=str(out / "CUAD-3.json"))
        (out / "CUAD-3.json").write_text(json.dumps(summary, indent=2))
        VOLUMES["/results"].commit()
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("records", "warmups")}), flush=True)
        return summary
    finally:
        pack.JoinAdmission.next_chunk = original_next
        loop._forward = original_forward
        session.close()


@app.local_entrypoint()
def main():
    """Spawn the comparison using the source before the packing change."""
    baseline = "2b0b67edfb97c8d239237cb406e236add32aa7e7"
    source = subprocess.check_output([
        "git", "show", f"{baseline}:quail/backends/quail/executor/pack.py"],
        text=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = compare.spawn(run, baseline, source, revision)
    print(f"function call id {call.object_id}", flush=True)
    print(f"results /results/ablations/extract-packing-{run}/", flush=True)
