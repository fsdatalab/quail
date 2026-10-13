"""Compare extraction CPU work and document scheduling on one H100.

The baseline loads extraction and scheduling sources from a snapshot or revision.
All configurations use the current stack's planner, prompt, and model.
Warm each configuration, then alternate three query runs per configuration.
Save answer tables, spans, timing, token counts, and CPU profiles to the
quail-results volume under /results/ablations/extract-tokenization-<run>/.

    uv run modal run --detach -m experiments.compare_extract_tokenization \
        2>&1 | tee /tmp/compare_extract_tokenization.log
"""

import cProfile
import gzip
import hashlib
import io
import json
import os
import pstats
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import modal

from quail.bench.images import gpu_image

app = modal.App("quail-milestone1")
image = gpu_image()
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name("quail-hf-cache"),
    "/root/.cache/kernels": modal.Volume.from_name("quail-kernel-cache"),
    "/results": modal.Volume.from_name("quail-results"),
}
MODEL = "qwen3-4b-fp8"
TABLES = "/root/.cache/huggingface/quail-b-tables"


def _load_source(name, source):
    """Load the baseline source into an isolated module."""
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def _trace_metrics(path):
    """Measure GPU activity within the complete query execution interval."""
    from experiments.profile_extract import _merge

    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    query = next(event for event in events
                 if event.get("name") == "query_execution"
                 and event.get("ph") == "X")
    start, end = query["ts"], query["ts"] + query["dur"]
    intervals = _merge([
        (max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
        for event in events if event.get("ph") == "X"
        and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")])
    active = sum(b - a for a, b in intervals)
    return {"query_s": query["dur"] / 1e6, "gpu_active_s": active / 1e6,
            "gpu_idle_fraction": 1 - active / query["dur"],
            "interval": "whole query execution, including host preparation"}


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def compare(run: str, query_id: str, baseline: str, sources: dict,
            reuse_only: bool = False) -> dict:
    """Measure the configurations and check their answers and spans agree."""
    import pyarrow.parquet as pq
    import torch

    import quail
    from quail.backends.quail.executor import stages as scheduler
    from quail.backends.quail.executor.operators import extract as operator
    from quail.bench import quailb
    from quail.planner.plan import EngineConfig
    from quail.specs import H100_USD_PER_HOUR
    from quail_b import get_query
    from quail_b.data import build_sets
    from quail_b.minimum import token_metrics

    out = f"/results/ablations/extract-tokenization-{run}"
    os.makedirs(out, exist_ok=True)
    prefix = "quail.backends.quail.executor.extract"
    _load_source("baseline_candidates", sources["extract/candidates.py"])
    _load_source("baseline_decode", sources["extract/decode.py"].replace(
        f"{prefix}.candidates", "baseline_candidates"))
    extraction = _load_source(
        "baseline_extract_stages", sources["extract/stages.py"].replace(
            f"{prefix}.candidates", "baseline_candidates").replace(
                f"{prefix}.decode", "baseline_decode"))
    scheduling = _load_source("baseline_stages", sources["stages.py"])
    original = operator.ExtractionStages, scheduler._StageExecutor
    original_loader = operator.load_vocabulary
    vocabulary_source = sources["extract/vocabulary.py"]
    if "tokens.py" in sources:
        _load_source("baseline_tokens", sources["tokens.py"])
        vocabulary_source = vocabulary_source.replace(
            "quail.execution.tokens", "baseline_tokens")
    baseline_vocabulary = _load_source("baseline_vocabulary", vocabulary_source)
    vocabularies = {}
    methods = {
        "baseline": (extraction.ExtractionStages, scheduling._StageExecutor),
        "cpu": (original[0], scheduling._StageExecutor),
        "overlap": original,
    }
    if reuse_only:
        methods = {"baseline": methods["baseline"], "reuse": original}
    target = next(reversed(methods))
    for name, source in sources.items():
        path = Path(out) / "baseline" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    data_dir = build_sets(TABLES, 0.1)
    tables = {name: pq.read_table(data_dir / f"{name}.parquet")
              for name in ("contracts", "clause_types")}
    VOLUMES["/root/.cache/huggingface"].commit()
    spec = get_query(query_id)
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm"))
    for name, table in tables.items():
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    reference = None
    reference_fresh = None
    stores = {}
    records = []

    def execute(configuration, name):
        nonlocal reference, reference_fresh
        operator.ExtractionStages, scheduler._StageExecutor = methods[configuration]

        def load_vocabulary(state):
            key = configuration, id(state.loaded_model)
            if key not in vocabularies:
                state.loaded_model.vocabulary = None
                loader = (baseline_vocabulary.load_vocabulary
                          if configuration == "baseline" else original_loader)
                vocabularies[key] = loader(state)
            return vocabularies[key]

        operator.load_vocabulary = load_vocabulary
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
            reference_fresh = result.report["fresh_tokens"]
        if not rows.equals(reference):
            raise AssertionError(f"{name}: answer text or spans changed")
        if result.report["fresh_tokens"] != reference_fresh:
            raise AssertionError(f"{name}: model token count changed")
        pq.write_table(rows, f"{out}/{name}.parquet")
        output = quailb.run_output(result, spec.info, tables)
        output.prompt_pieces = quailb.prompt_pieces(
            query, spec.info, quailb.join_anchors(result))
        metrics = token_metrics(spec, output, tables, stores)
        record = {
            "configuration": configuration, "name": name, "query_s": wall,
            "startup_s": startup,
            **metrics,
            "input_tokens_per_second": metrics["input_tokens"] / wall,
            "usd_per_query": wall / 3600 * H100_USD_PER_HOUR,
            "answers_and_spans_equal": True,
        }
        print(json.dumps(record), flush=True)
        return record

    warmups = [execute(configuration, f"warmup-{configuration}")
               for configuration in methods]
    for repeat in range(3):
        for configuration in methods:
            records.append(execute(configuration, f"{configuration}-{repeat}"))
    profiles = {}
    medians = {}
    for configuration in methods:
        own = [record for record in records
               if record["configuration"] == configuration]
        medians[configuration] = {
            key: statistics.median(record[key] for record in own)
            for key in ("query_s", "input_tokens_per_second", "usd_per_query")}
    summary = {
        "run": run, "query": query_id, "description": spec.description,
        "baseline": baseline,
        "baseline_sha256": {name: hashlib.sha256(source.encode()).hexdigest()
                            for name, source in sources.items()},
        "model": MODEL, "device": "h100-sxm", "gpus": 1, "scale_factor": 0.1,
        "documents": len(tables["contracts"]),
        "timing": "query building through collection; warm; startup excluded",
        "warmups": warmups, "records": records, "medians": medians,
        "cpu_profiles": profiles, "answers_and_spans_equal": True,
        "speedup": medians["baseline"]["query_s"] / medians[target]["query_s"],
    }
    with open(f"{out}/{query_id}.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    VOLUMES["/results"].commit()
    for configuration in methods:
        profiler = cProfile.Profile()
        profiler.enable()
        execute(configuration, f"profile-{configuration}")
        profiler.disable()
        text = io.StringIO()
        stats = pstats.Stats(profiler, stream=text)
        stats.sort_stats("tottime").print_stats(50)
        stats.sort_stats("cumulative").print_stats(50)
        with open(f"{out}/{configuration}.cprofile.txt", "w") as stream:
            stream.write(text.getvalue())
        profiles[configuration] = [
            {"file": file, "function": function, "calls": calls,
             "own_s": own, "cumulative_s": cumulative}
            for (file, _, function), (_, calls, own, cumulative, _)
            in stats.stats.items()
            if function in ("extend", "extend_streams", "ensure_tokens", "squeezed",
                            "encode", "encode_batch", "first_tokens", "lp",
                            "window", "choose", "_report", "_run_one",
                            "extract_stages", "_prepare", "positions_of")
            or "encode_batch" in function]
    operator.ExtractionStages, scheduler._StageExecutor = original
    traces = {}
    for configuration in methods:
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            execute(configuration, f"trace-{configuration}")
        path = f"{out}/{configuration}.chrome.json.gz"
        profiler.export_chrome_trace(path)
        traces[configuration] = _trace_metrics(path)
        print(json.dumps({"trace": configuration, **traces[configuration]}),
              flush=True)
        VOLUMES["/results"].commit()
    summary["gpu_traces"] = traces
    operator.ExtractionStages, scheduler._StageExecutor = original
    operator.load_vocabulary = original_loader
    session.close()
    with open(f"{out}/{query_id}.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    VOLUMES["/results"].commit()
    print(json.dumps({"medians": medians, "speedup": summary["speedup"],
                      "result_path": f"{out}/{query_id}.json"}), flush=True)
    return summary


@app.local_entrypoint()
def main(query: str = "CUAD-3", baseline: str = "2ced9658",
         baseline_dir: str = "", reuse_only: bool = False):
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    names = ("extract/candidates.py", "extract/decode.py", "extract/stages.py",
             "stages.py")
    sources = {name: ((Path(baseline_dir) / name).read_text() if baseline_dir
                      else subprocess.check_output(
                          ["git", "show",
                           f"{baseline}:quail/backends/quail/executor/{name}"],
                          text=True)) for name in names}
    vocabulary_path = Path(baseline_dir) / "extract/vocabulary.py"
    sources["extract/vocabulary.py"] = (
        vocabulary_path.read_text() if baseline_dir and vocabulary_path.exists()
        else subprocess.check_output(
            ["git", "show", f"{baseline}:quail/backends/quail/executor/"
             "extract/vocabulary.py"], text=True))
    tokens_path = Path(baseline_dir) / "tokens.py"
    if baseline_dir and tokens_path.exists():
        sources["tokens.py"] = tokens_path.read_text()
    call = compare.spawn(run, query, baseline_dir or baseline, sources, reuse_only)
    print(f"function call id {call.object_id}", flush=True)
    print(f"results under /results/ablations/extract-tokenization-{run}/", flush=True)
