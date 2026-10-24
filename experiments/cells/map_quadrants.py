"""Run AI.MAP over the quail-b corpora: short and long documents and answers.

Four configurations on one H100 with Quail alone, no baselines: IMDB
reviews (short documents, high length variance) and CUAD contracts
(long documents), each with a short map (one sentence, 64-token cap)
and a long map (a detailed summary, 256-token cap). Each configuration
runs twice, with the adaptive draft policy and with the fixed document
copy, so the run isolates what the policy buys.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-map-quadrants.log"
    uv run modal run --detach \
      experiments/cells/map_quadrants.py::launch \
      2>&1 | tee "$run_log"

The result JSON lands under /results/ablations/ on the quail-results
volume; the function prints its path.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, image, results_vol

RESULT_DIR = "/results/ablations"

SHORT_MAP = ("Summarize the negative arguments the writer makes, "
             "in one sentence.")
LONG_MAP = ("Write a detailed summary of the document: the parties or "
            "people involved, the main points they make, and anything "
            "unresolved.")

# (name, table, column, instruction, max_answer_tokens)
CONFIGURATIONS = (
    ("imdb_short_map", "reviews", "body", SHORT_MAP, 64),
    ("imdb_long_map", "reviews", "body", LONG_MAP, 256),
    ("cuad_short_map", "contracts", "text", SHORT_MAP, 64),
    ("cuad_long_map", "contracts", "text", LONG_MAP, 256),
)


def _ensure_tables(scale_factor: float, names) -> Path:
    """Write the corpora parquets to the volume, as ensure_data does."""
    import pyarrow.parquet as pq

    import quail_b as benchmark
    from quail_b.data import CORPUS_COLUMNS

    directory = Path(DATA_DIR) / f"sf{scale_factor}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(names)):
        path = directory / f"{name}.parquet"
        if (path.exists() and set(CORPUS_COLUMNS[name])
                <= set(pq.read_schema(path).names)):
            continue
        pq.write_table(benchmark.load_table(name, scale_factor=scale_factor),
                       path)
    results_vol.commit()
    return directory


def _chunk_log(records: list):
    """Return a stage executor pass method that records each chunk.

    Each record holds the chunk's tokens, its groups, how many packed
    a fresh prefix, and the forward pass's GPU milliseconds.
    """
    from quail.backends.quail.executor.stage_executor import StageExecutor

    original = StageExecutor._run_one

    def run_one(self, part):
        torch = self.torch
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record()
        original(self, part)
        after.record()
        records.append((sum(self._entry_rows(*entry) for entry in part), len(part),
                        sum(1 for entry in part if entry[4]), before, after))

    return run_one


def _print_chunk_log(records: list) -> None:
    """Print the chunks by token count: how many, their groups, prefills, GPU ms."""
    import numpy as np

    if not records:
        return
    records[-1][4].synchronize()
    tokens = np.array([r[0] for r in records])
    groups = np.array([r[1] for r in records])
    fresh = np.array([r[2] for r in records])
    gpu_ms = np.array([r[3].elapsed_time(r[4]) for r in records])
    print(f"chunks {len(records)}: tokens {tokens.sum()}, "
          f"gpu {gpu_ms.sum() / 1000:.2f} s, "
          f"{gpu_ms.sum() * 1000 / tokens.sum():.1f} us per token")
    edges = [0, 64, 128, 256, 512, 1024, 2048, 4096, 1 << 20]
    for low, high in zip(edges, edges[1:]):
        chosen = (tokens > low) & (tokens <= high)
        if not chosen.any():
            continue
        print(f"  tokens {low + 1}-{high}: {int(chosen.sum())} chunks, mean "
              f"{tokens[chosen].mean():.0f} tokens, {groups[chosen].mean():.1f} "
              f"groups, {fresh[chosen].mean():.1f} fresh, "
              f"{gpu_ms[chosen].mean():.2f} ms, "
              f"{gpu_ms[chosen].sum() * 1000 / tokens[chosen].sum():.1f} us/token")


def _run_one(session, table, column, instruction, cap, draft_policy,
             usd_per_hour, profile=False, profile_gpu=False) -> dict:
    """Run one map configuration and return its measurements.

    With profile, the run is traced with cProfile and the functions
    with the most own time are printed. With profile_gpu, the run is
    traced with the torch profiler, the kernels with the most GPU time
    are printed, and the chunks are summarized by size.
    """
    import cProfile
    import pstats
    from dataclasses import replace
    from unittest import mock

    from quail.physical import AiMap

    query = (session.docs(table).alias("d")
             .ai_map(f"d.{column}", instruction, name="answer")
             .select("d.id", "answer"))
    plan = query.plan()
    nodes = tuple(replace(
        node, spec=replace(node.spec, max_answer_tokens=cap,
                           draft_policy=draft_policy))
        if isinstance(node, AiMap) else node for node in plan.nodes)
    plan = replace(plan, nodes=nodes)
    profiler = cProfile.Profile() if profile else None
    records = []
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    if profile_gpu:
        import torch
        from torch.profiler import ProfilerActivity

        patched = mock.patch(
            "quail.backends.quail.executor.stage_executor.StageExecutor._run_one",
            _chunk_log(records))
        with patched, torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            result = query.run(plan=plan)
    else:
        result = query.run(plan=plan)
    if profiler is not None:
        profiler.disable()
    wall = time.perf_counter() - started
    if profiler is not None:
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(60)
        stats.sort_stats("cumtime").print_stats("quail/", 60)
    if profile_gpu:
        _print_chunk_log(records)
        print(trace.key_averages().table(sort_by="self_device_time_total",
                                         row_limit=45, max_name_column_width=90),
              flush=True)
    report = result.report
    inference = report.get("backend_metrics", {}).get("inferences", [{}])
    answers = result.collect().column("answer").to_pylist()
    lengths = sorted(len(a or "") for a in answers)
    return {
        "wall_s": round(wall, 2),
        "engine_wall_s": report.get("wall_s"),
        "gpu_s": report.get("gpu_s"),
        "chunks": report.get("chunks"),
        "fresh_tokens": report.get("fresh_tokens"),
        "cached_tokens": report.get("cached_tokens"),
        "usd_per_query": (report.get("wall_s") or wall) / 3600 * usd_per_hour,
        "num_documents": len(answers),
        "num_empty_answers": sum(not a for a in answers),
        "answer_chars": {"min": lengths[0], "median": lengths[len(lengths) // 2],
                         "max": lengths[-1]},
        "inference": inference[0] if inference else {},
        "sample_answers": answers[:3],
    }


@app.function(image=image, gpu="H100!", memory=98304, timeout=14400,
              volumes=VOLUMES)
def map_quadrants(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                  policies: str = "adaptive,fixed", only: str = "",
                  gpu_timing: bool = True, profile: bool = False,
                  profile_gpu: bool = False, graph_rows: str = "",
                  graph_query_tokens: int = 0, graph_context_tokens: int = 0) -> str:
    """Run the map configurations; return the result JSON's volume path.

    Args:
        model: The model to run.
        scale_factor: The corpus scale factor.
        policies: Comma-separated draft policies to run each
            configuration under.
        only: Comma-separated configuration names to run; empty runs
            all four.
        gpu_timing: Whether to time the forward passes with CUDA
            events and report their sum as gpu_s beside the wall.
        profile: Whether to print a cProfile of each run's host.
        profile_gpu: Whether to print each run's GPU kernels by time
            and its chunks by size.
        graph_rows: Comma-separated row buckets for the decode graphs;
            empty keeps the default.
        graph_query_tokens: The longest query a captured chunk may
            hold; 0 keeps the default.
        graph_context_tokens: The most resident tokens a captured
            sequence reads; 0 keeps the default.
    """
    import os

    import quail
    from bench.quailb import register_tables
    from quail.specs import H100_USD_PER_HOUR

    if graph_rows:
        os.environ["QUAIL_DECODE_GRAPH_ROWS"] = graph_rows
    if graph_query_tokens:
        os.environ["QUAIL_DECODE_GRAPH_QUERY_TOKENS"] = str(graph_query_tokens)
    if graph_context_tokens:
        os.environ["QUAIL_DECODE_GRAPH_CONTEXT_TOKENS"] = str(graph_context_tokens)
    directory = _ensure_tables(scale_factor,
                               [table for _, table, _, _, _ in CONFIGURATIONS])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurements = {"model": model, "scale_factor": scale_factor, "runs": {}}
    with quail.Session(quail.EngineConfig(
            model=model, device="h100-sxm", backend="quail",
            gpu_timing=gpu_timing)) as session:
        register_tables(session, directory)
        chosen = {name.strip() for name in only.split(",") if name.strip()}
        for policy in policies.split(","):
            for name, table, column, instruction, cap in CONFIGURATIONS:
                if chosen and name not in chosen:
                    continue
                run_name = f"{name}_{policy}"
                print(f"running {run_name}", flush=True)
                measurements["runs"][run_name] = _run_one(
                    session, table, column, instruction, cap, policy,
                    H100_USD_PER_HOUR, profile=profile, profile_gpu=profile_gpu)
                print(json.dumps(
                    {run_name: measurements["runs"][run_name]}, default=str),
                    flush=True)
    measurements["graph_rows"] = graph_rows
    measurements["graph_query_tokens"] = graph_query_tokens
    measurements["graph_context_tokens"] = graph_context_tokens
    out = Path(RESULT_DIR) / f"map_quadrants_{model}_sf{scale_factor}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurements, indent=2, default=str))
    results_vol.commit()
    print(f"saved {out}")
    return str(out)


@app.local_entrypoint()
def launch(model: str = "qwen3-4b-fp8", sf: float = 0.1,
           policies: str = "adaptive,fixed", only: str = "",
           profile: bool = False, profile_gpu: bool = False,
           graph_rows: str = "", graph_query_tokens: int = 0,
           graph_context_tokens: int = 0):
    call = map_quadrants.spawn(model=model, scale_factor=sf, policies=policies,
                               only=only, profile=profile, profile_gpu=profile_gpu,
                               graph_rows=graph_rows,
                               graph_query_tokens=graph_query_tokens,
                               graph_context_tokens=graph_context_tokens)
    print(f"function call id: {call.object_id}")
    print(call.get())
