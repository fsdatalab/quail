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


def _run_one(session, table, column, instruction, cap, draft_policy,
             usd_per_hour) -> dict:
    """Run one map configuration and return its measurements."""
    from dataclasses import replace

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
    started = time.perf_counter()
    result = query.run(plan=plan)
    wall = time.perf_counter() - started
    report = result.report
    inference = report.get("backend_metrics", {}).get("inferences", [{}])
    answers = result.collect().column("answer").to_pylist()
    lengths = sorted(len(a or "") for a in answers)
    return {
        "wall_s": round(wall, 2),
        "engine_wall_s": report.get("wall_s"),
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
                  policies: str = "adaptive,fixed") -> str:
    """Run the four map configurations; return the result JSON's volume path."""
    import quail
    from bench.quailb import register_tables
    from quail.specs import H100_USD_PER_HOUR

    directory = _ensure_tables(scale_factor,
                               [table for _, table, _, _, _ in CONFIGURATIONS])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurements = {"model": model, "scale_factor": scale_factor, "runs": {}}
    with quail.Session(quail.EngineConfig(
            model=model, device="h100-sxm", backend="quail")) as session:
        register_tables(session, directory)
        for policy in policies.split(","):
            for name, table, column, instruction, cap in CONFIGURATIONS:
                run_name = f"{name}_{policy}"
                print(f"running {run_name}", flush=True)
                measurements["runs"][run_name] = _run_one(
                    session, table, column, instruction, cap, policy,
                    H100_USD_PER_HOUR)
                print(json.dumps(
                    {run_name: measurements["runs"][run_name]}, default=str),
                    flush=True)
    out = Path(RESULT_DIR) / f"map_quadrants_{model}_sf{scale_factor}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurements, indent=2, default=str))
    results_vol.commit()
    print(f"saved {out}")
    return str(out)


@app.local_entrypoint()
def launch(model: str = "qwen3-4b-fp8", sf: float = 0.1,
           policies: str = "adaptive,fixed"):
    call = map_quadrants.spawn(model=model, scale_factor=sf, policies=policies)
    print(f"function call id: {call.object_id}")
    print(call.get())
