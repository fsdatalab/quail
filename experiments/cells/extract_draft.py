r"""Time AI.EXTRACT queries with each fixed draft length and the adaptive policy.

Each query runs on its own H100. An untimed adaptive run of the query
loads the model and captures CUDA graphs. Then the cell times every setting, `repeats`
passes over all of them: "adaptive" is the plan as Quail makes it, and
a number fixes the draft length of every extraction in the plan to it
(draft_policy "fixed"). Nothing survives in the arena between queries,
so each timed query computes its documents afresh. For example:

    uv run modal run --detach experiments/cells/extract_draft.py::sweep \
        --queries CUAD-1,CUAD-3,IMDB-16,IMDB-17,WRENCH-8 \
        --scale-factor 1.0 2>&1 | tee "$run_log"

Per run the cell records wall seconds, the draft settings the plan's
extractions carried, and a hash of every answer. Results go to
/results/ablations/extract_draft/<run id>/<query>.json.
"""

import hashlib
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

OUTPUT_DIR = "/results/ablations/extract_draft"
SETTINGS = ("adaptive", "0", "2", "4", "8", "16", "32", "63")
image = gpu_image()

# the draft setting the patched planner gives each extraction
_setting = {"value": "adaptive"}


def fix_draft_lengths():
    """Make the extraction planner apply the cell's draft setting."""
    from quail.planner.operators.extract import ExtractPlanner

    planned = ExtractPlanner.spec

    def spec(self, *args, **kwargs):
        made = planned(self, *args, **kwargs)
        if _setting["value"] == "adaptive":
            return made
        return replace(made, draft_policy="fixed",
                       draft_tokens=int(_setting["value"]))

    ExtractPlanner.spec = spec


def extract_settings(query) -> list[list]:
    """Return the (policy, draft tokens) of each extraction in the plan."""
    from quail.physical.nodes.extract import AiExtract

    return [[node.spec.draft_policy, node.spec.draft_tokens]
            for node in query.plan().nodes
            if isinstance(node, AiExtract) and node.spec is not None]


@app.function(image=image, gpu="H100!", memory=98304, timeout=6 * 3600,
              volumes=VOLUMES)
def run_query(query_id: str, scale_factor: float, settings: list[str],
              repeats: int, run_dir: str) -> str:
    """Time one QUAIL-B query under each draft setting and save the record.

    Returns:
        The record as JSON.
    """
    import quail
    import quail_b as benchmark
    from bench.quailb import build_query, data_directory, register_tables

    fix_draft_lengths()
    spec = benchmark.get_query(query_id)
    data_dir = data_directory(DATA_DIR, scale_factor, [query_id])
    config = quail.EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                                device="h100-sxm")
    record = {"query": query_id, "scale_factor": scale_factor,
              "settings": settings, "runs": []}
    with quail.Session(config) as session:
        register_tables(session, data_dir)
        # an untimed adaptive run loads the model and captures CUDA graphs;
        # nothing it writes survives in the arena for the timed runs
        build_query(session, spec).run().collect()
        for repeat in range(repeats):
            for setting in settings:
                _setting["value"] = setting
                query = build_query(session, spec)
                planned = extract_settings(query)
                result = query.run()
                table = result.collect()
                digest = hashlib.sha256()
                for name in table.column_names:
                    for value in table.column(name).to_pylist():
                        digest.update(repr(value).encode() + b"\0")
                run = {"repeat": repeat, "setting": setting,
                       "wall_s": round(result.report["wall_s"], 3),
                       "planned": planned, "rows": table.num_rows,
                       "answers_sha256": digest.hexdigest()}
                record["runs"].append(run)
                print(query_id, run, flush=True)
    path = Path(run_dir) / f"{query_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    results_vol.commit()
    return json.dumps(record, indent=1)


@app.local_entrypoint()
def sweep(queries: str = "CUAD-1,CUAD-3,IMDB-16,IMDB-17,WRENCH-8",
          scale_factor: float = 1.0, settings: str = ",".join(SETTINGS),
          repeats: int = 2):
    query_ids = [name for name in queries.split(",") if name]
    ensure_data.remote(scale_factor, query_ids, "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    chosen = [setting for setting in settings.split(",") if setting]
    calls = [run_query.spawn(query, scale_factor, chosen, repeats, run_dir)
             for query in query_ids]
    for query, call in zip(query_ids, calls):
        print(f"function call id: {call.object_id} ({query})", flush=True)
    for call in calls:
        print(call.get(), flush=True)
