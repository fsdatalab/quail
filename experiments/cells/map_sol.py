r"""Estimate the speed of light of a QUAIL-B map query from a saved run.

Maps have no reference answers, so each map answer, and each extraction
answer without a reference, is the one a saved Quail run wrote; every
other answer is the saved reference label. The estimate is
quail.speed_of_light_estimate: unlimited KV, ideal packing, each distinct
document prefix computed once, one decode pass per fed map token, and
the maps of one document reading it once per pass. CPU only. The record
goes to /results/ablations/sol/maps-<query>-sf<sf>-<model>.json.

    uv run modal run experiments/cells/map_sol.py::estimate \
        --query CUAD-22 --scale-factor 1.0 \
        --run-dir /results/ablations/vllm_preemption/<run id>/quail \
        2>&1 | tee "$run_log"
"""

import json
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

SOL_DIR = "/results/ablations/sol"


def saved_answers(run_dir: Path, spec, tables) -> dict:
    """Return each map and extraction's saved answers by row index.

    Args:
        run_dir: The backend's run directory, holding one folder per query.
        spec: The QUAIL-B query.
        tables: The corpus tables, by name.

    Returns:
        Quail's answer column name -> row index -> answer text.
    """
    import pyarrow.parquet as pq

    from bench.quailb import _extract_name

    info = spec.info
    record = json.loads((run_dir / "run.json").read_text())
    (query,) = [item for item in record["queries"] if item["id"] == spec.id]
    paths = {entry["key"]: entry["path"]
             for kind in ("maps", "extracts")
             for entry in query["files"].get(kind) or ()}
    names = {operator.id: operator.output for operator in info.maps}
    names.update({operator.id: _extract_name(info, operator)
                  for operator in info.extracts})
    tables_by_alias = {relation.alias: relation.table
                       for relation in info.relations}
    answers = {}
    for operator in (*info.maps, *info.extracts):
        table = pq.read_table(run_dir / spec.id / paths[operator.id])
        ids = tables[tables_by_alias[operator.relation]].column("id").to_pylist()
        row_of = {str(row_id): row for row, row_id in enumerate(ids)}
        answers[names[operator.id]] = {
            row_of[str(row_id)]: text for row_id, text in zip(
                table.column(operator.relation).to_pylist(),
                table.column("answer").to_pylist())}
    return answers


@app.function(image=gpu_image(), cpu=8, memory=65536, timeout=3600,
              volumes=VOLUMES)
def estimate_sol(query_id: str, scale_factor: float, collection_id: str,
                 run_dir: str, model: str) -> str:
    """Estimate one query's speed of light on one H100 and save it.

    Returns:
        The saved record as JSON.
    """
    import quail
    from bench.quailb import answer_oracle, build_query
    from quail_b.benchmark import load_benchmark

    suite = load_benchmark([query_id], scale_factor=scale_factor,
                           data_dir=Path(DATA_DIR) / f"sf{scale_factor}",
                           collection_id=collection_id or None)
    spec = suite.queries[0]
    saved = saved_answers(Path(run_dir), spec, suite.tables)
    config = quail.EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config) as session:
        for name, table in suite.tables.items():
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col="id"))
        query = build_query(session, spec)
        operators = query.logical.operators()
        saved_by_prompt = {node.call.prompt: saved[node.name]
                           for node in (*operators.maps, *operators.extracts)
                           if node.name in saved}
        reference = answer_oracle(suite.ground_truth, suite.tables,
                                  session.model.turn)
        from_run = set()

        def answer(prompt, assignment):
            try:
                return reference(prompt, assignment)
            except KeyError:
                # no reference: a map, or an extraction without labels
                from_run.add(prompt.template)
                (row,) = assignment.values()
                return saved_by_prompt[prompt][row]

        sol = quail.speed_of_light_estimate(query, answer, label=answer)
    latency = sol.latency
    record = {
        "query": query_id, "scale_factor": scale_factor, "model": model,
        "device": "h100-sxm", "chunk_tokens": sol.chunk_tokens,
        "answers_from_run": run_dir,
        "num_prompts_answered_from_run": len(from_run),
        "reference_model": suite.ground_truth.reference_model,
        "assumptions": ("unlimited KV; ideal packing; each distinct document "
                        "prefix computed once; one decode pass per fed map "
                        "token; the maps of one document read it once per "
                        "pass"),
        "seconds": sol.seconds, "usd_per_query": sol.usd_per_query,
        "passes": latency.passes, "bound_by": latency.bound_by,
        "components": {component.name: component.seconds
                       for component in latency.components},
        "fresh_tokens": sol.work.tokens, "kv_read_tokens": sol.work.kv_read,
        "documents_by_alias": sol.documents_by_alias,
        "stages": [{key: stage[key] for key in
                    ("operator", "alias", "name", "answer_rows") if key in stage}
                   for stage in sol.filter_stages],
    }
    path = Path(SOL_DIR) / f"maps-{query_id}-sf{scale_factor:g}-{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str))
    results_vol.commit()
    return json.dumps(record, indent=1, default=str)


@app.local_entrypoint()
def estimate(query: str, run_dir: str, scale_factor: float = 0.1,
             model: str = "qwen3-4b-fp8"):
    collection_id = ensure_data.remote(scale_factor, [query], "")
    call = estimate_sol.spawn(query, scale_factor, collection_id, run_dir, model)
    print(f"function call id: {call.object_id} ({query})", flush=True)
    print(call.get(), flush=True)
