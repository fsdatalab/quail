r"""Compare answer admission modes on AI.MAP queries over reviews and contracts.

Each mode and query runs on its own H100, so no KV or prefix cache
carries from one query to the next. QUAIL_ANSWER_ADMISSION picks a Quail
mode before the Session starts; the modes stock_vllm, shuffled_vllm, and
pipelined_vllm run that vLLM baseline backend instead, counting its
evictions and recomputed tokens from its scheduler step trace. A warm-up
query on copies of 64 reviews, whose first line differs, loads the model
and captures CUDA graphs. Then each query runs `repeats` times: a cell
query on the first `num_documents` reviews (all of them for 0), and a
QUAIL-B query on its whole tables at the scale factor. For example, the
four map queries of the comparison:

    uv run modal run --detach experiments/cells/answer_headroom.py::compare \
        --modes stock_vllm,shuffled_vllm,headroom=0,headroom \
        --queries CUAD-23,CUAD-22,IMDB-26,IMDB-25 --scale-factor 0.1 \
        --num-documents 0 --repeats 1 2>&1 | tee "$run_log"

Per run the cell records wall seconds, answer evictions,
recomputed tokens, the headroom answers were given, the passes and the
tokens, attention pairs, and KV tokens read in them with their SoL
seconds, answer-length percentiles, and a hash of every answer, so
modes can be checked for the same output. Results go to
/results/ablations/answer_headroom/<run id>/<mode>-<query>.json.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

# long-answer map drafts over IMDB reviews
NAMES = ("List every film, actor, director, and character the review names, "
         "with what the writer says about each.")
PLOT = ("Retell the plot of the film as the review describes it, in order, "
        "with every character and event the review mentions.")
REWRITE = ("Rewrite the review as a balanced one that gives both the strengths "
           "and the weaknesses of the film, keeping every point the writer makes.")
PARTIES = "List every party to the agreement with its role."
LIABILITY = "Quote every clause that limits liability or caps damages."
SECTIONS = "Summarize each section of the agreement in one sentence."
LIMIT = 1024
QUERIES = {
    "one long map": (f"SELECT r.id, AI.MAP(r.body, '{PLOT}', {LIMIT}) "
                     "AS plot FROM reviews r"),
    "three long maps": (f"SELECT r.id, AI.MAP(r.body, '{PLOT}', {LIMIT}) AS plot, "
                        f"AI.MAP(r.body, '{REWRITE}', {LIMIT}) AS balanced, "
                        f"AI.MAP(r.body, '{NAMES}', {LIMIT}) AS names "
                        "FROM reviews r"),
    # contract maps whose lengths differ by map and grow with the contract
    "three contract maps": (
        f"SELECT c.id, AI.MAP(c.text, '{PARTIES}', {LIMIT}) AS parties, "
        f"AI.MAP(c.text, '{LIABILITY}', {LIMIT}) AS liability, "
        f"AI.MAP(c.text, '{SECTIONS}', {LIMIT}) AS sections "
        "FROM contracts c"),
}
# modes that run a vLLM baseline backend instead of a Quail admission mode
VLLM_MODES = ("stock_vllm", "shuffled_vllm", "pipelined_vllm")

OUTPUT_DIR = "/results/ablations/answer_headroom"
WARMUP_DOCUMENTS = 64
image = gpu_image()


def register_tables(session, reviews, scale_factor: float, num_documents: int,
                    num_contracts: int = 0) -> dict:
    """Register the first num_documents reviews and every other table whole.

    num_documents of 0 keeps every review; num_contracts above zero keeps
    only the first contracts.

    Returns:
        Each registered table's row ids, in row order.
    """
    import pyarrow.parquet as pq

    import quail

    tables = {"reviews": reviews.slice(0, num_documents or None)}
    for path in sorted((Path(DATA_DIR) / f"sf{scale_factor}").glob("*.parquet")):
        if path.stem != "reviews":
            tables[path.stem] = pq.read_table(path)
    if num_contracts and "contracts" in tables:
        tables["contracts"] = tables["contracts"].slice(0, num_contracts)
    for name, table in tables.items():
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    return {name: table.column("id").to_pylist() for name, table in tables.items()}


def slug(query: str) -> str:
    """Return a query name fit for a file name."""
    return "".join(c if c.isalnum() else "_" for c in query)


def answers_path(run_dir: str, mode: str, query: str) -> Path:
    """Return where a mode's last run of a query keeps its answers."""
    return Path(run_dir) / f"{mode}-answers" / f"{slug(query)}.json"


def report_path(run_dir: str, mode: str, query_ids) -> Path:
    """Return the mode's measurements file: one per query run alone."""
    name = mode if len(query_ids) > 1 else f"{mode}-{slug(query_ids[0])}"
    return Path(run_dir) / f"{name}.json"


def build(session, name: str):
    """Return the query a cell query name or a QUAIL-B query ID names."""
    import quail_b as benchmark
    from bench.quailb import build_query

    if name in QUERIES:
        return session.sql(QUERIES[name])
    return build_query(session, benchmark.get_query(name))


def answer_columns(query) -> list[str]:
    """Return the query's map answer columns, in plan order."""
    return [node.name for node in query.logical.operators().maps]


def step_totals(steps: Path, first: int) -> tuple[int, int, int]:
    """Return the step trace's lines, and its evictions and recomputed tokens.

    The evictions and recomputed tokens count from line `first` on.
    """
    lines = steps.read_text().splitlines() if steps.exists() else []
    records = [json.loads(line) for line in lines[first:]]
    return (len(lines), sum(r["preempted"] for r in records),
            sum(r["recomputed"] for r in records))


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=VOLUMES)
def run_mode(mode: str, query_ids: list[str], scale_factor: float,
             num_documents: int, repeats: int, run_dir: str,
             num_contracts: int = 0) -> str:
    """Run the queries under one admission mode and save their measurements.

    A query id names one of QUERIES, run on the first num_documents
    reviews, or a QUAIL-B query, run on its whole tables.

    Returns:
        The measurements as JSON.
    """
    from quail.baselines.vllm_trace import STEP_TRACE_ENV

    # the engine's workers start with the Session and read these variables
    directory = Path(run_dir)
    traces = directory / f"{mode}-traces"
    traces.mkdir(parents=True, exist_ok=True)
    steps = traces / "steps.jsonl"
    if mode in VLLM_MODES:
        os.environ[STEP_TRACE_ENV] = str(steps)
    else:
        os.environ["QUAIL_ANSWER_ADMISSION"] = mode
        os.environ["QUAIL_HEADROOM_TRACE"] = str(traces)
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    import quail

    table = pq.read_table(Path(DATA_DIR) / f"sf{scale_factor}" / "reviews.parquet")
    config = quail.EngineConfig(
        gpus=1, model="qwen3-4b-fp8",
        backend=mode if mode in VLLM_MODES else "quail", device="h100-sxm")
    report = {"mode": mode, "num_documents": num_documents,
              "num_contracts": num_contracts, "queries": {}}
    with quail.Session(config) as session:
        # warm up on copies whose first tokens differ from every review, so
        # no KV the warm-up writes serves a timed query
        warmup_rows = table.slice(len(table) - WARMUP_DOCUMENTS)
        warmup_rows = warmup_rows.set_column(
            warmup_rows.schema.get_field_index("body"), "body",
            pc.binary_join_element_wise("Warm-up copy.\n", warmup_rows["body"], ""))
        session.register("warmup", quail.DocumentProvider.from_table(
            warmup_rows, id_col="id"))
        warmup = next(iter(QUERIES.values())).replace("FROM reviews", "FROM warmup")
        session.sql(warmup).run().collect()
        register_tables(session, table, scale_factor, num_documents,
                        num_contracts)
        tokenizer = session.tokenizer
        for name in query_ids:
            runs = []
            for _ in range(repeats):
                first, _, _ = step_totals(steps, 0)
                query = build(session, name)
                result = query.run()
                rows = result.collect()
                digest = hashlib.sha256()
                lengths = []
                for column in answer_columns(query):
                    for text in rows.column(column).to_pylist():
                        digest.update((text or "").encode() + b"\0")
                        lengths.append(len(tokenizer(text or "")))
                maps = [metrics for metrics in
                        result.report.get("node_metrics", {}).values()
                        if "answer_evictions" in metrics]
                runs.append({
                    "wall_s": round(result.report["wall_s"], 2),
                    "answer_evictions": sum(m["answer_evictions"] for m in maps),
                    "recomputed_tokens": sum(m["recomputed_tokens"] for m in maps),
                    "answer_tokens": sum(m.get("answer_tokens", 0) for m in maps),
                    "decode_rows": sum(m.get("decode_rows", 0) for m in maps),
                    **{key: value for m in maps for key, value in m.items()
                       if key.startswith(("headroom_", "pass_", "admission_s"))
                       and value},
                    "answer_p50": int(np.percentile(lengths, 50)) if lengths else None,
                    "answer_p90": int(np.percentile(lengths, 90)) if lengths else None,
                    "answers_sha256": digest.hexdigest(),
                })
                if mode in VLLM_MODES:
                    _, evictions, recomputed = step_totals(steps, first)
                    runs[-1].update(answer_evictions=evictions,
                                    recomputed_tokens=recomputed)
                print(mode, name, runs[-1], flush=True)
            path = answers_path(run_dir, mode, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows.to_pydict()))
            report["queries"][name] = runs
    report_path(run_dir, mode, query_ids).write_text(json.dumps(report, indent=1))
    results_vol.commit()
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def compare(modes: str = "headroom=0,headroom", queries: str = ",".join(QUERIES),
            scale_factor: float = 0.1, num_documents: int = 1000, repeats: int = 2,
            num_contracts: int = 0):
    query_ids = [name for name in queries.split(",") if name]
    # IMDB-20 brings the reviews and CUAD-22 the contracts
    ensure_data.remote(scale_factor, list(dict.fromkeys(["IMDB-20", "CUAD-22"] + [
        name for name in query_ids if name not in QUERIES])), "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    # each query on its own GPU, so no KV or prefix cache carries over
    calls = [run_mode.spawn(mode, [query], scale_factor, num_documents, repeats,
                            run_dir, num_contracts)
             for mode in filter(None, modes.split(",")) for query in query_ids]
    for call in calls:
        print(call.get(), flush=True)


@app.function(image=image, cpu=8, memory=65536, timeout=3600, volumes=VOLUMES)
def estimate_sol(query: str, run_dir: str, mode: str, scale_factor: float,
                 num_documents: int, num_contracts: int = 0) -> str:
    """Price one query's SoL with the answers a saved run wrote.

    quail.speed_of_light_estimate: unlimited KV, ideal packing, each
    distinct prefix computed once, one decode pass per fed answer
    token, and the maps of one document reading it once per pass.

    Returns:
        The SoL record as JSON.
    """
    import pyarrow.parquet as pq

    import quail

    saved = json.loads(answers_path(run_dir, mode, query).read_text())
    table = pq.read_table(Path(DATA_DIR) / f"sf{scale_factor}" / "reviews.parquet")
    config = quail.EngineConfig(gpus=1, model="qwen3-4b-fp8", device="h100-sxm")
    with quail.Session(config) as session:
        ids_by_table = register_tables(session, table, scale_factor, num_documents,
                                       num_contracts)
        built = build(session, query)
        maps = {node.call.prompt: node for node in built.logical.operators().maps}
        # the one saved column that is not an answer holds the row ids
        (id_column,) = set(saved) - set(answer_columns(built))
        written = {}
        for prompt, node in maps.items():
            ids = ids_by_table[node.call.prompt.args[0].provider]
            by_id = dict(zip(map(str, saved[id_column]), saved[node.name]))
            written[prompt] = [by_id[str(row_id)] or "" for row_id in ids]

        def answer(prompt, assignment):
            (row,) = assignment.values()
            return written[prompt][row]

        sol = quail.speed_of_light_estimate(built, answer)
    record = {"query": query, "answers_from": f"{run_dir} ({mode})",
              "seconds": sol.seconds, "passes": sol.latency.passes,
              "components": {component.name: component.seconds
                             for component in sol.latency.components},
              "fresh_tokens": sol.work.tokens, "kv_read_tokens": sol.work.kv_read,
              "pairs": sol.work.pairs}
    path = answers_path(run_dir, mode, query).with_suffix(".sol.json")
    path.write_text(json.dumps(record, indent=1))
    results_vol.commit()
    return json.dumps(record, indent=1)


@app.local_entrypoint()
def sol(run_dir: str, queries: str = "CUAD-23", mode: str = "headroom",
        scale_factor: float = 0.1, num_documents: int = 0, num_contracts: int = 0):
    for query in filter(None, queries.split(",")):
        call = estimate_sol.spawn(query, run_dir, mode, scale_factor, num_documents,
                                  num_contracts)
        print(f"function call id: {call.object_id} ({query})", flush=True)
        print(call.get(), flush=True)

