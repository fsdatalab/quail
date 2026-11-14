r"""Run Quail Server on an H100 and compare its rows with a local Session's.

The cell starts `quail-server` as a process on the GPU machine, registers
the QUAIL-B reviews through a remote Session, and submits three queries:
a filter, a classification of the first 200 reviews, and a map of the
first 200 reviews. It then stops the server with SIGTERM and runs the
same queries in a local Session on the same GPU. Each path first runs a
warm-up query on copies of 64 reviews, so the model is loaded before a
timed query.

    uv run modal run --detach experiments/cells/server_smoke.py::compare \
        --scale-factor 0.1 2>&1 | tee "$run_log"

Per query and path the cell records seconds from submitting the query to
its collected rows, the engine's wall seconds from the region reports,
and the rows. The result goes to /results/server/<run id>/smoke.json.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

OUTPUT_DIR = "/results/server"
ENDPOINT = "127.0.0.1:50051"
WARMUP_DOCUMENTS = 64
SENTIMENT = "ARRAY['positive', 'negative', 'mixed']"
ENDING = "PROMPT('Does this review discuss the ending of the movie?\\n\\n{0}', r.body)"
WARMUP_SQL = ("SELECT w.id FROM warmup w "
              "WHERE AI.IF(PROMPT('Is it long?\\n\\n{0}', w.body))")
QUERIES = {
    "filter": f"SELECT r.id FROM reviews r WHERE AI.IF({ENDING})",
    "classify first 200": (
        f"SELECT r.id, AI.CLASSIFY(r.body, {SENTIMENT}) AS sentiment "
        "FROM reviews r ORDER BY r.id LIMIT 200"),
    "map first 200": (
        "SELECT r.id, AI.MAP(r.body, 'Name the film the review is about.', 16) "
        "AS film FROM reviews r ORDER BY r.id LIMIT 200"),
}

image = gpu_image()


def measure(session, sql: str, remote: bool) -> dict:
    """Run one query and return its seconds, engine seconds, and rows."""
    started = time.perf_counter()
    query = session.sql(sql)
    if remote:
        run = query.submit()
        table = run.result().collect()
        regions = run.report()
    else:
        result = query.run()
        table = result.collect()
        regions = result.report.get("regions") or [result.report]
    seconds = time.perf_counter() - started
    return {
        "seconds": round(seconds, 3),
        "wall_s": round(sum(region.get("wall_s", 0.0) for region in regions), 3),
        "fresh_tokens": sum(region.get("fresh_tokens", 0) for region in regions),
        "rows": {str(row.pop(table.column_names[0])): json.dumps(row, sort_keys=True)
                 for row in table.to_pylist()},
    }


def run_queries(session, reviews, remote: bool) -> dict:
    """Register the reviews, run the warm-up, and measure each query."""
    import pyarrow as pa

    import quail

    warmup = pa.table({
        "id": [f"w{index}" for index in range(WARMUP_DOCUMENTS)],
        "body": [f"Copy {index}.\n{body}" for index, body in enumerate(
            reviews.column("body").slice(0, WARMUP_DOCUMENTS).to_pylist())]})
    session.register("warmup", quail.DocumentProvider.from_table(warmup, id_col="id"))
    session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    started = time.perf_counter()
    measure(session, WARMUP_SQL, remote)
    found = {"warmup_seconds": round(time.perf_counter() - started, 3), "queries": {}}
    for name, sql in QUERIES.items():
        found["queries"][name] = measure(session, sql, remote)
        print("server" if remote else "local", name,
              {key: value for key, value in found["queries"][name].items()
               if key != "rows"}, flush=True)
    return found


@app.function(image=image, gpu="H100!", memory=98304, timeout=2 * 3600,
              volumes=VOLUMES)
def serve_and_query(scale_factor: float, run_dir: str) -> str:
    """Run the queries through Quail Server, then locally, and save both.

    Returns:
        The record as JSON, without the rows.
    """
    import signal
    import subprocess
    import sys
    import tempfile

    import pyarrow.flight as flight
    import pyarrow.parquet as pq

    import quail

    reviews = pq.read_table(Path(DATA_DIR) / f"sf{scale_factor}" / "reviews.parquet")
    config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    record = {"scale_factor": scale_factor, "num_reviews": reviews.num_rows}
    with tempfile.TemporaryDirectory() as data:
        started = time.perf_counter()
        server = subprocess.Popen(
            [sys.executable, "-m", "quail.remote.server", "--model", config.model,
             "--device", config.device, "--listen", ENDPOINT, "--data", data])
        try:
            client = flight.FlightClient(f"grpc://{ENDPOINT}")
            while True:
                try:
                    list(client.list_actions())
                    break
                except flight.FlightUnavailableError:
                    if server.poll() is not None:
                        raise RuntimeError("quail-server exited") from None
                    if time.perf_counter() - started > 600:
                        raise RuntimeError("quail-server did not start") from None
                    time.sleep(0.5)
            client.close()
            record["server_listening_seconds"] = round(
                time.perf_counter() - started, 3)
            with quail.Session(config=config,
                               endpoint=f"grpc://{ENDPOINT}") as session:
                record["server"] = run_queries(session, reviews, remote=True)
        finally:
            stopping = time.perf_counter()
            server.send_signal(signal.SIGTERM)
            try:
                record["server_exit_code"] = server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()
                record["server_exit_code"] = "killed after 120 s"
            record["server_stop_seconds"] = round(time.perf_counter() - stopping, 3)
    with quail.Session(config) as session:
        record["local"] = run_queries(session, reviews, remote=False)
    for name in QUERIES:
        served = record["server"]["queries"][name].pop("rows")
        local = record["local"]["queries"][name].pop("rows")
        record["server"]["queries"][name]["num_rows"] = len(served)
        record["local"]["queries"][name]["num_rows"] = len(local)
        record.setdefault("comparison", {})[name] = {
            "same_ids": served.keys() == local.keys(),
            "num_rows_differing": sum(
                served[key] != local[key] for key in served.keys() & local.keys()),
        }
    path = Path(run_dir) / "smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str))
    results_vol.commit()
    return json.dumps(record, default=str)


@app.local_entrypoint()
def compare(scale_factor: float = 0.1):
    # this QUAIL-B query brings the reviews table
    ensure_data.remote(scale_factor, ["IMDB-20"], "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    call = serve_and_query.spawn(scale_factor, run_dir)
    print(f"function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)
