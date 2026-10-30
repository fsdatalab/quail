r"""Compare answer admission modes on AI.MAP queries over IMDB reviews.

Each mode runs on its own H100: QUAIL_ANSWER_ADMISSION picks it before
the Session starts. A warm-up query on the last 64 reviews loads the
model and captures CUDA graphs; then each query runs `repeats` times on
the first `num_documents` reviews. Per run the cell records wall
seconds, answer evictions, recomputed tokens, answer-length
percentiles, and a hash of every answer, so modes can be checked for
the same output. Results go to
/results/ablations/answer_headroom/<run id>/<mode>.json.

    uv run modal run --detach experiments/cells/answer_headroom.py::compare \
        --modes prompt,headroom 2>&1 | tee "$run_log"
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data, results_vol
from quail.deploy.images import gpu_image

# the long-answer map drafts of experiments/cells/map_length_probe.py
NAMES = ("List every film, actor, director, and character the review names, "
         "with what the writer says about each.")
PLOT = ("Retell the plot of the film as the review describes it, in order, "
        "with every character and event the review mentions.")
REWRITE = ("Rewrite the review as a balanced one that gives both the strengths "
           "and the weaknesses of the film, keeping every point the writer makes.")
LIMIT = 1024
QUERIES = {
    "one long map": (f"SELECT r.id, AI.MAP(r.body, '{PLOT}', {LIMIT}) "
                     "AS plot FROM reviews r"),
    "three long maps": (f"SELECT r.id, AI.MAP(r.body, '{PLOT}', {LIMIT}) AS plot, "
                        f"AI.MAP(r.body, '{REWRITE}', {LIMIT}) AS balanced, "
                        f"AI.MAP(r.body, '{NAMES}', {LIMIT}) AS names "
                        "FROM reviews r"),
}

OUTPUT_DIR = "/results/ablations/answer_headroom"
WARMUP_DOCUMENTS = 64
image = gpu_image()


@app.function(image=image, gpu="H100!", memory=98304, timeout=4 * 3600,
              volumes=VOLUMES)
def run_mode(mode: str, scale_factor: float, num_documents: int, repeats: int,
             run_dir: str) -> str:
    """Run every query under one admission mode and save its measurements.

    Returns:
        The measurements as JSON.
    """
    os.environ["QUAIL_ANSWER_ADMISSION"] = mode
    import numpy as np
    import pyarrow.parquet as pq

    import quail

    table = pq.read_table(Path(DATA_DIR) / f"sf{scale_factor}" / "reviews.parquet")
    config = quail.EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                                device="h100-sxm")
    report = {"mode": mode, "num_documents": num_documents, "queries": {}}
    with quail.Session(config) as session:
        # warm up on reviews the timed queries do not read, so no KV is reused
        session.register("warmup", quail.DocumentProvider.from_table(
            table.slice(len(table) - WARMUP_DOCUMENTS), id_col="id"))
        warmup = next(iter(QUERIES.values())).replace("FROM reviews", "FROM warmup")
        session.sql(warmup).run().collect()
        session.register("reviews", quail.DocumentProvider.from_table(
            table.slice(0, num_documents), id_col="id"))
        tokenizer = session.tokenizer
        for name, sql in QUERIES.items():
            runs = []
            for _ in range(repeats):
                result = session.sql(sql).run()
                rows = result.collect()
                digest = hashlib.sha256()
                lengths = []
                for column in rows.column_names:
                    if column.endswith("id"):
                        continue
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
                    "headroom_pages": max(m.get("headroom_pages", 0.0) for m in maps),
                    "answer_p50": int(np.percentile(lengths, 50)),
                    "answer_p90": int(np.percentile(lengths, 90)),
                    "answers_sha256": digest.hexdigest(),
                })
                print(mode, name, runs[-1], flush=True)
            report["queries"][name] = runs
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{mode}.json").write_text(json.dumps(report, indent=1))
    results_vol.commit()
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def compare(modes: str = "prompt,headroom", scale_factor: float = 0.1,
            num_documents: int = 1000, repeats: int = 2):
    ensure_data.remote(scale_factor, ["IMDB-20"], "")
    run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
              f"{uuid.uuid4().hex[:8]}")
    run_dir = f"{OUTPUT_DIR}/{run_id}"
    print(f"run directory: {run_dir}", flush=True)
    calls = [run_mode.spawn(mode, scale_factor, num_documents, repeats, run_dir)
             for mode in filter(None, modes.split(","))]
    for call in calls:
        print(call.get(), flush=True)
