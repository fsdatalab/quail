"""Run draft map queries on Quail and report how long their answers run.

Draft queries need answers whose lengths vary well below their limit.
This runs each SQL over a QUAIL-B table on one H100 and prints, per map
column, the answer length percentiles in tokens and the share that hit
the limit. Tables come from the bench data volume (ensure_data).

    uv run --no-sync modal run experiments/cells/map_length_probe.py::probe_lengths
"""

import json

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, ensure_data
from quail.deploy.images import gpu_image

image = gpu_image()

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


@app.function(image=image, gpu="H100!", memory=98304, timeout=3600, volumes=VOLUMES)
def probe(scale_factor: float, num_documents: int) -> str:
    from pathlib import Path

    import numpy as np
    import pyarrow.parquet as pq

    import quail

    table = pq.read_table(Path(DATA_DIR) / f"sf{scale_factor}" / "reviews.parquet")
    table = table.slice(0, num_documents)
    report = {}
    config = quail.EngineConfig(gpus=1, model="qwen3-4b-fp8", backend="quail",
                                device="h100-sxm")
    with quail.Session(config) as session:
        session.register("reviews",
                         quail.DocumentProvider.from_table(table, id_col="id"))
        tokenizer = session.tokenizer
        for name, sql in QUERIES.items():
            result = session.sql(sql).run()
            rows = result.collect()
            columns = {}
            for column in rows.column_names:
                if column.endswith("id"):
                    continue
                lengths = np.array([len(tokenizer(text or ""))
                                    for text in rows.column(column).to_pylist()])
                columns[column] = {
                    "p10": int(np.percentile(lengths, 10)),
                    "p50": int(np.percentile(lengths, 50)),
                    "p90": int(np.percentile(lengths, 90)),
                    "mean": round(float(lengths.mean()), 1),
                    "at_limit": round(float((lengths >= LIMIT - 1).mean()), 3),
                }
            report[name] = {"wall_s": round(result.report["wall_s"], 1),
                            "columns": columns}
    return json.dumps(report, indent=1)


@app.local_entrypoint()
def probe_lengths(scale_factor: float = 0.1, num_documents: int = 1000):
    ensure_data.remote(scale_factor, ["IMDB-20"], "")
    print(probe.remote(scale_factor, num_documents))
