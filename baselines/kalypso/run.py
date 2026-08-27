"""Kalypso baseline orchestrator: exports quail data as .txt files
and drives KalypsoWorker over the 3 join queries.

    uv run modal run -m baselines.kalypso.run::main
"""

import json
import time
from pathlib import Path

import modal

from .config import APP_NAME, DATA_DIR, SF
from .worker import KalypsoWorker, app, hf_cache_vol, results_vol

orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "pandas",
        "pyarrow",
        "numpy",
        "datasets",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("quail")
    .add_local_python_source("baselines")
)

KALYPSO_DATA_DIR = "/results/kalypso_data"


def _export_table_as_txt(data_dir: str, sf: float, table_name: str,
                         text_col: str, vol) -> str:
    """Write each row of a quailb parquet table as a numbered .txt file
    under KALYPSO_DATA_DIR. Returns the directory path."""
    import pyarrow.parquet as pq

    src = Path(data_dir) / f"sf{sf}" / f"{table_name}.parquet"
    t = pq.read_table(src)
    texts = t.column(text_col).to_pylist()

    out_dir = Path(KALYPSO_DATA_DIR) / f"sf{sf}" / table_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, text in enumerate(texts):
        (out_dir / f"{idx}.txt").write_text(str(text), encoding="utf-8")

    print(f"[kalypso-run] exported {len(texts)} rows to {out_dir}", flush=True)
    vol.commit()
    return str(out_dir)


def build_join_queries():
    """The 3 join queries matching the vllm_opbench baseline."""
    return {
        "join-reports": {
            "left_table": "reports",
            "left_text_col": "report",
            "right_table": "terms",
            "right_text_col": "term",
            "instruction": (
                "Does the medical report describe the reaction "
                "as something the patient experienced?"
            ),
        },
        "join-claims": {
            "left_table": "claims",
            "left_text_col": "claim",
            "right_table": "evidence",
            "right_text_col": "text",
            "instruction": (
                "Does the Wikipedia passage support the claim?"
            ),
        },
        "join-imdb": {
            "left_table": "reviews",
            "left_text_col": "body",
            "right_table": "aspects",
            "right_text_col": "aspect",
            "instruction": (
                "Does the review discuss the movie aspect?"
            ),
        },
    }


@app.function(
    image=orchestrator_image,
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
def run_baseline(
    model: str = "qwen3-4b",
    query_id: str | None = None,
) -> dict:
    from quail.bench.quailb import build_sets

    build_sets(DATA_DIR, SF)

    queries = build_join_queries()
    ids = [query_id] if query_id else list(queries)

    exported: dict[str, str] = {}
    for qid in ids:
        q = queries[qid]
        for side in ("left", "right"):
            tname = q[f"{side}_table"]
            if tname not in exported:
                exported[tname] = _export_table_as_txt(
                    DATA_DIR, SF, tname, q[f"{side}_text_col"], results_vol
                )

    worker = KalypsoWorker(model=model)

    out_dir = Path("/results/kalypso_baseline") / time.strftime(
        "%Y-%m-%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for qid in ids:
        q = queries[qid]
        left_path = exported[q["left_table"]]
        right_path = exported[q["right_table"]]

        result = worker.run_join.remote(
            left_path=left_path,
            right_path=right_path,
            instruction=q["instruction"],
            query_name=qid,
        )

        entry = {
            "query": qid,
            "model": model,
            "wall_time_s": result["wall_time_s"],
            "server_latency_s": result["server_latency_s"],
            "num_output_rows": result["num_output_rows"],
        }
        summary.append(entry)

        with open(out_dir / f"{qid}.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(
                {"model": model, "sf": SF, "queries": summary},
                f,
                indent=2,
            )
        results_vol.commit()

        print(
            f"[kalypso-run] {qid}: wall={result['wall_time_s']:.2f}s "
            f"server_latency={result['server_latency_s']}s "
            f"output_rows={result['num_output_rows']}",
            flush=True,
        )

    print(f"[kalypso-run] saved {out_dir}/summary.json", flush=True)
    return {"out_dir": str(out_dir), "n_queries": len(summary)}


@app.local_entrypoint()
def main(model: str = "qwen3-4b", query: str = ""):
    fc = run_baseline.spawn(model=model, query_id=(query or None))
    print(f"function call id: {fc.object_id}")
    print(fc.get())
