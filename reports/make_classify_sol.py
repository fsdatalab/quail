"""SoL of the sf 0.5 QUAIL-B classification queries, reference survivors.

Runs Quail's `speed_of_light_estimate` on each query with QUAIL-B's saved
Qwen3 32B labels (collection gt_397dae1857e5850c84b61dccabb3431c) as
the answer and label oracles, for Qwen3 4B and DiffusionGemma, on the
CPU. Needs Quail at e2b5889 or later (fsdatalab/quail#178 with #180).
The labels load from QUAIL-B's public bucket. Run from the repository
root:

    W=/tmp/classify-sol; mkdir -p "$W/data"
    uv run modal volume get quail-results quailb_data/sf0.5 "$W/data/"
    uv run python reports/make_classify_sol.py "$W/data/sf0.5" \
      "$W/sol_reference_sf0.5.json"
    uv run modal volume put quail-results "$W/sol_reference_sf0.5.json" \
      sol/2026-10-01-classify-sf0.5.json

A third argument, a comma-separated list of query IDs, limits the run.
"""

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

import quail
from quail.bench.quailb import answer_oracle, build_query
from quail.planner.plan import EngineConfig
from quail.specs import DIFFUSION_GEMMA_26B_FP8, QWEN3_4B_FP8
from quail_b.data import CORPUS_COLUMNS
from quail_b.labels import load_ground_truth
from quail_b.queries import get_query

COLLECTION = "gt_397dae1857e5850c84b61dccabb3431c"
QUERIES = ([f"IMDB-{n}" for n in range(11, 16)]
           + ["BIO-5", "BIO-6", "FEV-11", "LEP-6"]
           + [f"AGENT-{n}" for n in range(3, 6)])
MODELS = {"qwen": QWEN3_4B_FP8, "dgemma": DIFFUSION_GEMMA_26B_FP8}


def main():
    """Write each query's SoL estimate for both models to the output file."""
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else QUERIES
    tables = {table: pq.read_table(data / f"{table}.parquet",
                                   columns=list(columns))
              for table, columns in CORPUS_COLUMNS.items()
              if (data / f"{table}.parquet").exists()}
    truth = load_ground_truth(None, 0.5, collection_id=COLLECTION)
    oracle = answer_oracle(truth, tables)
    results = json.loads(out.read_text()) if out.exists() else {}
    for key, model in MODELS.items():
        session = quail.Session(EngineConfig(model=model.name,
                                             device="h100-sxm"))
        for name, table in tables.items():
            session.register(name, quail.DocumentProvider.from_table(
                table, id_col="id"))
        for query_id in only:
            started = time.time()
            query = build_query(session, get_query(query_id))
            estimate = quail.speed_of_light_estimate(
                query, oracle, label=oracle)
            results.setdefault(key, {})[query_id] = {
                "sol_s": estimate.seconds,
                "tokens": estimate.fresh_tokens,
                "pairs": estimate.work.pairs,
                "classification_evaluations":
                    estimate.classification_evaluations,
                "join_pair_evaluations": estimate.join_pair_evaluations,
                "relation_order": list(estimate.relation_order),
                "chunk_tokens": estimate.chunk_tokens,
            }
            print(f"{key} {query_id}: SoL {estimate.seconds:.2f} s, "
                  f"{estimate.fresh_tokens:,.0f} tokens, "
                  f"{time.time() - started:.0f} s to estimate", flush=True)
            out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
