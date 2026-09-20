"""Speed of light estimates for QUAIL-B queries, on the CPU.

Prices each query's minimum work with the model's answer rows counted
once per evaluation: a decoder answers on the suffix's last row, a
diffusion model on the canvas rows appended to every evaluation. No
inference runs; the answers come from the saved reference labels.

Usage, from the repository root, with the benchmark tables pulled off
the volume (`modal volume get quail-results benchmarks/quailb/data/sf0.1 $W`):

    uv run python experiments/quailb_sol.py $W/sf0.1
        --collection gt_be81cb241d74555dc2da79b5b0662554
        --model qwen3-4b-fp8 --model diffusion-gemma-26b-a4b-fp8-canvas256
        --query IMDB-1,IMDB-2

(one command; the options continue the first line)
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

import quail
import quail_b as benchmark
from quail.bench.quailb import answer_oracle, build_query, register_tables
from quail.bench.substrait import read_plan
from quail.planner.plan import EngineConfig


def estimates(data_dir: str, collection: str, models: list[str],
              query_ids: list[str]) -> list[dict]:
    """Return one record per query and model with the SoL estimate."""
    suite = benchmark.load_benchmark(
        query_ids, scale_factor=0.1, data_dir=data_dir,
        collection_id=collection or None)
    tables = {
        path.stem: pq.read_table(path)
        for path in sorted(Path(data_dir).glob("*.parquet"))
    }
    answer = answer_oracle(suite.ground_truth, tables)
    records = []
    for model in models:
        config = EngineConfig(gpus=1, model=model, backend="quail",
                              device="h100-sxm")
        with quail.Session(config) as session:
            register_tables(session, data_dir)
            for query_id in query_ids:
                spec = benchmark.get_query(query_id)
                query = build_query(session, spec)
                estimate = quail.speed_of_light_estimate(query, answer)
                records.append({
                    "query": query_id, "model": model,
                    "sol_s": estimate.seconds,
                    "fresh_tokens": estimate.fresh_tokens,
                    "usd_per_query": estimate.usd_per_query,
                    "filter_evaluations": estimate.filter_evaluations,
                    "join_pair_evaluations": estimate.join_pair_evaluations,
                    "relations": [r.table for r in read_plan(spec.plan).relations],
                })
                print(json.dumps(records[-1]), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--collection", default="")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--query", required=True,
                        help="comma-separated query ids")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    records = estimates(args.data_dir, args.collection, args.model,
                        [q.strip() for q in args.query.split(",") if q.strip()])
    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
