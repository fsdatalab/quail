"""Speed-of-light estimates of every published QUAIL-B query for one model.

For each query, builds the Quail query on a Session for the model and
prices it with quail.speed_of_light_estimate, using the saved reference
labels as the answer and label oracles. The estimate computes every
distinct document prefix once, keeps KV and checkpoints without limit,
and has exact survivors from the reference labels. CPU only; the session
tokenizes the scanned columns and does not plan or run the query.

    uv run modal run experiments/cells/hybrid_sweep_sol.py \
        --model qwen3.5-4b-bf16 --sf 0.1 2>&1 | tee /tmp/hybrid_sweep_sol.log

The estimates are written to /results/hybrid_sweep/sol_<model>_sf<sf>.json
on the quail-results volume. A query the estimate does not price is
recorded with the reason.
"""

import json
import os

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
results = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)


@app.function(image=image, cpu=8, memory=65536, timeout=7200,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def estimate(model: str, sf: float, query_ids: list[str],
             collection_id: str) -> dict:
    """Price each query's ideal execution on one H100."""
    import pyarrow.parquet as pq

    import quail
    import quail_b
    from quail.bench.quailb import answer_oracle, build_query

    directory = f"/results/quailb_data/sf{sf}"
    only = list(query_ids) or None
    suite = quail_b.load_benchmark(only, scale_factor=sf, data_dir=directory,
                                   collection_id=collection_id or None)
    truth = suite.ground_truth
    out = {"model": model, "sf": sf, "collection_id": truth.collection_id,
           "reference_model": truth.reference_model, "queries": {}}
    session = quail.Session(quail.EngineConfig(model=model, device="h100-sxm"))
    for spec in suite.queries:
        query_id = spec.id
        names = {r.table for r in spec.info.relations}
        try:
            tables = {name: pq.read_table(f"{directory}/{name}.parquet")
                      for name in names}
            for name, table in tables.items():
                if name not in session.catalog:
                    session.register(name, quail.DocumentProvider.from_table(
                        table, id_col="id"))
            oracle = answer_oracle(truth, tables)
            sol = quail.speed_of_light_estimate(
                build_query(session, spec), oracle, label=oracle)
            out["queries"][query_id] = {
                "seconds": sol.seconds, "usd_per_query": sol.usd_per_query,
                "fresh_tokens": sol.fresh_tokens,
                "documents_by_alias": sol.documents_by_alias,
                "post_filter_counts": sol.post_filter_counts,
                "chunk_tokens": sol.chunk_tokens,
                "survivors": "exact, from the reference labels",
                "kv": "unlimited; every distinct prefix computed once"}
        except Exception as error:    # record why a query is not priced
            out["queries"][query_id] = {
                "not_priced": f"{type(error).__name__}: {error}"[:300]}
        print(query_id, json.dumps(out["queries"][query_id])[:240], flush=True)
    session.close()
    path = f"/results/hybrid_sweep/sol_{model}_sf{sf}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    results.commit()
    return {"path": path, **out}


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16", sf: float = 0.1, queries: str = "",
         collection_id: str = ""):
    call = estimate.spawn(model, sf, queries.split(",") if queries else [],
                          collection_id)
    print(f"estimate function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2, default=str), flush=True)
