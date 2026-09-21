"""Add the CUAD contracts to the QUAIL-B corpora and label them from the annotation.

    uv run modal run --detach -m experiments.cells.cuad_labels::cuad \
        2>&1 | tee /tmp/cuad-labels.log

One CPU call per published scale factor: build the corpus with the two
contract tables (a new corpus id), write the eight CUAD label sets from
the lawyer annotation, and activate a collection that takes every other
label set from the corpus's current collection. Then count, without
inference, the rows each CUAD query reads and the rows its reference
labels keep. The summary and the label sets stay on the quail-results
volume.
"""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from quail.bench import labeling as labels
from quail_b.data import FILE_TABLES, PUBLISHED_CORPORA

app = labels.app
WORKLOAD = "cuad"
QUERIES = ("CUAD-1", "CUAD-2", "CUAD-3", "CUAD-4", "CUAD-5")
SUMMARY_PATH = Path("/results/ablations/cuad-reference-labels.json")
PREDICTION_TEXT = (
    "Every table but the two contract tables hashes as it did, so all "
    "23 existing label sets are reused. The annotation answers every "
    "contract and page with no inference: 510 contracts and 9,348 pages "
    "at sf=1.0, of which 430 contracts have at most 32 pages. The "
    "sf=0.1 selectivities in quail_b.queries put CUAD-2 near 1% of "
    "pages and CUAD-5 near 1% of the bounded contracts."
)


def source_collection(sf: float) -> str:
    """The collection the published corpus at this scale factor points at."""
    path = (labels.ROOT / "corpora" / PUBLISHED_CORPORA[sf]
            / f"active_collection.{labels.PROMPT_FORMAT}.json")
    return json.loads(path.read_text())["collection_id"]


def check_queries(sf: float, corpus_id: str, collection_id: str) -> dict:
    """Each CUAD query's input rows and reference survivors, without inference."""
    from quail_b import get_query
    from quail_b.labels import load_ground_truth
    from quail_b.scoring import expected_rows, expected_survivors, relation_ids

    truth = load_ground_truth(
        "/results", scale_factor=sf, corpus_id=corpus_id,
        collection_id=collection_id)
    tables = {
        name: pq.read_table(labels.ROOT / "corpora" / corpus_id / f"{name}.parquet")
        for name in FILE_TABLES
    }
    checks = {}
    for query_id in QUERIES:
        spec = get_query(query_id)
        survivors = expected_survivors(spec, truth, tables)
        checks[query_id] = {
            "input_rows": {
                relation.alias: len(relation_ids(relation, tables[relation.table]))
                for relation in spec._info.relations},
            "filter_survivors": {alias: len(ids)
                                 for alias, ids in survivors.items()},
            "reference_output_rows": expected_rows(spec, truth, tables).num_rows,
        }
    return checks


# Building the sf=1.0 corpus downloads SWE-Next and tokenizes its
# traces, the slowest table; the CUAD archive is one more download.
@app.function(
    image=labels.image, memory=2 * labels.CPU_MEMORY_MB,
    timeout=2 * labels.CPU_TIMEOUT_S,
    volumes={"/root/.cache/huggingface": labels.hf_cache,
             "/results": labels.results_vol},
)
def build(sf: float) -> dict:
    """Build one scale factor's corpus, its CUAD labels, and its collection."""
    labels._mount()
    started = time.perf_counter()
    prepared = labels.prepare_corpus(sf)
    corpus_id = prepared["corpus"]["corpus_id"]
    partial = labels.label_annotated_workload(corpus_id, WORKLOAD)
    source = source_collection(sf)
    summary = labels.activate_reused_collection(sf, corpus_id, source, WORKLOAD)
    labels.results_vol.commit()
    return {
        "scale_factor": sf,
        "corpus_id": corpus_id,
        "source_corpus_id": PUBLISHED_CORPORA[sf],
        "source_collection_id": source,
        "collection_id": summary["collection_id"],
        "reused_predicates": summary["reused_predicates"],
        "new_predicates": summary["new_predicates"],
        "label_sets": {
            key: {field: manifest[field]
                  for field in ("label_set_id", "rows", "true_rows")}
            for key, manifest in partial["manifests"].items()},
        "queries": check_queries(sf, corpus_id, summary["collection_id"]),
        "total_wall_s": round(time.perf_counter() - started, 1),
    }


@app.function(
    image=labels.image, memory=labels.CPU_MEMORY_MB,
    timeout=labels.CPU_TIMEOUT_S,
    volumes={"/results": labels.results_vol},
)
def finish(build_call_ids: str) -> dict:
    """Gather the build results into one summary on the volume."""
    import modal

    results = [modal.FunctionCall.from_id(call_id).get()
               for call_id in build_call_ids.split(",")]
    labels._mount()
    summary = {
        "cell": "cuad_reference_labels",
        "prediction": PREDICTION_TEXT,
        "scale_factors": {str(r["scale_factor"]): r for r in results},
    }
    labels._atomic_json(SUMMARY_PATH, summary)
    labels.results_vol.commit()
    return {**summary, "result_volume_path": str(SUMMARY_PATH)}


@app.local_entrypoint()
def cuad(sfs: str = "0.1,0.5,1.0", build_call_ids: str = ""):
    """Build every scale factor side by side, then write the summary."""
    print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
    if not build_call_ids:
        calls = {sf: build.spawn(float(sf)) for sf in sfs.split(",")}
        for sf, call in calls.items():
            print(f"function call id (build sf={sf}): {call.object_id}",
                  flush=True)
        build_call_ids = ",".join(call.object_id for call in calls.values())
        for call in calls.values():
            call.get()
    call = finish.spawn(build_call_ids)
    print(f"function call id (finish): {call.object_id}", flush=True)
    result = call.get()
    for sf, entry in result["scale_factors"].items():
        print(sf, entry["corpus_id"], entry["collection_id"], flush=True)
        for query_id, check in entry["queries"].items():
            print(f"  {query_id}: {check}", flush=True)
    print(result["result_volume_path"], flush=True)
