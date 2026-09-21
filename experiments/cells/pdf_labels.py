"""Add a PDF workload's tables to the QUAIL-B corpora and label its predicates.

    uv run modal run --detach -m experiments.cells.pdf_labels::pdf \
        --workload financebench 2>&1 | tee /tmp/pdf-labels.log

For each published scale factor: build the corpus with the workload's
tables (a new corpus id), label the workload's predicates (on a GPU
when a model judges any of them, else from the annotation on the CPU),
and activate a collection that takes every other label set from the
corpus's current collection. Then count, without inference, the rows
each of the workload's queries reads and the rows its reference labels
keep. The summary and the label sets stay on the quail-results volume.
"""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from quail.bench import labeling as labels
from quail_b.data import FILE_TABLES, PUBLISHED_CORPORA
from quail_b.queries import QUERY_FAMILY_WORKLOADS, QUERY_ORDER

app = labels.app
CUAD_PREDICTION_TEXT = (
    "Every table but the two contract tables hashes as it did, so all "
    "23 existing label sets are reused. The annotation answers every "
    "contract and page with no inference: 510 contracts and 9,348 pages "
    "at sf=1.0, of which 430 contracts have at most 32 pages. The "
    "sf=0.1 selectivities in quail_b.queries put CUAD-2 near 1% of "
    "pages and CUAD-5 near 1% of the bounded contracts.")
FINANCEBENCH_PREDICTION_TEXT = (
    "Every table but the two filing tables hashes as it did, so all 31 "
    "existing label sets are reused. The evidence annotation answers "
    "the page join with no inference; the judge reads 150 questions at "
    "sf=1.0 for the calculation filter, under a minute of model time. "
    "About a third of the questions need a calculation (the 50 "
    "metrics-generated ones and some of the rest), and a question's "
    "filing has one to three evidence pages among 100 to 400.")
PREDICTIONS = {"cuad": CUAD_PREDICTION_TEXT,
               "financebench": FINANCEBENCH_PREDICTION_TEXT}


def summary_path(workload: str) -> Path:
    return Path(f"/results/ablations/{workload}-reference-labels.json")


def workload_queries(workload: str) -> tuple[str, ...]:
    """The benchmark queries whose family this workload labels."""
    families = [family for family, name in QUERY_FAMILY_WORKLOADS.items()
                if name == workload]
    return tuple(query_id for query_id in QUERY_ORDER
                 if query_id.split("-")[0] in families)


def source_collection(sf: float) -> str:
    """The collection the published corpus at this scale factor points at."""
    path = (labels.ROOT / "corpora" / PUBLISHED_CORPORA[sf]
            / f"active_collection.{labels.PROMPT_FORMAT}.json")
    return json.loads(path.read_text())["collection_id"]


def check_queries(sf: float, corpus_id: str, collection_id: str,
                  queries: tuple[str, ...]) -> dict:
    """Each query's input rows and reference survivors, without inference."""
    from quail_b import get_query
    from quail_b.labels import load_ground_truth
    from quail_b.scoring import expected_rows, expected_survivors, relation_ids

    truth = load_ground_truth(
        "/results", scale_factor=sf, corpus_id=corpus_id,
        collection_id=collection_id)
    tables = {}
    checks = {}
    for query_id in queries:
        spec = get_query(query_id)
        for relation in spec._info.relations:
            if relation.table not in tables:
                tables[relation.table] = pq.read_table(
                    labels.ROOT / "corpora" / corpus_id
                    / f"{relation.table}.parquet")
        survivors = expected_survivors(spec, truth, tables)
        checks[query_id] = {
            "input_rows": {
                relation.alias: len(relation_ids(relation, tables[relation.table]))
                for relation in spec._info.relations},
            "filter_survivors": {alias: len(ids)
                                 for alias, ids in survivors.items()},
            "reference_output_rows": expected_rows(spec, truth, tables).num_rows,
            "file_tables": sorted({relation.table
                                   for relation in spec._info.relations}
                                  & set(FILE_TABLES)),
        }
    return checks


# Building the sf=1.0 corpus downloads SWE-Next and tokenizes its
# traces, the slowest table; the PDF sets add their file downloads.
@app.function(
    image=labels.image, memory=2 * labels.CPU_MEMORY_MB,
    timeout=2 * labels.CPU_TIMEOUT_S,
    volumes={"/root/.cache/huggingface": labels.hf_cache,
             "/results": labels.results_vol},
)
def build(sf: float) -> str:
    """Build one scale factor's corpus on the volume; return its id."""
    labels._mount()
    prepared = labels.prepare_corpus(sf)
    labels.results_vol.commit()
    return prepared["corpus"]["corpus_id"]


@app.function(
    image=labels.image, memory=labels.CPU_MEMORY_MB,
    timeout=labels.CPU_TIMEOUT_S,
    volumes={"/results": labels.results_vol},
)
def activate(sf: float, corpus_id: str, workload: str, partial: str) -> dict:
    """Activate the corpus's collection from the new labels and the reused ones."""
    labels._mount()
    manifests = json.loads(partial)["manifests"]
    source = source_collection(sf)
    summary = labels.activate_reused_collection(sf, corpus_id, source, workload)
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
                  for field in ("label_set_id", "rows", "true_rows",
                                "source_rows")}
            for key, manifest in manifests.items()},
        "queries": check_queries(sf, corpus_id, summary["collection_id"],
                                 workload_queries(workload)),
    }


def label_scale_factor(sf: float, workload: str) -> dict:
    """Build, label, and activate one scale factor; print each call id."""
    started = time.perf_counter()
    call = build.spawn(sf)
    print(f"function call id (build sf={sf}): {call.object_id}", flush=True)
    corpus_id = call.get()
    call = labels.spawn_workload(corpus_id, workload)
    print(f"function call id (label {workload} sf={sf}): {call.object_id}",
          flush=True)
    partial = call.get()
    call = activate.spawn(sf, corpus_id, workload, partial)
    print(f"function call id (activate sf={sf}): {call.object_id}", flush=True)
    result = call.get()
    result["total_wall_s"] = round(time.perf_counter() - started, 1)
    return result


@app.function(
    image=labels.image, memory=labels.CPU_MEMORY_MB,
    timeout=labels.CPU_TIMEOUT_S,
    volumes={"/results": labels.results_vol},
)
def finish(workload: str, results: str) -> dict:
    """Write the per scale factor results as one summary on the volume."""
    labels._mount()
    path = summary_path(workload)
    summary = {
        "cell": f"{workload}_reference_labels",
        "prediction": PREDICTIONS.get(workload, ""),
        "scale_factors": {str(r["scale_factor"]): r
                          for r in json.loads(results)},
    }
    labels._atomic_json(path, summary)
    labels.results_vol.commit()
    return {**summary, "result_volume_path": str(path)}


@app.local_entrypoint()
def pdf(workload: str, sfs: str = "0.1,0.5,1.0"):
    """Label every scale factor side by side, then write the summary."""
    from concurrent.futures import ThreadPoolExecutor

    if workload not in labels.WORKLOADS:
        raise SystemExit(f"unknown workload {workload!r}; one of "
                         f"{sorted(labels.WORKLOADS)}")
    print(f"PREDICTION: {PREDICTIONS.get(workload, '(none written)')}",
          flush=True)
    scale_factors = [float(sf) for sf in sfs.split(",")]
    with ThreadPoolExecutor(len(scale_factors)) as pool:
        results = list(pool.map(
            lambda sf: label_scale_factor(sf, workload), scale_factors))
    call = finish.spawn(workload, json.dumps(results))
    print(f"function call id (finish): {call.object_id}", flush=True)
    result = call.get()
    for sf, entry in result["scale_factors"].items():
        print(sf, entry["corpus_id"], entry["collection_id"], flush=True)
        for query_id, check in entry["queries"].items():
            print(f"  {query_id}: {check}", flush=True)
    print(result["result_volume_path"], flush=True)
