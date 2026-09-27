"""Label BIO-4 reaction categories once and reuse them at all three scales.

    uv run modal run --detach -m experiments.cells.bio4_labels::bio4 \
        2>&1 | tee /tmp/bio4-labels.log

Results and label sets stay on the quail-results volume.
"""

import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quail.bench import labeling as labels
from quail_b.data import PUBLISHED_CORPORA
from quail_b.predicates import PREDICATES

app = labels.app
SPECS = tuple(spec for spec in PREDICATES
              if spec.workload == "biodex" and spec.left_table == "terms")
SOURCES = {
    0.1: "gt_be81cb241d74555dc2da79b5b0662554",
    0.5: "gt_d6bd3ff26e8cc41a975acaa15150c685",
    1.0: "gt_2bae4ad2a1a87a9a16e5c2463f046de5",
}
SUMMARY_PATH = Path("/results/ablations/bio4-reference-labels.json")


def term_corpus(sf):
    """Read a saved corpus's term table and identity."""
    directory = labels.ROOT / "corpora" / PUBLISHED_CORPORA[sf]
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = {"terms": pq.read_table(directory / "terms.parquet").to_pylist()}
    identities = {
        spec.key: labels.label_set_identity(
            spec, manifest["corpus_id"], manifest["corpus_full_hash"])
        for spec in SPECS
    }
    return manifest, rows, identities


@app.function(
    image=labels.image, gpu="H100!", memory=98304, timeout=1800,
    volumes={"/root/.cache/huggingface": labels.hf_cache,
             "/root/.cache/kernels": labels.kernel_cache,
             "/results": labels.results_vol},
)
def classify():
    """Classify the full benchmark vocabulary and repeat a saved sample."""
    labels._mount()
    manifest, rows, identities = term_corpus(1.0)
    started = time.perf_counter()
    judge = labels.QuailJudge()
    try:
        for table, specs, batch_rows in labels.filter_groups(SPECS):
            labels._write_filter_parts(
                judge, labels.VerificationSample(), rows[table], specs,
                identities, manifest["corpus_id"], batch_rows)
        manifests = {spec.key: labels._complete_manifest(
            spec, identities[spec.key], rows) for spec in SPECS}
        verification = labels._saved_verification_sample(
            rows, identities, SPECS).run(judge)
        if verification["answer_differences"]:
            raise ValueError(f"repeat judgments differ: {verification}")
        result = {
            "scale_factor": 1.0,
            "label_sets": manifests,
            "verification": verification,
            "model_wall_s": judge.model_wall_s,
            "rows_answered": judge.rows_answered,
            "total_wall_s": time.perf_counter() - started,
        }
        labels._atomic_json(SUMMARY_PATH, result)
        labels.results_vol.commit()
        return result
    finally:
        judge.close()
        labels.kernel_cache.commit()


def copy_categories(sf, source_manifests):
    """Copy category labels by exact term content into a smaller corpus."""
    manifest, rows, identities = term_corpus(sf)
    for spec in SPECS:
        source = pq.read_table(source_manifests[spec.key]["compact_path"])
        by_content = {
            row["left_content_sha256"]: row
            for row in source.to_pylist()
        }
        identity = identities[spec.key]
        for start, end in labels._part_bounds(spec, identity, rows):
            output = []
            for row in rows["terms"][start:end]:
                saved = by_content[labels._content_hash(row, "term")]
                output.append(labels._answer_row(
                    spec, identity, manifest["corpus_id"], row, None,
                    saved["answer"], saved["label_source"], None))
            labels._atomic_parquet(
                labels._part_path(spec, identity, start, end), output)
        labels._complete_manifest(spec, identity, rows)
    labels.results_vol.commit()


def check_collection(sf, collection_id):
    """Load BIO-4's labels and count its reference output without inference."""
    import quail_b as benchmark
    from quail_b.scoring import RunOutput, expected_survivors, row_counts

    suite = benchmark.load_benchmark(
        "BIO-4", scale_factor=sf, collection_id=collection_id, root="/results")
    spec = suite.queries[0]
    survivors = expected_survivors(spec, suite.ground_truth, suite.tables)
    empty = pa.table({alias: pa.array([], pa.string()) for alias in ("r", "n", "c")})
    _, expected, _ = row_counts(
        spec, RunOutput(None, None, empty), suite.ground_truth, suite.tables)
    return {
        "reference_output_rows": expected,
        "filter_survivors": {alias: len(ids) for alias, ids in survivors.items()},
        "input_documents": {alias: suite.tables[table].num_rows
                            for alias, table in (("r", "reports"),
                                                 ("n", "terms"), ("c", "terms"))},
    }


@app.function(
    image=labels.publish_image, memory=32768, timeout=1800,
    volumes={"/results": labels.results_vol},
)
def finish(classification_call_id: str):
    """Derive smaller label sets and activate verified reference collections."""
    import modal

    source = modal.FunctionCall.from_id(classification_call_id).get()
    labels._mount()
    summaries = {}
    for sf in (1.0, 0.5, 0.1):
        if sf != 1.0:
            copy_categories(sf, source["label_sets"])
        summary = labels.activate_reused_collection(
            sf, PUBLISHED_CORPORA[sf], SOURCES[sf], "",
            relabeled_predicates=tuple(spec.key for spec in SPECS))
        summary["bio4"] = check_collection(sf, summary["collection_id"])
        summaries[str(sf)] = summary
        labels._atomic_json(SUMMARY_PATH, {**source, "collections": summaries})
        labels.results_vol.commit()
    return {"collections": summaries, "result_volume_path": str(SUMMARY_PATH)}


@app.local_entrypoint()
def bio4(classification_call_id: str = ""):
    """Submit classification and collection checks and print their call ids."""
    if not classification_call_id:
        print("PREDICTION: 8,288 new labels and 32 repeated judgments; "
              "no repeat differences and no inference at smaller scales.", flush=True)
        call = classify.spawn()
        classification_call_id = call.object_id
        print(f"function call id (classify): {classification_call_id}", flush=True)
        call.get()
    call = finish.spawn(classification_call_id)
    print(f"function call id (finish): {call.object_id}", flush=True)
    result = call.get()
    for sf, summary in result["collections"].items():
        print(sf, summary["collection_id"], summary["bio4"], flush=True)
    print(result["result_volume_path"], flush=True)
