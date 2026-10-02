"""Label IMDB-16's five review guideline checks and build its collections.

    uv run modal run --detach -m experiments.cells.imdb16_labels::imdb16 \
        2>&1 | tee /tmp/imdb16-labels.log

Each check is labeled over the sf=1.0 reviews on its own H100 with
Qwen3 32B through Quail, then copied to sf=0.5 and sf=0.1 by review
content. The collections reuse every other label set of the published
ones. Results and label sets stay on the quail-results volume.
"""

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from quail.bench import labeling as labels
from quail_b.data import PUBLISHED_CORPORA
from quail_b.predicates import (
    CLASSIFY_PREDICATES,
    PREDICATES,
    workload_specs,
)

app = labels.app
SPECS = tuple(spec for spec in PREDICATES
              if spec.key.startswith("quailb.imdb.review.follows_"))
GENRE_KEY = "quailb.imdb.review.genre"
KEPT_GENRES = frozenset({"war", "western"})
# the published collections this pass adds the five label sets to
SOURCES = {
    0.1: "gt_72abc9af3feaea668e493ece67e980a0",
    0.5: "gt_397dae1857e5850c84b61dccabb3431c",
    1.0: "gt_a72baf70eae95e6b1681a11091eaafca",
}
SUMMARY_PATH = Path("/results/ablations/imdb16-reference-labels.json")
PREDICTION_TEXT = (
    "50,000 sf=1.0 reviews x 5 checks = 250,000 new labels, about 61 M "
    "fresh tokens per check; at 14,000 fresh tokens/s each check takes "
    "about 75 minutes on its own H100 (about 6.3 H100 hours, $25), the "
    "16 repeated judgments per check match, and sf=0.5 and sf=0.1 copy "
    "every label by content with no inference.")


def review_corpus(sf):
    """Read a saved corpus's review table and the five label-set identities."""
    directory = labels.ROOT / "corpora" / PUBLISHED_CORPORA[sf]
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = {"reviews": pq.read_table(directory / "reviews.parquet").to_pylist()}
    identities = {
        spec.key: labels.label_set_identity(
            spec, manifest["corpus_id"], manifest["corpus_full_hash"])
        for spec in SPECS
    }
    return manifest, rows, identities


def batch_rows():
    """Rows per part, as _part_bounds derives it for the IMDB filter group."""
    return next(n for _table, members, n
                in labels.filter_groups(workload_specs("imdb"))
                if SPECS[0] in members)


@app.function(
    image=labels.image, gpu="H100!", memory=98304, timeout=4 * 3600,
    volumes={"/root/.cache/huggingface": labels.hf_cache,
             "/root/.cache/kernels": labels.kernel_cache,
             "/results": labels.results_vol},
)
def check_one(key: str):
    """Label one check over every sf=1.0 review and repeat a saved sample."""
    labels._mount()
    spec = next(spec for spec in SPECS if spec.key == key)
    manifest, rows, identities = review_corpus(1.0)
    started = time.perf_counter()
    judge = labels.QuailJudge()
    try:
        labels._write_filter_parts(
            judge, labels.VerificationSample(), rows["reviews"], [spec],
            identities, manifest["corpus_id"], batch_rows())
        label_set = labels._complete_manifest(spec, identities[key], rows)
        verification = labels._saved_verification_sample(
            rows, identities, (spec,)).run(judge)
        if verification["answer_differences"]:
            raise ValueError(f"repeat judgments differ: {verification}")
        return {
            "key": key,
            "label_set": label_set,
            "verification": verification,
            "boot_s": judge.boot_s,
            "model_wall_s": judge.model_wall_s,
            "rows_answered": judge.rows_answered,
            "total_wall_s": time.perf_counter() - started,
        }
    finally:
        judge.close()
        labels.kernel_cache.commit()


def copy_checks(sf, source_label_sets):
    """Copy the check labels by exact review content into a smaller corpus."""
    manifest, rows, identities = review_corpus(sf)
    for spec in SPECS:
        source = pq.read_table(source_label_sets[spec.key]["compact_path"])
        by_content = {row["left_content_sha256"]: row
                      for row in source.to_pylist()}
        identity = identities[spec.key]
        for start, end in labels._part_bounds(spec, identity, rows):
            output = []
            for row in rows["reviews"][start:end]:
                saved = by_content[labels._content_hash(row, "body")]
                output.append(labels._answer_row(
                    spec, identity, manifest["corpus_id"], row, None,
                    saved["answer"], saved["label_source"], None))
            labels._atomic_parquet(
                labels._part_path(spec, identity, start, end), output)
        labels._complete_manifest(spec, identity, rows)
    labels.results_vol.commit()


def selectivities(sf, collection_id):
    """Each check's TRUE share, the genre filter's share, and IMDB-16's rows."""
    from quail_b.labels import load_ground_truth

    truth = load_ground_truth(
        root="/results", scale_factor=sf, collection_id=collection_id,
        templates=[spec.template for spec in SPECS]
        + [next(spec.template for spec in CLASSIFY_PREDICATES
                if spec.key == GENRE_KEY)])
    reviews = None
    passing = None
    shares = {}
    for spec in SPECS:
        table = truth.predicates[spec.key].table
        kept = {left for left, answer in zip(
            table.column("left_id").to_pylist(),
            table.column("answer").to_pylist()) if answer}
        reviews = table.num_rows
        shares[spec.key] = round(len(kept) / table.num_rows, 4)
        passing = kept if passing is None else passing & kept
    genre = truth.predicates[GENRE_KEY].table
    genres = dict(zip(genre.column("left_id").to_pylist(),
                      genre.column("label").to_pylist()))
    kept_genre = {left for left, label in genres.items() if label in KEPT_GENRES}
    return {
        "reviews": reviews,
        "check_true_share": shares,
        "all_checks_true_share": round(len(passing) / reviews, 4),
        "genre_kept_share": round(len(kept_genre) / len(genres), 4),
        "reference_output_rows": len(passing & kept_genre),
    }


@app.function(
    image=labels.publish_image, memory=32768, timeout=3600,
    volumes={"/results": labels.results_vol},
)
def finish(call_ids: list[str]):
    """Copy to smaller corpora, activate the collections, and count shares."""
    import modal

    results = [modal.FunctionCall.from_id(call_id).get() for call_id in call_ids]
    label_sets = {result["key"]: result["label_set"] for result in results}
    labels._mount()
    collections = {}
    for sf in (1.0, 0.5, 0.1):
        if sf != 1.0:
            copy_checks(sf, label_sets)
        summary = labels.activate_reused_collection(
            sf, PUBLISHED_CORPORA[sf], SOURCES[sf], "",
            relabeled_predicates=tuple(spec.key for spec in SPECS))
        collections[str(sf)] = {
            "collection_id": summary["collection_id"],
            "imdb16": selectivities(sf, summary["collection_id"]),
        }
        labels._atomic_json(SUMMARY_PATH, {
            "prediction": PREDICTION_TEXT, "checks": results,
            "collections": collections})
        labels.results_vol.commit()
    return {"collections": collections, "result_volume_path": str(SUMMARY_PATH)}


@app.local_entrypoint()
def imdb16(check_call_ids: str = ""):
    """Submit the five checks and the collection step; print their call ids."""
    if check_call_ids:
        call_ids = check_call_ids.split(",")
    else:
        print(f"PREDICTION: {PREDICTION_TEXT}", flush=True)
        calls = [check_one.spawn(spec.key) for spec in SPECS]
        call_ids = [call.object_id for call in calls]
        for spec, call_id in zip(SPECS, call_ids):
            print(f"function call id (check {spec.slug}): {call_id}", flush=True)
        for call in calls:
            result = call.get()
            print(f"[check] {result['key']} rows {result['rows_answered']} "
                  f"model {result['model_wall_s']:.0f}s "
                  f"total {result['total_wall_s']:.0f}s", flush=True)
    call = finish.spawn(call_ids)
    print(f"function call id (finish): {call.object_id}", flush=True)
    result = call.get()
    for sf, summary in result["collections"].items():
        print(sf, summary["collection_id"], json.dumps(summary["imdb16"]),
              flush=True)
    print(result["result_volume_path"], flush=True)
