"""Saved reference labels: one TRUE or FALSE per predicate and row.

A label set holds every answer of one predicate over one corpus. A
collection names one complete label set per predicate for one corpus
and is what a benchmark run scores against.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from functools import cached_property

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet as pq

from quail_b._files import _list_files, _read_bytes
from quail_b.data import GROUND_TRUTH_ROOT, _full_hash

LABEL_COLUMNS = ("left_id", "right_id", "answer")


def _answer_table(answers) -> pa.Table:
    """Return labels as a table of left_id, right_id, and answer.

    Args:
        answers: A table with those columns, or a dict of
            (left_id, right_id or None) -> answer.
    """
    if isinstance(answers, pa.Table):
        table = answers
    else:
        pairs = list(answers)
        table = pa.table({
            "left_id": pa.array([left for left, _ in pairs], pa.string()),
            "right_id": pa.array([right for _, right in pairs], pa.string()),
            "answer": pa.array(list(answers.values()), pa.bool_()),
        })
    if table.column("left_id").null_count or table.column("answer").null_count:
        raise ValueError("ground truth needs a left id and an answer per row")
    return pa.table({
        "left_id": pc.cast(table.column("left_id"), pa.string()),
        "right_id": pc.cast(table.column("right_id"), pa.string()),
        "answer": pc.cast(table.column("answer"), pa.bool_()),
    })


class PredicateLabels:
    """The saved answers of one predicate over one corpus.

    Attributes:
        key: The predicate key.
        label_set_id: The label set the answers came from.
        predicate: The predicate as its manifest records it.
        table: One row per labeled document or pair: `left_id`,
            `right_id` (null for a filter), and the boolean `answer`.
        source_rows: Rows per source the label set was built from.
    """

    def __init__(self, key: str, label_set_id: str, predicate: dict,
                 answers, source_rows: dict[str, int], predicate_payload=None):
        self.key = key
        self.label_set_id = label_set_id
        self.predicate = predicate
        self.table = _answer_table(answers)
        self.source_rows = source_rows
        self.predicate_payload = predicate_payload

    @cached_property
    def answers(self) -> dict[tuple[str, str | None], bool]:
        """(left_id, right_id or None) -> answer, for one lookup at a time.

        Scoring joins `table` instead; this dict is built on first use
        by callers that ask about one document or pair, such as the
        speed of light estimate.
        """
        return dict(zip(
            zip(self.table.column("left_id").to_pylist(),
                self.table.column("right_id").to_pylist()),
            self.table.column("answer").to_pylist()))

    @property
    def true_pairs(self) -> pa.Table:
        """The rows answered TRUE."""
        return self.table.filter(self.table.column("answer"))

    def answer(self, left_id: str, right_id: str | None = None) -> bool:
        pair = (str(left_id), None if right_id is None else str(right_id))
        try:
            return self.answers[pair]
        except KeyError as error:
            raise KeyError(
                f"no ground truth for {self.key} and row ids {pair}") \
                from error


@dataclass(frozen=True)
class GroundTruthCollection:
    collection_id: str
    corpus_id: str
    scale_factor: float
    reference_model: str | None
    predicates: dict[str, PredicateLabels]

    def __post_init__(self):
        templates = {}
        for key, labels in self.predicates.items():
            template = labels.predicate["template"]
            if template in templates:
                raise ValueError(
                    f"predicates {templates[template]} and {key} share a "
                    "prompt template")
            templates[template] = key
        object.__setattr__(self, "_template_keys", templates)

    def key_for_template(self, template: str) -> str:
        """Return the predicate key of one prompt template as written."""
        try:
            return self._template_keys[template]
        except KeyError as error:
            raise KeyError("the query predicate has no ground truth") from error

    def answer(self, predicate_key: str, left_id: str,
               right_id: str | None = None) -> bool:
        try:
            labels = self.predicates[predicate_key]
        except KeyError as error:
            raise KeyError(
                f"ground truth collection has no {predicate_key}") from error
        return labels.answer(left_id, right_id)


def _read_json(root, path: str) -> dict:
    return json.loads(_read_bytes(root, path))


def _read_many(root, paths: list[str]) -> dict[str, bytes]:
    workers = min(8, len(paths))
    if workers <= 1:
        return {path: _read_bytes(root, path) for path in paths}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_context().run, _read_bytes, root, path)
                   for path in paths]
        return {path: future.result() for path, future in zip(paths, futures)}


def _choose_collection(root, scale_factor: float,
                       corpus_id: str | None,
                       collection_id: str | None,
                       prompt_format: str | None = None) -> tuple[str, dict]:
    if collection_id:
        path = (f"{GROUND_TRUTH_ROOT}/collections/{collection_id}"
                "/manifest.json")
        manifest = _read_json(root, path)
        if manifest.get("status") != "complete":
            raise ValueError(f"ground truth collection {collection_id} "
                             "is not complete")
        return path, manifest

    if corpus_id:
        active_path = (f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}"
                       "/active_collection.json")
        corpus_files = _list_files(root,
            f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}")
        if prompt_format:
            format_path = (f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}/"
                           f"active_collection.{prompt_format}.json")
            if format_path in corpus_files:
                active_path = format_path
        if active_path in corpus_files:
            active = _read_json(root, active_path)
            active_id = active["collection_id"]
            path = (f"{GROUND_TRUTH_ROOT}/collections/{active_id}"
                    "/manifest.json")
            manifest = _read_json(root, path)
            if manifest.get("status") != "complete":
                raise ValueError(
                    f"active ground truth collection {active_id} is not "
                    "complete")
            if manifest.get("corpus_id") != corpus_id:
                raise ValueError(
                    f"active ground truth collection {active_id} belongs "
                    "to another corpus")
            if float(manifest.get("scale_factor")) != float(scale_factor):
                raise ValueError(
                    f"active ground truth collection {active_id} has scale "
                    f"factor {manifest.get('scale_factor')}")
            return path, manifest

    paths = [
        path for path in _list_files(root,
            f"{GROUND_TRUTH_ROOT}/collections")
        if path.endswith("/manifest.json")
    ]
    matches = []
    for path in paths:
        manifest = _read_json(root, path)
        if manifest.get("status") != "complete":
            continue
        if float(manifest.get("scale_factor")) != float(scale_factor):
            continue
        if corpus_id and manifest.get("corpus_id") != corpus_id:
            continue
        matches.append((path, manifest))
    if not matches:
        detail = f" for corpus {corpus_id}" if corpus_id else ""
        raise FileNotFoundError(
            f"no complete sf={scale_factor} ground truth collection{detail}")
    if len(matches) > 1:
        ids = [manifest["collection_id"] for _, manifest in matches]
        raise ValueError(
            "more than one ground truth collection matches; pass one "
            f"collection id from {ids}")
    return matches[0]


def load_ground_truth(root=None, scale_factor: float = 0.1,
                      corpus_id: str | None = None,
                      collection_id: str | None = None,
                      templates=None, *,
                      prompt_format: str | None = None) -> GroundTruthCollection:
    """Load labels from the public bucket or a local root directory.

    Args:
        root: Local directory or S3 URI, or None for the public bucket.
        scale_factor: The corpus scale factor.
        corpus_id: The corpus, or None for any at that scale factor.
        collection_id: The collection, or None for the active one.
        templates: Prompt templates whose label sets to load, or None
            for every label set of the collection.
        prompt_format: Prefer the active collection for this prompt format.
    """
    _path, collection = _choose_collection(
        root, scale_factor, corpus_id, collection_id, prompt_format)
    return _load_ground_truth_collection(root, collection, templates)


def _predicate_tables(predicate: dict) -> tuple[str, ...]:
    tables = {predicate["left_table"]}
    if predicate["kind"] == "join":
        tables.add(predicate["right_table"])
    return tuple(sorted(tables))


def _validate_label_set_corpora(root, collection: dict,
                                manifests: dict[str, dict]) -> None:
    """Validate every label set against the tables its predicate reads."""
    target_corpus_id = collection["corpus_id"]
    reused = collection.get("reused_label_sets", {})
    corpus_manifests = {}
    source_collections = {}

    def corpus_manifest(corpus_id):
        if corpus_id not in corpus_manifests:
            path = f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}/manifest.json"
            corpus_manifests[corpus_id] = _read_json(root, path)
        return corpus_manifests[corpus_id]

    target = None
    for key, manifest in manifests.items():
        source_corpus_id = manifest.get("corpus_id")
        if not source_corpus_id:
            continue
        if source_corpus_id == target_corpus_id:
            target = target or corpus_manifest(target_corpus_id)
            if (manifest.get("corpus_full_hash")
                    and manifest["corpus_full_hash"]
                    != target["corpus_full_hash"]):
                raise ValueError(
                    f"label set {manifest['label_set_id']} has the wrong "
                    "corpus hash")
            continue

        record = reused.get(key)
        if record is None:
            raise ValueError(
                f"label set {manifest['label_set_id']} belongs to corpus "
                f"{source_corpus_id}, not {target_corpus_id}")
        if record.get("source_corpus_id") != source_corpus_id:
            raise ValueError(f"reused label set {key} has the wrong source")

        source_collection_id = record.get("source_collection_id")
        if not source_collection_id:
            raise ValueError(
                f"reused label set {key} has no source collection")
        if source_collection_id not in source_collections:
            path = (f"{GROUND_TRUTH_ROOT}/collections/"
                    f"{source_collection_id}/manifest.json")
            source_collections[source_collection_id] = _read_json(root, path)
        source_collection = source_collections[source_collection_id]
        if (source_collection.get("status") != "complete"
                or source_collection.get("collection_id")
                != source_collection_id
                or source_collection.get("label_sets", {}).get(key)
                != manifest["label_set_id"]):
            raise ValueError(
                f"source collection {source_collection_id} does not "
                f"contain reused label set {key}")

        required = _predicate_tables(manifest["predicate"])
        if tuple(record.get("required_tables", ())) != required:
            raise ValueError(
                f"reused label set {key} lists the wrong required tables")
        target = target or corpus_manifest(target_corpus_id)
        source = corpus_manifest(source_corpus_id)
        if (manifest.get("corpus_full_hash")
                and manifest["corpus_full_hash"]
                != source["corpus_full_hash"]):
            raise ValueError(
                f"reused label set {key} has the wrong source corpus hash")
        for table in required:
            recorded = record.get("verified_table_manifests", {}).get(table)
            if recorded != source["tables"].get(table):
                raise ValueError(
                    f"reused label set {key} has the wrong saved manifest "
                    f"for table {table}")
            if source["tables"].get(table) != target["tables"].get(table):
                raise ValueError(
                    f"cannot reuse {key}: table {table} changed between "
                    f"{source_corpus_id} and {target_corpus_id}")


def _read_label_set(parts: list[bytes], key: str, label_set_id: str,
                    rows: int) -> pa.Table:
    """Read one label set's Parquet parts and check them as columns."""
    table = pa.concat_tables([
        pq.read_table(io.BytesIO(part), columns=[
            "predicate_key", "label_set_id", *LABEL_COLUMNS])
        for part in parts
    ])
    for column, expected in (("predicate_key", key),
                             ("label_set_id", label_set_id)):
        mismatch = pc.fill_null(pc.not_equal(table.column(column), expected), True)
        if pc.any(mismatch).as_py():
            raise ValueError(
                f"a label file of {key} has the wrong {column}")
    table = _answer_table(table.select(list(LABEL_COLUMNS)))
    distinct = table.group_by(["left_id", "right_id"]).aggregate([]).num_rows
    if distinct != table.num_rows:
        raise ValueError(f"duplicate ground truth for {key}")
    if table.num_rows != rows:
        raise ValueError(
            f"{key} loaded {table.num_rows} rows, expected {rows}")
    return table


def _load_ground_truth_collection(root, collection: dict, templates=None
                                  ) -> GroundTruthCollection:
    """Load a collection's label sets, or only those of some templates."""
    wanted = collection["label_sets"]
    all_paths = _list_files(root, f"{GROUND_TRUTH_ROOT}/label_sets")
    manifest_paths = {}
    for key, label_set_id in sorted(wanted.items()):
        marker = f"/{label_set_id}/manifest.json"
        matches = [path for path in all_paths if path.endswith(marker)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one manifest for {label_set_id}, found "
                f"{len(matches)}")
        manifest_paths[key] = matches[0]
    manifest_bytes = _read_many(root, list(manifest_paths.values()))
    manifests = {
        key: json.loads(manifest_bytes[path])
        for key, path in manifest_paths.items()
    }
    _validate_label_set_corpora(root, collection, manifests)
    if templates is not None:
        wanted = {
            key: label_set_id for key, label_set_id in wanted.items()
            if manifests[key]["predicate"]["template"] in templates
        }
    data_paths = {}
    for key, label_set_id in sorted(wanted.items()):
        manifest_path = manifest_paths[key]
        manifest = manifests[key]
        if manifest.get("status") != "complete":
            raise ValueError(f"label set {label_set_id} is not complete")
        label_dir = manifest_path.removesuffix("/manifest.json")
        compact = f"{label_dir}/labels.parquet"
        if compact in all_paths:
            part_paths = [compact]
        else:
            part_paths = sorted(
                path for path in all_paths
                if path.startswith(f"{label_dir}/parts/")
                and path.endswith(".parquet"))
        if not part_paths:
            raise FileNotFoundError(f"label set {label_set_id} has no rows")
        data_paths[key] = part_paths
    data_bytes = _read_many(
        root, [path for paths in data_paths.values() for path in paths])
    predicates = {}
    for key, label_set_id in sorted(wanted.items()):
        manifest = manifests[key]
        predicates[key] = PredicateLabels(
            key=key,
            label_set_id=label_set_id,
            predicate=manifest["predicate"],
            answers=_read_label_set(
                [data_bytes[path] for path in data_paths[key]],
                key, label_set_id, manifest["rows"]),
            source_rows=manifest["source_rows"],
            predicate_payload=manifest.get("predicate_payload"),
        )
    summary = collection.get("summary", {})
    return GroundTruthCollection(
        collection_id=collection["collection_id"],
        corpus_id=collection["corpus_id"],
        scale_factor=float(collection["scale_factor"]),
        reference_model=summary.get("model"),
        predicates=predicates,
    )


def load_ground_truth_workload(root=None, *, scale_factor: float, corpus_id: str,
                               corpus_full_hash: str, workload: str
                               ) -> GroundTruthCollection:
    """Load completed label sets for one benchmark workload."""
    from quail_b.predicates import MODEL_NAME, PREDICATES, label_set_identity

    specs = [spec for spec in PREDICATES if spec.workload == workload]
    if not specs:
        raise ValueError(f"unknown ground truth workload {workload!r}")
    identities = {
        spec.key: label_set_identity(spec, corpus_id, corpus_full_hash)
        for spec in specs
    }
    label_sets = {
        key: identity["label_set_id"]
        for key, identity in identities.items()
    }
    payload = {
        "schema_version": 1,
        "benchmark": "quailb",
        "scale_factor": scale_factor,
        "corpus_id": corpus_id,
        "workload": workload,
        "label_sets": label_sets,
    }
    collection = {
        **payload,
        "collection_id": f"gtw_{_full_hash(payload)[:32]}",
        "summary": {"model": MODEL_NAME},
    }
    return _load_ground_truth_collection(root, collection)
