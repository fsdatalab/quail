"""Read QUAIL-B labels and score one benchmark result."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
from pyarrow import parquet as pq


GROUND_TRUTH_ROOT = "ground_truth/quailb/schema_v1"
RESULTS_VOLUME = "quail-results"
H100_USD_PER_HOUR = 3.9492
H100_PRICE_SOURCE = "https://modal.com/pricing"

# Must match quail.bench.judge_pass.CORPUS_COLUMNS exactly - this is
# used to recompute corpus_id for verification.
CORPUS_COLUMNS = {
    "reviews": ("id", "body"),
    "aspects": ("id", "aspect"),
    "reports": ("id", "report", "reactions"),
    "terms": ("id", "term"),
    "claims": ("id", "claim", "label", "evidence_wiki_url"),
    "evidence": ("id", "text"),
    "citation_contexts": ("id", "destination_context",
                          "cited_passage_ids"),
    "citation_passages": ("id", "passage_text", "passage_ids"),
    "agent_traces": ("id", "trace", "trajectory_id", "turn_index",
                     "token_count"),
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _full_hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _python_rows(rows) -> list[dict]:
    return rows.to_pylist() if isinstance(rows, pa.Table) else rows


def _row_id(rows, index: int):
    if isinstance(rows, pa.Table):
        return rows.column("id")[index].as_py()
    return rows[index]["id"]


def _ids(rows):
    if isinstance(rows, pa.Table):
        return rows.column("id").to_pylist()
    return [row["id"] for row in rows]


def corpus_identity(rows: dict[str, pa.Table | list[dict]],
                    scale_factor: float,
                    data_seed: int, source_revisions: dict) -> dict:
    tables = {}
    for table in sorted(rows):
        row_hashes = [_full_hash(row) for row in _python_rows(rows[table])]
        tables[table] = {
            "rows": len(row_hashes),
            "ordered_rows_full_hash": _full_hash(row_hashes),
        }
    payload = {
        "schema_version": 1,
        "benchmark": "quailb",
        "scale_factor": scale_factor,
        "data_seed": data_seed,
        "source_revisions": source_revisions,
        "tables": tables,
    }
    full = _full_hash(payload)
    return {
        **payload,
        "corpus_id": f"c_{full[:32]}",
        "corpus_full_hash": full,
    }


def read_corpus(data_dir: str | Path) -> dict[str, pa.Table]:
    data_dir = Path(data_dir)
    return {
        table: pq.read_table(
            data_dir / f"{table}.parquet", columns=list(columns))
        for table, columns in CORPUS_COLUMNS.items()
    }


class LocalVolumeFiles:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read_bytes(self, path: str) -> bytes:
        return (self.root / path.lstrip("/")).read_bytes()

    def list_files(self, path: str) -> list[str]:
        base = self.root / path.lstrip("/")
        return sorted(
            item.relative_to(self.root).as_posix()
            for item in base.rglob("*") if item.is_file())

    def write_json(self, path: str, payload: dict) -> None:
        destination = self.root / path.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def write_parquet(self, path: str, table: pa.Table) -> None:
        destination = self.root / path.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination, compression="zstd")


class ModalVolumeFiles:
    def __init__(self, volume_name: str = RESULTS_VOLUME):
        import modal

        self.volume = modal.Volume.from_name(volume_name)

    def read_bytes(self, path: str) -> bytes:
        return b"".join(self.volume.read_file(path.lstrip("/")))

    def list_files(self, path: str) -> list[str]:
        return sorted(
            entry.path for entry in self.volume.listdir(
                path.lstrip("/"), recursive=True)
            if entry.path.endswith((".json", ".parquet")))

    def write_json(self, path: str, payload: dict) -> None:
        data = io.BytesIO(json.dumps(
            payload, indent=2, sort_keys=True).encode("utf-8"))
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(data, path.lstrip("/"))

    def write_parquet(self, path: str, table: pa.Table) -> None:
        data = io.BytesIO()
        pq.write_table(table, data, compression="zstd")
        data.seek(0)
        with self.volume.batch_upload(force=True) as batch:
            batch.put_file(data, path.lstrip("/"))


@dataclass(frozen=True)
class PredicateLabels:
    key: str
    label_set_id: str
    predicate: dict
    answers: dict[tuple[str, str | None], bool]
    source_rows: dict[str, int]

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
        from quail.logical import ColumnRef, bind_join_prompt, bind_prompt

        templates = {}
        for key, labels in self.predicates.items():
            predicate = labels.predicate
            left = ColumnRef(
                "left", predicate["left_table"], predicate["left_column"])
            if predicate["kind"] == "filter":
                template = bind_prompt(
                    predicate["template"], (left,)).template
            else:
                right = ColumnRef(
                    "right", predicate["right_table"],
                    predicate["right_column"])
                template = bind_join_prompt(
                    predicate["template"], (left, right)).template
            if template in templates:
                raise ValueError(
                    f"predicates {templates[template]} and {key} share a "
                    "prompt template")
            templates[template] = key
        object.__setattr__(self, "_template_keys", templates)

    def key_for_template(self, template: str) -> str:
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


def _read_json(files, path: str) -> dict:
    return json.loads(files.read_bytes(path))


def _read_many(files, paths: list[str]) -> dict[str, bytes]:
    workers = min(8, len(paths))
    if workers <= 1:
        return {path: files.read_bytes(path) for path in paths}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        contents = pool.map(files.read_bytes, paths)
        return dict(zip(paths, contents))


def _choose_collection(files, scale_factor: float,
                       corpus_id: str | None,
                       collection_id: str | None) -> tuple[str, dict]:
    if collection_id:
        path = (f"{GROUND_TRUTH_ROOT}/collections/{collection_id}"
                "/manifest.json")
        manifest = _read_json(files, path)
        if manifest.get("status") != "complete":
            raise ValueError(f"ground truth collection {collection_id} "
                             "is not complete")
        return path, manifest

    if corpus_id:
        active_path = (f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}"
                       "/active_collection.json")
        corpus_files = files.list_files(
            f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}")
        if active_path in corpus_files:
            active = _read_json(files, active_path)
            active_id = active["collection_id"]
            path = (f"{GROUND_TRUTH_ROOT}/collections/{active_id}"
                    "/manifest.json")
            manifest = _read_json(files, path)
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
        path for path in files.list_files(
            f"{GROUND_TRUTH_ROOT}/collections")
        if path.endswith("/manifest.json")
    ]
    matches = []
    for path in paths:
        manifest = _read_json(files, path)
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


def load_ground_truth(files, scale_factor: float = 0.1,
                      corpus_id: str | None = None,
                      collection_id: str | None = None
                      ) -> GroundTruthCollection:
    _path, collection = _choose_collection(
        files, scale_factor, corpus_id, collection_id)
    return _load_ground_truth_collection(files, collection)


def _predicate_tables(predicate: dict) -> tuple[str, ...]:
    tables = {predicate["left_table"]}
    if predicate["kind"] == "join":
        tables.add(predicate["right_table"])
    return tuple(sorted(tables))


def _validate_label_set_corpora(files, collection: dict,
                                manifests: dict[str, dict]) -> None:
    """Validate every label set against the tables its predicate reads."""
    target_corpus_id = collection["corpus_id"]
    reused = collection.get("reused_label_sets", {})
    corpus_manifests = {}
    source_collections = {}

    def corpus_manifest(corpus_id):
        if corpus_id not in corpus_manifests:
            path = f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}/manifest.json"
            corpus_manifests[corpus_id] = _read_json(files, path)
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
            source_collections[source_collection_id] = _read_json(files, path)
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


def _load_ground_truth_collection(files, collection: dict
                                  ) -> GroundTruthCollection:
    wanted = collection["label_sets"]
    all_paths = files.list_files(f"{GROUND_TRUTH_ROOT}/label_sets")
    manifest_paths = {}
    for key, label_set_id in sorted(wanted.items()):
        marker = f"/{label_set_id}/manifest.json"
        matches = [path for path in all_paths if path.endswith(marker)]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one manifest for {label_set_id}, found "
                f"{len(matches)}")
        manifest_paths[key] = matches[0]
    manifest_bytes = _read_many(files, list(manifest_paths.values()))
    manifests = {
        key: json.loads(manifest_bytes[path])
        for key, path in manifest_paths.items()
    }
    _validate_label_set_corpora(files, collection, manifests)
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
        files, [path for paths in data_paths.values() for path in paths])
    predicates = {}
    for key, label_set_id in sorted(wanted.items()):
        manifest = manifests[key]
        answers = {}
        for part_path in data_paths[key]:
            table = pq.read_table(
                io.BytesIO(data_bytes[part_path]),
                columns=["predicate_key", "label_set_id", "answer",
                         "left_id", "right_id"])
            for row in table.to_pylist():
                if row["predicate_key"] != key:
                    raise ValueError(
                        f"{part_path} contains {row['predicate_key']}, "
                        f"expected {key}")
                if row["label_set_id"] != label_set_id:
                    raise ValueError(
                        f"{part_path} has the wrong label set id")
                pair = (str(row["left_id"]),
                        None if row["right_id"] is None
                        else str(row["right_id"]))
                if pair in answers:
                    raise ValueError(
                        f"duplicate ground truth for {key} and {pair}")
                answers[pair] = bool(row["answer"])
        if len(answers) != manifest["rows"]:
            raise ValueError(
                f"{key} loaded {len(answers)} rows, expected "
                f"{manifest['rows']}")
        predicates[key] = PredicateLabels(
            key=key,
            label_set_id=label_set_id,
            predicate=manifest["predicate"],
            answers=answers,
            source_rows=manifest["source_rows"],
        )
    summary = collection.get("summary", {})
    return GroundTruthCollection(
        collection_id=collection["collection_id"],
        corpus_id=collection["corpus_id"],
        scale_factor=float(collection["scale_factor"]),
        reference_model=summary.get("model"),
        predicates=predicates,
    )


def load_ground_truth_workload(files, scale_factor: float, corpus_id: str,
                               corpus_full_hash: str, workload: str
                               ) -> GroundTruthCollection:
    """Load completed label sets for one benchmark workload."""
    from quail.bench.judge_pass import (
        MODEL_NAME,
        PREDICATES,
        label_set_identity,
    )

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
    return _load_ground_truth_collection(files, collection)


@dataclass
class BinaryCounts:
    correct: int = 0
    evaluated: int = 0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, predicted: bool, expected: bool) -> None:
        self.evaluated += 1
        self.correct += int(predicted == expected)
        if predicted and expected:
            self.true_positive += 1
        elif predicted:
            self.false_positive += 1
        elif expected:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def merge(self, other: "BinaryCounts") -> None:
        for name in ("correct", "evaluated", "true_positive",
                     "true_negative", "false_positive", "false_negative"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict:
        accuracy = self.correct / self.evaluated if self.evaluated else 0.0
        pden = self.true_positive + self.false_positive
        rden = self.true_positive + self.false_negative
        precision = self.true_positive / pden if pden else 0.0
        recall = self.true_positive / rden if rden else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        return {
            "evaluated": self.evaluated,
            "correct": self.correct,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass
class _PredicateCount:
    predicate_key: str
    op: str
    alias: str | None = None
    counts: BinaryCounts = field(default_factory=BinaryCounts)

    def as_dict(self) -> dict:
        return {
            "predicate_key": self.predicate_key,
            "op": self.op,
            "alias": self.alias,
            **self.counts.as_dict(),
        }


def _row_metrics(predicted_count: int, expected_count: int,
                 matched: int) -> dict:
    if not predicted_count and not expected_count:
        precision = recall = f1 = 1.0
    else:
        precision = matched / predicted_count if predicted_count else 0.0
        recall = matched / expected_count if expected_count else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
    return {
        "predicted_rows": predicted_count,
        "expected_rows": expected_count,
        "matching_rows": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact_match": (predicted_count == expected_count == matched),
        "false_positive_rows": predicted_count - matched,
        "false_negative_rows": expected_count - matched,
    }


class BenchmarkEvaluator:
    def __init__(self, ground_truth: GroundTruthCollection,
                 corpus_rows: dict[str, pa.Table | list[dict]]):
        self.ground_truth = ground_truth
        self.corpus_rows = corpus_rows

    def _key(self, prompt) -> str:
        return self.ground_truth.key_for_template(prompt.template)

    def _answer_for_prompt(self, prompt, assignment: dict[str, int]) -> bool:
        ids = [
            str(_row_id(
                self.corpus_rows[arg.provider], assignment[arg.alias]))
            for arg in prompt.args
        ]
        key = self._key(prompt)
        if len(ids) == 1:
            return self.ground_truth.answer(key, ids[0])
        if len(ids) == 2:
            return self.ground_truth.answer(key, ids[0], ids[1])
        raise NotImplementedError(
            "ground truth evaluation supports one or two prompt arguments")

    def _expected_answer_tables(self, query, scans, filters, joins):
        from quail.runtime.result import document_index_table

        survivors = {}
        for scan in scans:
            kept = []
            for index in range(len(self.corpus_rows[scan.provider])):
                assignment = {scan.alias: index}
                if all(self._answer_for_prompt(predicate.prompt, assignment)
                       for predicate in filters.get(scan.alias, ())):
                    kept.append(index)
            survivors[scan.alias] = pa.array(kept, type=pa.int32())

        id_indices = {
            scan.alias: {
                str(row_id): index
                for index, row_id in enumerate(
                    _ids(self.corpus_rows[scan.provider]))
            }
            for scan in scans
        }
        true_join_tables = {}
        for written_pos, join in enumerate(joins):
            if join.semantics != "full":
                raise NotImplementedError(
                    "QUAIL-B accuracy expects full join semantics")
            args = list(join.predicate.args)
            if len(args) != 2:
                raise NotImplementedError(
                    "QUAIL-B accuracy supports binary AI joins")
            key = self._key(join.predicate)
            labels = self.ground_truth.predicates[key]
            columns = {arg.alias: [] for arg in args}
            for (left_id, right_id), answer in labels.answers.items():
                if not answer:
                    continue
                columns[args[0].alias].append(
                    id_indices[args[0].alias][str(left_id)])
                columns[args[1].alias].append(
                    id_indices[args[1].alias][str(right_id)])
            true_join_tables[written_pos] = document_index_table(
                columns, "ground_truth_join_answers")
        return survivors, true_join_tables

    def _output_metrics(self, query, scans, filters, joins, result) -> dict:
        from quail.runtime.result import (
            build_result_declaration,
            count_rows,
            intersect_indices,
            intersect_tables,
        )

        if query.logical.root.limit is not None:
            raise NotImplementedError(
                "QUAIL-B output accuracy does not support LIMIT")
        expected_survivors, expected_join_tables = \
            self._expected_answer_tables(query, scans, filters, joins)
        join_order = [index for index, join in enumerate(joins)
                      if join.semantics == "full"]
        base_alias = query.logical.root.columns[0].alias
        expected_declaration, _ = build_result_declaration(
            [expected_join_tables[index] for index in join_order],
            expected_survivors,
            base_alias,
        )
        matching_survivors = {
            alias: intersect_indices(
                result.survivor_indices[alias], expected_survivors[alias])
            for alias in expected_survivors
        }
        matching_join_tables = {
            index: intersect_tables(
                result.true_join_tables[index],
                expected_join_tables[index])
            for index in join_order
        }
        matching_declaration, _ = build_result_declaration(
            [matching_join_tables[index] for index in join_order],
            matching_survivors,
            base_alias,
        )
        return _row_metrics(
            result.count(),
            count_rows(expected_declaration),
            count_rows(matching_declaration),
        )

    def evaluate(self, query, result) -> dict:
        from quail.planner.decide import _collect
        from quail.physical import PackedFilter

        scans, filters, joins = _collect(query.logical)
        plan = query.plan()
        providers = {scan.alias: scan.provider for scan in scans}
        per_predicate = []
        total = BinaryCounts()

        for node in plan.nodes:
            if not isinstance(node, PackedFilter):
                continue
            alias = node.alias
            for stage in node.stages:
                predicate = filters[alias][stage.written_pos]
                key = self._key(predicate.prompt)
                item = _PredicateCount(key, "filter", alias)
                table = result.answer_tables["filters"][
                    (alias, stage.written_pos)]
                indices = table.column(alias).to_pylist()
                answers = table.column("answer").to_pylist()
                for index, predicted in zip(indices, answers):
                    left_id = str(
                        _row_id(self.corpus_rows[providers[alias]], index))
                    expected = self.ground_truth.answer(key, left_id)
                    item.counts.add(bool(predicted), expected)
                total.merge(item.counts)
                per_predicate.append(item.as_dict())

        join_plan = plan.expected_join_stages()
        if len(join_plan) != len(result.answer_tables["joins"]):
            raise ValueError(
                "query plan and returned join stages have different lengths")
        for stage in join_plan:
            join = joins[stage.written_pos]
            key = self._key(join.predicate)
            item = _PredicateCount(key, "join")
            table = result.answer_tables["joins"][stage.written_pos]
            aliases = [arg.alias for arg in join.predicate.args]
            index_columns = {
                alias: table.column(alias).to_pylist()
                for alias in aliases
            }
            predictions = table.column("answer").to_pylist()
            for row_index, predicted in enumerate(predictions):
                assignment = {
                    alias: int(index_columns[alias][row_index])
                    for alias in aliases
                }
                expected = self._answer_for_prompt(
                    join.predicate, assignment)
                item.counts.add(bool(predicted), expected)
            total.merge(item.counts)
            per_predicate.append(item.as_dict())

        input_document_rows = sum(
            len(self.corpus_rows[scan.provider]) for scan in scans)
        unique_documents = {
            (scan.provider, str(row_id))
            for scan in scans
            for row_id in _ids(self.corpus_rows[scan.provider])
        }
        return {
            "ground_truth_collection_id": self.ground_truth.collection_id,
            "ground_truth_reference_model": self.ground_truth.reference_model,
            "answer_accuracy": total.as_dict(),
            "output_accuracy": self._output_metrics(
                query, scans, filters, joins, result),
            "per_predicate": per_predicate,
            "input_document_rows": input_document_rows,
            "unique_input_documents": len(unique_documents),
        }


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def add_query_metrics(row: dict, evaluation: dict,
                      h100_usd_per_hour: float, gpus: int) -> dict:
    wall_s = float(row["wall_s"])
    boot_s = float(row.get("boot_s") or 0.0)
    tokens = int(row["fresh_tokens"])
    gpu_rate = h100_usd_per_hour * gpus
    inference_cost = wall_s * gpu_rate / 3600
    cost_with_boot = (wall_s + boot_s) * gpu_rate / 3600
    input_rows = int(evaluation["input_document_rows"])
    calls = int(evaluation["answer_accuracy"]["evaluated"])
    row.update({
        "runtime_s": wall_s,
        "runtime_with_boot_s": round(wall_s + boot_s, 2),
        "tokens_processed": tokens,
        "tokens_per_second": round(_divide(tokens, wall_s), 2),
        "input_document_rows": input_rows,
        "unique_input_documents": evaluation["unique_input_documents"],
        "documents_per_second": round(_divide(input_rows, wall_s), 4),
        "inference_calls": calls,
        "inference_calls_per_second": round(_divide(calls, wall_s), 4),
        "inference_cost_usd": round(inference_cost, 8),
        "inference_cost_per_token_usd": round(
            _divide(inference_cost, tokens), 12),
        "inference_cost_per_million_tokens_usd": round(
            _divide(inference_cost, tokens) * 1_000_000, 6),
        "cost_with_boot_usd": round(cost_with_boot, 8),
        "cost_with_boot_per_million_tokens_usd": round(
            _divide(cost_with_boot, tokens) * 1_000_000, 6),
        "accuracy": {
            key: value for key, value in evaluation.items()
            if key not in ("input_document_rows", "unique_input_documents")
        },
    })
    return row


def summarize_queries(rows: list[dict], h100_usd_per_hour: float,
                      gpus: int) -> dict:
    good = [row for row in rows if "error" not in row]
    wall_s = sum(float(row["wall_s"]) for row in good)
    boot_s = sum(float(row.get("boot_s") or 0.0) for row in good)
    tokens = sum(int(row["tokens_processed"]) for row in good)
    input_rows = sum(int(row["input_document_rows"]) for row in good)
    calls = sum(int(row["inference_calls"]) for row in good)
    cost = wall_s * h100_usd_per_hour * gpus / 3600
    cost_with_boot = ((wall_s + boot_s) * h100_usd_per_hour * gpus
                      / 3600)
    counts = BinaryCounts()
    for row in good:
        accuracy = row["accuracy"]["answer_accuracy"]
        counts.merge(BinaryCounts(
            correct=accuracy["correct"],
            evaluated=accuracy["evaluated"],
            true_positive=accuracy["true_positive"],
            true_negative=accuracy["true_negative"],
            false_positive=accuracy["false_positive"],
            false_negative=accuracy["false_negative"],
        ))
    return {
        "queries_completed": len(good),
        "queries_failed": len(rows) - len(good),
        "query_runtime_s": round(wall_s, 2),
        "boot_s": round(boot_s, 2),
        "runtime_with_boot_s": round(wall_s + boot_s, 2),
        "tokens_processed": tokens,
        "tokens_per_second": round(_divide(tokens, wall_s), 2),
        "input_document_rows": input_rows,
        "documents_per_second": round(_divide(input_rows, wall_s), 4),
        "inference_calls": calls,
        "inference_cost_usd": round(cost, 8),
        "inference_cost_per_token_usd": round(
            _divide(cost, tokens), 12),
        "inference_cost_per_million_tokens_usd": round(
            _divide(cost, tokens) * 1_000_000, 6),
        "cost_with_boot_usd": round(cost_with_boot, 8),
        "cost_with_boot_per_million_tokens_usd": round(
            _divide(cost_with_boot, tokens) * 1_000_000, 6),
        "answer_accuracy": counts.as_dict(),
    }
