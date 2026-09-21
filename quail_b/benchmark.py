"""Load and validate benchmark inputs and reference labels."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quail_b.cuad import resolve_documents, verify_document_files
from quail_b.data import (
    CORPUS_COLUMNS,
    DATA_SEED,
    FILE_TABLES,
    GROUND_TRUTH_ROOT,
    PUBLISHED_CORPORA,
    SOURCE_REVISIONS,
    corpus_files_dir,
    corpus_identity,
    load_table,
)
from quail_b.labels import GroundTruthCollection, _read_json, load_ground_truth
from quail_b.predicates import PREDICATE_BY_KEY, predicate_payload
from quail_b.queries import QUERY_SUITES, QuerySpec, queries, suite_query_ids
from quail_b.rendering import PROMPT_FORMAT


def select_queries(only=None, *, scale_factor=0.1) -> tuple[QuerySpec, ...]:
    """Validate a scale factor and return the requested query definitions.

    Args:
        only: Query ids, or None for every query. A suite name from
            QUERY_SUITES ("QUAIL-B", "QUAIL-B-PDF") stands for the ids
            of that suite's queries.
        scale_factor: A published scale factor.
    """
    if scale_factor not in PUBLISHED_CORPORA:
        raise ValueError("scale factor must be 0.1, 0.5, or 1.0")
    available = queries()
    ids = list(available) if only is None else (
        [only] if isinstance(only, str) else list(only))
    ids = [query_id for item in ids for query_id in (
        suite_query_ids(item) if item in QUERY_SUITES else (item,))]
    unknown = set(ids) - available.keys()
    if unknown:
        raise ValueError(f"unknown query IDs: {sorted(unknown)}")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("query IDs must be nonempty and unique")
    if isinstance(only, (set, frozenset)):
        ids = [query_id for query_id in available if query_id in only]
    return tuple(available[query_id] for query_id in ids)


@dataclass
class Benchmark:
    """Selected query definitions, validated inputs, and reference labels."""

    queries: tuple[QuerySpec, ...]
    tables: dict[str, pa.Table]
    scale_factor: float
    corpus_id: str
    ground_truth: GroundTruthCollection | None


def load_benchmark(only=None, *, scale_factor=0.1, data_dir=None,
                   collection_id=None, accuracy=True, root=None) -> Benchmark:
    """Load selected inputs and labels from the public benchmark.

    Args:
        only: Query IDs, or None for all queries.
        scale_factor: Published scale factor: 0.1, 0.5, or 1.0.
        data_dir: Optional directory of input Parquet files to validate,
            with the `files/` directory of any file table beside them.
        collection_id: Reference collection ID, or None for the active one.
        accuracy: Whether to load reference labels for scoring.
        root: Published data root, or None for the public S3 bucket.

    Returns:
        The benchmark. A file table's `document` column names local
        files, absolute, with the page fragment kept.
    """
    specs = select_queries(only, scale_factor=scale_factor)
    corpus_id = PUBLISHED_CORPORA[scale_factor]
    manifest = _read_json(
        root, f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}/manifest.json")
    if manifest["corpus_id"] != corpus_id:
        raise ValueError("published corpus manifest has the wrong corpus ID")
    names = sorted({
        relation.table
        for spec in specs
        for relation in spec._info.relations
    })
    tables = {
        name: (
            load_table(name, scale_factor=scale_factor, root=root).select(
                CORPUS_COLUMNS[name])
            if data_dir is None else pq.read_table(
                Path(data_dir) / f"{name}.parquet", columns=list(CORPUS_COLUMNS[name])))
        for name in names
    }
    identity = corpus_identity(tables, scale_factor, DATA_SEED, SOURCE_REVISIONS)
    for name in names:
        if identity["tables"][name] != manifest["tables"][name]:
            raise ValueError(f"{name} does not match published corpus {corpus_id}")
    # a file table's published references are relative; the adapter
    # gets them pointed at local files whose content is checked
    for name in names:
        if name not in FILE_TABLES:
            continue
        directory = (Path(data_dir) if data_dir is not None else corpus_files_dir(
            name, tables[name], scale_factor=scale_factor, root=root))
        verify_document_files(tables[name], directory)
        tables[name] = resolve_documents(tables[name], directory)
    truth = None
    if accuracy:
        # only the label sets the selected queries score against
        truth = load_ground_truth(
            root, scale_factor=scale_factor, corpus_id=corpus_id,
            collection_id=collection_id, prompt_format=PROMPT_FORMAT, templates={
                operator.prompt for spec in specs
                for operator in spec._info.operators})
        if collection_id is not None and truth.collection_id != collection_id:
            raise ValueError("ground truth has the wrong collection ID")
        if truth.corpus_id != corpus_id or truth.scale_factor != scale_factor:
            raise ValueError("ground truth does not match the input corpus")
        for spec in specs:
            for operator in spec._info.operators:
                try:
                    truth.key_for_template(operator.prompt)
                except KeyError as error:
                    raise ValueError(
                        f"{spec.id} has no reference labels for {operator.id}; "
                        "provide a complete collection or set accuracy=False"
                    ) from error
        for key, labels in truth.predicates.items():
            predicate = PREDICATE_BY_KEY.get(key)
            if (predicate is not None and "qwen3_32b" in predicate.source_policy
                    and labels.predicate_payload != predicate_payload(predicate)):
                raise ValueError(
                    f"reference labels for {key} use a different prompt format; "
                    "regenerate labels for the current benchmark")
    return Benchmark(specs, tables, scale_factor, corpus_id, truth)
