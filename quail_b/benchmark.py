"""Load benchmark inputs and score engine results."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from quail_b.data import (
    CORPUS_COLUMNS,
    DATA_SEED,
    GROUND_TRUTH_ROOT,
    PUBLISHED_CORPORA,
    SOURCE_REVISIONS,
    corpus_identity,
    load_table,
)
from quail_b.labels import GroundTruthCollection, _read_json, load_ground_truth
from quail_b.queries import (
    SELECTIVITY_ESTIMATE_COLLECTION,
    SELECTIVITY_ESTIMATE_CORPUS,
    SELECTIVITY_ESTIMATE_SCALE_FACTOR,
    QuerySpec,
    queries,
)
from quail_b.scoring import Evaluator, RunOutput, add_query_metrics, summarize_queries


def select_queries(only=None, *, scale_factor=0.1) -> tuple[QuerySpec, ...]:
    """Validate a scale factor and return the requested query definitions."""
    if scale_factor not in PUBLISHED_CORPORA:
        raise ValueError("scale factor must be 0.1, 0.5, or 1.0")
    available = queries()
    ids = list(available) if only is None else (
        [only] if isinstance(only, str) else list(only))
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

    def score(self, query_id, output: RunOutput, measurements, *,
              h100_usd_per_hour, gpus=1):
        """Compute one query's accuracy, throughput, and GPU cost."""
        spec = next((spec for spec in self.queries if spec.id == query_id), None)
        if spec is None:
            raise ValueError(f"query {query_id!r} is not in this benchmark")
        row = dict(measurements, query=query_id, desc=spec.description,
                   rows=output.rows.num_rows)
        if self.ground_truth is not None:
            evaluation = Evaluator(self.ground_truth, self.tables).evaluate(
                spec, output)
            add_query_metrics(row, evaluation, h100_usd_per_hour, gpus)
        if spec.joins:
            pairs = sum(len(table) for table in output.join_answers.values())
            row["evaluated_document_pairs"] = pairs
            row["document_pairs_per_second"] = (
                round(pairs / row["wall_s"], 4) if row["wall_s"] else 0.0)
        return row

    def summarize(self, rows, *, model, backend, h100_usd_per_hour,
                  gpus=1, pass_wall_s):
        """Build the suite summary from scored query records."""
        expected = [spec.id for spec in self.queries]
        if [row["query"] for row in rows] != expected:
            raise ValueError("scored queries do not match the selected queries")
        passed = dict(queries=rows, pass_wall_s=round(pass_wall_s, 1))
        if self.ground_truth is not None:
            passed["summary"] = summarize_queries(rows, h100_usd_per_hour, gpus)
        return dict(
            sf=self.scale_factor, model=model, backend=backend, gpus=gpus,
            corpus_id=self.corpus_id,
            input_tables={name: len(table) for name, table in self.tables.items()},
            selectivity_estimates=dict(
                source_collection=SELECTIVITY_ESTIMATE_COLLECTION,
                source_corpus=SELECTIVITY_ESTIMATE_CORPUS,
                source_scale_factor=SELECTIVITY_ESTIMATE_SCALE_FACTOR),
            pricing=dict(h100_usd_per_hour=h100_usd_per_hour, gpu_count=gpus),
            ground_truth=(
                dict(collection_id=self.ground_truth.collection_id,
                     reference_model=self.ground_truth.reference_model)
                if self.ground_truth else None),
            passes={"single": passed})


def load_benchmark(only=None, *, scale_factor=0.1, data_dir=None,
                   collection_id=None, accuracy=True, root=None) -> Benchmark:
    """Load selected inputs and labels from the public benchmark.

    Args:
        only: Query IDs, or None for all queries.
        scale_factor: Published scale factor: 0.1, 0.5, or 1.0.
        data_dir: Optional directory of input Parquet files to validate.
        collection_id: Reference collection ID, or None for the active one.
        accuracy: Whether to load reference labels for scoring.
        root: Published data root, or None for the public S3 bucket.
    """
    specs = select_queries(only, scale_factor=scale_factor)
    corpus_id = PUBLISHED_CORPORA[scale_factor]
    manifest = _read_json(
        root, f"{GROUND_TRUTH_ROOT}/corpora/{corpus_id}/manifest.json")
    if manifest["corpus_id"] != corpus_id:
        raise ValueError("published corpus manifest has the wrong corpus ID")
    names = sorted({alias.table for spec in specs for alias in spec.aliases})
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
    truth = None
    if accuracy:
        truth = load_ground_truth(
            root, scale_factor=scale_factor, corpus_id=corpus_id,
            collection_id=collection_id)
        if truth.corpus_id != corpus_id or truth.scale_factor != scale_factor:
            raise ValueError("ground truth does not match the input corpus")
    return Benchmark(specs, tables, scale_factor, corpus_id, truth)
