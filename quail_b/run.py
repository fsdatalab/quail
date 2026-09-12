"""Run engine callbacks and save reproducible benchmark results."""

import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from quail_b._files import download_cache
from quail_b.benchmark import load_benchmark
from quail_b.scoring import RunOutput, corpus_ids, encode_ids, evaluate


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _query_hash(spec):
    return hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True).encode()).hexdigest()


def _save_output(directory, output):
    directory.mkdir()
    pq.write_table(output.rows, directory / "rows.parquet", compression="zstd")
    paths = {"rows": "rows.parquet", "filters": None, "joins": None}
    for kind, answers in (
            ("filters", output.filter_answers), ("joins", output.join_answers)):
        if answers is None:
            continue
        paths[kind] = []
        for index, (key, table) in enumerate(answers.items()):
            name = f"{kind}-{index}.parquet"
            pq.write_table(table, directory / name, compression="zstd")
            paths[kind].append({"key": key, "path": name})
    return paths


def _read_output(directory, record):
    def table(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("answer file is outside the query directory")
        return pq.read_table(path)

    paths = record["files"]
    filters = paths["filters"]
    joins = paths["joins"]
    return RunOutput(
        None if filters is None else {
            tuple(item["key"]): table(item["path"]) for item in filters},
        None if joins is None else {
            item["key"]: table(item["path"]) for item in joins},
        table(paths["rows"]), record.get("runtime_s"), record.get("measurements", {}))


def _validate_output(spec, output, tables):
    if (isinstance(output.runtime_s, bool)
            or not isinstance(output.runtime_s, (int, float))
            or not math.isfinite(output.runtime_s) or output.runtime_s < 0):
        raise ValueError("runtime_s must be a finite nonnegative number")
    references = corpus_ids(spec, tables)

    def validate_ids(table, aliases):
        # a join can return hundreds of millions of rows: one Arrow pass
        # maps each id to its corpus position, and the checks run on
        # those small integers
        if any(table[alias].null_count for alias in aliases):
            raise ValueError("document IDs cannot be null")
        codes = encode_ids(table, aliases, references)
        for alias in aliases:
            if codes[alias].null_count:
                raise ValueError(f"unknown document ID for alias {alias}")
        if codes.num_rows and (
                codes.group_by(aliases).aggregate([]).num_rows != codes.num_rows):
            raise ValueError("duplicate document IDs in an answer table")

    selected = [name.split(".")[0] for name in spec.select]
    if set(output.rows.column_names) != set(selected):
        raise ValueError("output columns must match the query's selected aliases")
    validate_ids(output.rows, selected)
    for (alias, position), table in (output.filter_answers or {}).items():
        if position < 0 or position >= len(spec.alias(alias).filters):
            raise ValueError("unknown filter position")
        validate_ids(table, [alias])
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")
    for position, table in (output.join_answers or {}).items():
        if position < 0 or position >= len(spec.joins):
            raise ValueError("unknown join position")
        validate_ids(table, spec.joins[position].aliases)
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")


def _score(spec, output, suite, gpu_count, gpu_hourly_rate_usd):
    _validate_output(spec, output, suite.tables)
    accuracy = evaluate(spec, output, suite.ground_truth, suite.tables)
    seconds = output.runtime_s
    inputs = {alias.alias: len(suite.tables[alias.table]) for alias in spec.aliases}
    metrics = {
        "runtime_s": seconds, "input_rows": inputs,
        "accuracy": accuracy, "cost_usd": None,
    }
    if gpu_hourly_rate_usd is not None:
        metrics["cost_usd"] = seconds / 3600 * gpu_count * gpu_hourly_rate_usd
    if spec.joins:
        pairs = output.measurements.get("evaluated_document_pairs")
        if output.join_answers is not None:
            if set(output.join_answers) == set(range(len(spec.joins))):
                pairs = sum(len(table) for table in output.join_answers.values())
        if pairs is not None and (
                isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 0):
            raise ValueError("evaluated_document_pairs must be a nonnegative integer")
        metrics["evaluated_document_pairs"] = pairs
        metrics["document_pairs_per_second"] = (
            pairs / seconds if pairs is not None and seconds else None)
    else:
        metrics["documents_per_second"] = sum(inputs.values()) / seconds if (
            seconds) else None
    return metrics


def run(run_query, *, queries=None, scale_factor=0.1, output_dir,
        metadata=None, gpu_count=1, gpu_hourly_rate_usd=None,
        collection_id=None, cache_dir=None, data_dir=None, root=None):
    """Run selected queries, save their answers, and generate a report.

    Args:
        run_query: Function(query, tables) returning a RunOutput.
        queries: Query IDs, or None for all published queries.
        scale_factor: Published scale factor: 0.1, 0.5, or 1.0.
        output_dir: New directory for this run.
        metadata: Engine, model, configuration, warmup, and cache settings.
        gpu_count: Number of GPUs executing each query.
        gpu_hourly_rate_usd: Price per GPU hour, or None to omit cost.
        collection_id: Label collection ID, or None to resolve the active one.
        cache_dir: Download cache directory, or None for the default.
        data_dir: Optional directory of input Parquet files to validate.
        root: Optional published-data mirror for offline use.

    Returns:
        The saved run record. Query failures raise after saving their status.
    """
    from quail_b import __version__
    from quail_b.reporting import _write_report

    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(f"run directory already exists: {directory}")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("gpu_count must be a positive integer")
    if gpu_hourly_rate_usd is not None and (
            isinstance(gpu_hourly_rate_usd, bool)
            or not math.isfinite(gpu_hourly_rate_usd) or gpu_hourly_rate_usd < 0):
        raise ValueError("GPU price must be finite and nonnegative")
    json.dumps(metadata or {}, allow_nan=False)
    with download_cache(cache_dir):
        suite = load_benchmark(
            queries, scale_factor=scale_factor, collection_id=collection_id,
            data_dir=data_dir, root=root)
    truth = suite.ground_truth
    record = {
        "schema_version": 1, "quail_b_version": __version__,
        "scale_factor": scale_factor, "corpus_id": suite.corpus_id,
        "collection_id": truth.collection_id, "reference_model": truth.reference_model,
        "metadata": metadata or {}, "gpu_count": gpu_count,
        "gpu_hourly_rate_usd": gpu_hourly_rate_usd,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "queries": [
            {"id": spec.id, "definition_hash": _query_hash(spec),
             "status": "pending"} for spec in suite.queries],
    }
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "run.json"
    _write_json(path, record)
    try:
        for spec, item in zip(suite.queries, record["queries"]):
            item["status"] = "running"
            _write_json(path, record)
            print(f"[quail-b] {spec.id}: {spec.description}", flush=True)
            tables = {alias.table: suite.tables[alias.table] for alias in spec.aliases}
            try:
                output = run_query(spec, tables)
                if not isinstance(output, RunOutput):
                    raise TypeError("run_query must return a RunOutput")
                item.update(
                    files=_save_output(directory / spec.id, output), status="saved")
                json.dumps({"runtime_s": output.runtime_s,
                            "measurements": output.measurements}, allow_nan=False)
                item.update(runtime_s=output.runtime_s,
                            measurements=output.measurements)
                _write_json(path, record)
                item["metrics"] = _score(
                    spec, output, suite, gpu_count, gpu_hourly_rate_usd)
                item["status"] = "complete"
            except Exception as error:
                item.update(
                    status="scoring_failed" if item["status"] == "saved" else "failed",
                    error=f"{type(error).__name__}: {error}")
                raise
            finally:
                _write_json(path, record)
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, record)
        _write_report(directory, record)
    return record
