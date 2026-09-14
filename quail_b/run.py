"""Run engine callbacks and save reproducible benchmark results."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quail_b._files import download_cache
from quail_b.benchmark import load_benchmark
from quail_b.minimum import token_metrics
from quail_b.scoring import (
    RunOutput,
    corpus_ids,
    encode_ids,
    evaluate,
    implied_row_count,
    implied_rows_mask,
    scores_from_answers,
)
from quail_b.substrait import _Filter

RUN_SCHEMA_VERSION = 2


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _query_hash(spec):
    plan = spec.plan
    urns = {
        extension.extension_urn_anchor: extension.urn
        for extension in plan.extension_urns
    }
    functions = []
    for declaration in plan.extensions:
        if not declaration.HasField("extension_function"):
            continue
        function = declaration.extension_function
        functions.append(
            (
                urns[function.extension_urn_reference],
                function.name,
            )
        )
    functions.sort()
    operators = []
    for operator in spec._info.operators:
        if isinstance(operator, _Filter):
            operators.append({
                "kind": "filter",
                "id": operator.id,
                "relation": operator.relation,
                "prompt": operator.prompt,
            })
        else:
            operators.append({
                "kind": "join",
                "id": operator.id,
                "relations": operator.relations,
                "prompt": operator.prompt,
                "on": operator.on,
            })
    definition = {
        "substrait_version": [
            plan.version.major_number,
            plan.version.minor_number,
            plan.version.patch_number,
        ],
        "functions": functions,
        "relations": [
            {
                "alias": relation.alias,
                "table": relation.table,
                "text_column": relation.text_column,
            }
            for relation in spec._info.relations
        ],
        "operators": operators,
        "select": spec._info.select,
    }
    encoded = json.dumps(
        definition,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_output(directory, output, spec):
    directory.mkdir()
    (directory / "plan.substrait").write_bytes(spec.plan_bytes)
    pq.write_table(output.rows, directory / "rows.parquet", compression="zstd")
    paths = {
        "plan": "plan.substrait",
        "rows": "rows.parquet",
        "filters": None,
        "joins": None,
    }
    for kind, answers in (
            ("filters", output.filter_answers), ("joins", output.join_answers)):
        if answers is None:
            continue
        paths[kind] = []
        for index, (key, table) in enumerate(answers.items()):
            name = f"{kind}-{index}.parquet"
            pq.write_table(table, directory / name, compression="zstd")
            paths[kind].append({"key": key, "path": name})
    if output.prompt_pieces is not None:
        _write_json(directory / "prompt_pieces.json", output.prompt_pieces)
        paths["prompt_pieces"] = "prompt_pieces.json"
    return paths


def _read_output(directory, record, rows=True):
    """Read a saved output; rows=False skips the rows table, which can be huge."""
    def file(name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("answer file is outside the query directory")
        return path

    paths = record["files"]
    filters = paths["filters"]
    joins = paths["joins"]
    pieces = paths.get("prompt_pieces")
    return RunOutput(
        None if filters is None else {
            item["key"]: pq.read_table(file(item["path"])) for item in filters},
        None if joins is None else {
            item["key"]: pq.read_table(file(item["path"])) for item in joins},
        pq.read_table(file(paths["rows"])) if rows else None,
        record.get("runtime_s"),
        record.get("measurements", {}),
        None if pieces is None else json.loads(file(pieces).read_text()))


ROW_SAMPLE = 100_000    # rows of a traced result checked one by one


def _sample_rows(table, count):
    """Up to count rows of a table, evenly spaced.

    Taken chunk by chunk: a take over the whole table concatenates each
    column first, and a string column of hundreds of millions of rows
    overflows Arrow's 32-bit string offsets.
    """
    if table.num_rows <= count:
        return table
    step = table.num_rows // count
    last = step * count
    pieces = []
    start = 0
    for batch in table.to_batches():
        end = start + batch.num_rows
        first = -(-start // step) * step
        indices = range(first, min(end, last), step)
        if indices:
            pieces.append(batch.take(pa.array(
                [index - start for index in indices], pa.int64())))
        start = end
    return pa.Table.from_batches(pieces, schema=table.schema)


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

    selected = [name.split(".")[0] for name in spec._info.select]
    answers = scores_from_answers(spec, output, tables)
    if output.rows is None:
        # a saved run rescored from its answers: its rows were checked
        # when they were saved
        if answers is None:
            raise ValueError("a run without answers must include its rows")
    elif set(output.rows.column_names) != set(selected):
        raise ValueError("output columns must match the query's selected aliases")
    elif answers is None:
        validate_ids(output.rows, selected)
    else:
        # a traced run's rows are implied by its answers: the count must
        # agree, and a sample of the rows must all be implied; the
        # answer tables below get the full checks
        survivors, relations = answers
        implied = implied_row_count(spec, survivors, relations)
        if output.rows.num_rows != implied:
            raise ValueError(
                f"the engine returned {output.rows.num_rows:,} rows but its "
                f"answers imply {implied:,}")
        sample = _sample_rows(output.rows, ROW_SAMPLE)
        validate_ids(sample, selected)
        mask = implied_rows_mask(
            pa.table({alias: sample[alias].cast(pa.string())
                      for alias in selected}),
            survivors, relations, spec)
        if not (pc.all(mask).as_py() if sample.num_rows else True):
            raise ValueError("a returned row is not implied by the answers")
    filters = {
        filter_spec.id: filter_spec for filter_spec in spec._info.filters
    }
    for operator_id, table in (output.filter_answers or {}).items():
        if operator_id not in filters:
            raise ValueError(f"unknown filter operator {operator_id!r}")
        alias = filters[operator_id].relation
        validate_ids(table, [alias])
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")
    joins = {join.id: join for join in spec._info.joins}
    for operator_id, table in (output.join_answers or {}).items():
        if operator_id not in joins:
            raise ValueError(f"unknown join operator {operator_id!r}")
        validate_ids(table, joins[operator_id].relations)
        if table["answer"].null_count or str(table["answer"].type) != "bool":
            raise ValueError("predicate answers must be non-null booleans")


def _score(spec, output, suite, gpu_count, gpu_hourly_rate_usd, tokens=None):
    """Score one query; `tokens` keeps document tokens across a run's queries."""
    _validate_output(spec, output, suite.tables)
    accuracy = evaluate(spec, output, suite.ground_truth, suite.tables)
    seconds = output.runtime_s
    inputs = {
        relation.alias: len(suite.tables[relation.table])
        for relation in spec._info.relations
    }
    metrics = {
        "runtime_s": seconds, "input_rows": inputs,
        "accuracy": accuracy, "cost_usd": None,
        **token_metrics(spec, output, suite.tables, tokens),
        "evaluated_document_pairs": None,
    }
    if gpu_hourly_rate_usd is not None:
        metrics["cost_usd"] = seconds / 3600 * gpu_count * gpu_hourly_rate_usd
    if spec._info.joins:
        pairs = output.measurements.get("evaluated_document_pairs")
        if output.join_answers is not None:
            if set(output.join_answers) == {
                join.id for join in spec._info.joins
            }:
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
        "schema_version": RUN_SCHEMA_VERSION, "quail_b_version": __version__,
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
    tokens = {}
    try:
        for spec, item in zip(suite.queries, record["queries"]):
            item["status"] = "running"
            _write_json(path, record)
            print(f"[quail-b] {spec.id}: {spec.description}", flush=True)
            tables = {
                relation.table: suite.tables[relation.table]
                for relation in spec._info.relations
            }
            try:
                output = run_query(spec, tables)
                if not isinstance(output, RunOutput):
                    raise TypeError("run_query must return a RunOutput")
                item.update(
                    files=_save_output(directory / spec.id, output, spec),
                    status="saved",
                )
                json.dumps({"runtime_s": output.runtime_s,
                            "measurements": output.measurements}, allow_nan=False)
                item.update(runtime_s=output.runtime_s,
                            measurements=output.measurements)
                _write_json(path, record)
                item["metrics"] = _score(
                    spec, output, suite, gpu_count, gpu_hourly_rate_usd, tokens)
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
