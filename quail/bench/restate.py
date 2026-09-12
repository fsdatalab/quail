"""Recompute the token minimum and regret of saved QUAIL-B runs.

    uv run python -m quail.bench.restate <run_dir> --sf 0.1 [--data-dir DIR]

<run_dir> is a family run directory as the family runner writes it:
<method>/<family>/run.json beside one directory per query holding its
answer tables. Every completed query gets minimum_tokens and
regret_tokens in its measurements, computed from its saved answer
tables on the CPU, and <run_dir>/measurements.parquet gets one row per
query and method. The corpus is tokenized once per document set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import quail
import quail_b as benchmark
from quail.bench.quailb import _document_ids, build_query
from quail.bench.results import measurement_rows, write_json, write_measurements
from quail.runtime.minimum import minimum_input_tokens
from quail_b.queries import get_query


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _rows_by_id(table) -> dict:
    return {str(row_id): index
            for index, row_id in enumerate(_document_ids(table))}


def _positions(table, aliases, rows_by_alias):
    columns = {
        alias: pa.array([rows_by_alias[alias][value]
                         for value in table.column(alias).to_pylist()],
                        type=pa.int32())
        for alias in aliases
    }
    return pa.table({**columns, "answer": table.column("answer")})


def _anchors(record) -> dict:
    """Written position -> anchor alias, from the saved stage order.

    The join answer files and the join stages are written in the same
    executed order, so they pair up by position.
    """
    stages = [stage for stage in record["measurements"]["stages"]
              if stage["op"] == "join"]
    files = record["files"]["joins"] or []
    if len(stages) != len(files):
        raise ValueError(
            f"{record['id']}: {len(files)} join answer files for "
            f"{len(stages)} join stages")
    return {item["key"]: stage["anchor"] for item, stage in zip(files, stages)}


def _answer_files(directory: Path, record: dict, kind: str) -> dict:
    """Key -> saved answer table of one kind, read without the output rows."""
    tables = {}
    for item in record["files"][kind] or []:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("answer file is outside the query directory")
        key = item["key"]
        tables[tuple(key) if kind == "filters" else key] = pq.read_table(path)
    return tables


def restate_query(session, tables, directory: Path, record: dict) -> dict:
    """Return the minimum and regret of one saved query."""
    spec = get_query(record["id"])
    query = build_query(session, spec)
    stores = query.token_inputs()
    rows_by_alias = {
        alias.alias: _rows_by_id(tables[alias.table]) for alias in spec.aliases
    }
    filters = {
        key: _positions(table, [key[0]], rows_by_alias)
        for key, table in _answer_files(directory, record, "filters").items()
    }
    joins = {
        written_pos: _positions(
            table, spec.joins[written_pos].aliases, rows_by_alias)
        for written_pos, table in _answer_files(directory, record, "joins").items()
    }
    minimum = minimum_input_tokens(
        query.logical, stores, filters, joins, anchors=_anchors(record))
    fresh = int(record["measurements"]["fresh_tokens"])
    return {"minimum_tokens": minimum, "regret_tokens": fresh - minimum}


def restate_run(run_dir: Path, sf: float, data_dir: str | None) -> pa.Table:
    """Restate every run.json under a run directory and write the table."""
    reports = sorted(run_dir.glob("*/*/run.json"))
    if not reports:
        raise FileNotFoundError(f"no <method>/<family>/run.json under {run_dir}")
    query_ids = sorted({
        item["id"] for path in reports
        for item in _load(path)["queries"]
    })
    names = sorted({
        alias.table for query_id in query_ids
        for alias in get_query(query_id).aliases
    })
    tables = {
        name: (benchmark.load_table(name, scale_factor=sf) if data_dir is None
               else pq.read_table(Path(data_dir) / f"{name}.parquet"))
        for name in names
    }
    rows = []
    with quail.Session(quail.EngineConfig()) as session:
        for name, table in tables.items():
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        for path in reports:
            method = path.parent.parent.name
            report = _load(path)
            for record in report["queries"]:
                if record.get("status") != "complete":
                    continue
                record["measurements"].update(
                    restate_query(session, tables, path.parent / record["id"],
                                  record))
                print(f"{method} {record['id']}: fresh "
                      f"{record['measurements']['fresh_tokens']:,}, minimum "
                      f"{record['measurements']['minimum_tokens']:,}, regret "
                      f"{record['measurements']['regret_tokens']:,}", flush=True)
            write_json(path, report)
            rows.extend(measurement_rows(method, report))
    return write_measurements(run_dir / "measurements.parquet", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--sf", type=float, default=0.1)
    parser.add_argument("--data-dir", help="directory of input Parquet files")
    args = parser.parse_args()
    table = restate_run(Path(args.run_dir), args.sf, args.data_dir)
    print(f"wrote {table.num_rows} rows to {args.run_dir}/measurements.parquet")


if __name__ == "__main__":
    main()
