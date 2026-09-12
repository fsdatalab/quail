"""Add prompt pieces to QUAIL-B runs saved without them and score their regret.

    uv run python -m quail.bench.restate <run_dir> --sf 0.1 [--data-dir DIR]

<run_dir> is a family run directory as the family runner writes it:
<method>/run.json listing every query beside <method>/<family>/run.json.
A method without its run.json (a family failed, so the runner never
merged) gets one built from the family files it has. A query saved
before Quail reported its prompt pieces gets its
prompt_pieces.json, built from the query and the anchor its run chose.
QUAIL-B then computes minimum_tokens and regret_tokens from the saved
answer tables, and every report and measurements table is rewritten.

Only the answer tables are read, not the output rows, which can hold
hundreds of millions of rows. `quail-b report` reads those too and
gives the same token numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import quail
import quail_b as benchmark
from quail.bench.quailb import build_query, prompt_pieces
from quail.bench.results import combine_measurements, write_json
from quail_b.minimum import regret_metrics
from quail_b.queries import get_query
from quail_b.scoring import RunOutput


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


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


def restate_query(session, tables, stores, directory: Path, record: dict):
    """Add the pieces of one saved query and its minimum and regret."""
    spec = get_query(record["id"])
    if "prompt_pieces" not in record["files"]:
        pieces = prompt_pieces(build_query(session, spec), _anchors(record))
        write_json(directory / "prompt_pieces.json", pieces)
        record["files"]["prompt_pieces"] = "prompt_pieces.json"
    output = RunOutput(
        _answer_files(directory, record, "filters"),
        _answer_files(directory, record, "joins"),
        pa.table({}), record["runtime_s"], record["measurements"],
        _load(directory / record["files"]["prompt_pieces"]))
    record["metrics"].update(regret_metrics(spec, output, tables, stores))
    for key in ("minimum_tokens", "regret_tokens"):
        record["measurements"].pop(key, None)


def merge_families(method_dir: Path) -> Path:
    """Write <method>/run.json from the family run.json files under it.

    The family runner writes it only when every family finished; a run
    with a failed family still has the other families' records, and
    every completed query keeps its answers. Records point at their
    family's query directory.
    """
    parts = sorted(method_dir.glob("*/run.json"))
    if not parts:
        raise FileNotFoundError(f"no <family>/run.json under {method_dir}")
    merged = None
    for path in parts:
        part = _load(path)
        if merged is None:
            merged = {key: value for key, value in part.items() if key != "queries"}
            merged["queries"] = []
        for record in part["queries"]:
            merged["queries"].append(
                dict(record, directory=f"{path.parent.name}/{record['id']}"))
    merged["status"] = ("complete" if all(
        record["status"] == "complete" for record in merged["queries"]) else "failed")
    write_json(method_dir / "run.json", merged)
    return method_dir / "run.json"


def restate_run(run_dir: Path, sf: float, data_dir: str | None) -> pa.Table:
    """Restate every run.json under a run directory and write the tables."""
    for method_dir in sorted(run_dir.iterdir()):
        if (method_dir.is_dir() and not (method_dir / "run.json").exists()
                and any(method_dir.glob("*/run.json"))):
            merge_families(method_dir)
    reports = sorted(run_dir.glob("*/run.json")) + sorted(run_dir.glob("*/*/run.json"))
    if not reports:
        raise FileNotFoundError(f"no <method>/run.json under {run_dir}")
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
    stores = {}
    with quail.Session(quail.EngineConfig()) as session:
        for name, table in tables.items():
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        for path in reports:
            report = _load(path)
            for record in report["queries"]:
                if record.get("status") != "complete":
                    continue
                directory = path.parent / record.get("directory", record["id"])
                restate_query(session, tables, stores, directory, record)
                metrics = record["metrics"]
                print(f"{path.parent.relative_to(run_dir)} {record['id']}: fresh "
                      f"{record['measurements']['fresh_tokens']:,}, minimum "
                      f"{metrics['minimum_tokens']:,}, regret "
                      f"{metrics['regret_tokens']:,}", flush=True)
            write_json(path, report)
            benchmark.report(path.parent, rescore=False)
    methods = sorted(path.parent.name for path in run_dir.glob("*/run.json"))
    return combine_measurements(run_dir, methods)


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
