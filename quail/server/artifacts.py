"""Save a finished QueryResult to disk and load it back.

A result directory holds ``result.arrow`` (the projected rows as an
Arrow IPC file), ``report.json``, and one Arrow IPC file per filter or
join answer table under ``answers/``. Every file is written under a
temporary name and renamed into place, so a reader never sees a
partial file.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from quail.execution.result import QueryResult

RESULT_FILE = "result.arrow"
REPORT_FILE = "report.json"
ANSWERS_FILE = "answers.jsonl"


class Answers:
    """Join answers saved one anchor per line, while the query runs.

    Each line is the dict the engine reports when an anchor finishes.
    The file is append-only and never renamed, so a reader can fetch
    the lines it has not seen yet by line number while more arrive.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.count = 0
        self._handle = None

    def append(self, payload: dict) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
        self._handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._handle.flush()
        self.count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read_answers(path: str | Path, after: int = 0,
                 limit: int | None = None) -> list[dict]:
    """Return the saved anchor answers from line ``after`` on."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.endswith("\n"):
                break       # the writer is still on this line
            if index < after:
                continue
            if limit is not None and len(out) >= limit:
                break
            out.append(json.loads(line))
    return out


def write_ipc_file(path: Path, reader_or_table) -> int:
    """Write Arrow batches to an IPC file and return the row count.

    The file appears under ``path`` only once it is complete.
    """
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    rows = 0
    if isinstance(reader_or_table, pa.Table):
        schema = reader_or_table.schema
        batches = reader_or_table.to_batches()
    else:
        schema = reader_or_table.schema
        batches = reader_or_table
    with pa.OSFile(str(temporary), "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
                rows += batch.num_rows
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    return rows


def write_result(result: QueryResult, directory: str | Path) -> dict:
    """Stream a result and its answer tables into ``directory``.

    Returns the manifest the saved record keeps: row count, column
    names, and the relative path of every file written.
    """
    directory = Path(directory)
    (directory / "answers" / "filters").mkdir(parents=True, exist_ok=True)
    (directory / "answers" / "joins").mkdir(parents=True, exist_ok=True)
    reader = result.execute_stream()
    try:
        rows = write_ipc_file(directory / RESULT_FILE, reader)
    finally:
        reader.close()
    filters = []
    for (alias, position), table in sorted(
            result.answer_tables.get("filters", {}).items()):
        name = f"answers/filters/{alias}--{position}.arrow"
        write_ipc_file(directory / name, table)
        filters.append({"alias": alias, "position": position, "file": name})
    joins = []
    for position, table in sorted(result.answer_tables.get("joins", {}).items()):
        name = f"answers/joins/{position}.arrow"
        write_ipc_file(directory / name, table)
        joins.append({"position": position, "file": name})
    report_path = directory / REPORT_FILE
    temporary = report_path.with_name(f"{REPORT_FILE}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result.report, handle, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
    return {
        "rows": rows,
        "columns": list(result.columns),
        "files": {
            "result": RESULT_FILE,
            "report": REPORT_FILE,
            "answers": {"filters": filters, "joins": joins},
        },
    }


def verify_result(directory: str | Path, manifest: dict) -> None:
    """Reopen every file in the manifest; raise if any is unreadable."""
    directory = Path(directory)
    files = manifest["files"]
    with ipc.open_file(str(directory / files["result"])) as reader:
        rows = sum(reader.get_batch(i).num_rows
                   for i in range(reader.num_record_batches))
    if rows != manifest["rows"]:
        raise ValueError(
            f"result file holds {rows} rows, manifest says {manifest['rows']}")
    json.loads((directory / files["report"]).read_text(encoding="utf-8"))
    for entry in files["answers"]["filters"] + files["answers"]["joins"]:
        with ipc.open_file(str(directory / entry["file"])):
            pass


def read_table(path: str | Path) -> pa.Table:
    with ipc.open_file(str(path)) as reader:
        return reader.read_all()


class _Chunks:
    """A write sink that hands back what was written since the last take."""

    closed = False

    def __init__(self):
        self.parts = []

    def write(self, data) -> int:
        self.parts.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        out = b"".join(self.parts)
        self.parts.clear()
        return out


def iter_ipc_stream(path: str | Path):
    """Yield the batches of a saved IPC file as one Arrow IPC stream.

    The stream format is sequential, so a reader can take each batch as
    it arrives instead of waiting for the whole file.
    """
    chunks = _Chunks()
    with ipc.open_file(str(path)) as reader:
        with ipc.new_stream(chunks, reader.schema) as writer:
            for index in range(reader.num_record_batches):
                writer.write_batch(reader.get_batch(index))
                yield chunks.take()
    yield chunks.take()


def result_from_parts(table: pa.Table, report: dict,
                      filters: dict[tuple[str, int], pa.Table],
                      joins: dict[int, pa.Table],
                      codecs=None) -> QueryResult:
    """Rebuild a QueryResult from saved parts.

    Args:
        table: The projected rows.
        report: The saved report dictionary.
        filters: Filter answer tables keyed by (alias, position).
        joins: Join answer tables keyed by position.
        codecs: Node codecs used to decode the executed plan, when the
            caller wants ``result.explain()`` to work.
    """
    result = QueryResult.from_table(table, report)
    result.answer_tables = {"filters": dict(filters), "joins": dict(joins)}
    if codecs is not None:
        result.attach_executed_plan(codecs)
    return result


def load_result(directory: str | Path, manifest: dict, codecs=None
                ) -> QueryResult:
    """Load a saved result directory into a QueryResult."""
    directory = Path(directory)
    files = manifest["files"]
    report = json.loads((directory / files["report"]).read_text("utf-8"))
    filters = {
        (entry["alias"], entry["position"]): read_table(directory / entry["file"])
        for entry in files["answers"]["filters"]
    }
    joins = {
        entry["position"]: read_table(directory / entry["file"])
        for entry in files["answers"]["joins"]
    }
    return result_from_parts(read_table(directory / files["result"]), report,
                             filters, joins, codecs)
