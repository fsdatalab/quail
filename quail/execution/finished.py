"""Where the region running on this process reports the rows it finishes.

Rust's ``FinishedRows`` builds a region's rows from the rows each
operator reports, each tagged with the node and port that wrote it, and
hands each region row on once every operator writing it has reported
it. The operators and the runner read it from module state: the region
runs on one thread, and the operators sit several calls below the code
that starts it, as with ``quail.progress.answer_sink``. The state is
per thread: a test runs each GPU's stages on a thread of its own.

The region's hint, Rust's ``InputHint`` over the rows of the tables the
region reads, lives here too: an operator asks ``keeps`` before the
model reads a document.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pyarrow as pa

from quail.common.pairs import FIRST_DOCUMENT
from quail.execution.types import output_table


class _Region(threading.local):
    rows = None
    held = False
    hint = None
    # alias -> the table the hint holds rows of
    tables = {}
    # alias -> (the hint's generation, whether each row is kept)
    keep = {}
    # alias -> each row's first row with the same document, for a table
    # the plan deduplicates
    first = {}


_REGION = _Region()


def finished_rows():
    """Return the FinishedRows the running region reports to, or None."""
    return None if _REGION.held else _REGION.rows


@contextlib.contextmanager
def reporting(rows, hint=None, tables: dict | None = None,
              first: dict | None = None):
    """Report to rows, and skip the rows of tables by hint, while the block runs.

    Args:
        rows: The region's FinishedRows, or None to report nowhere.
        hint: Rust's ``InputHint`` over the rows of the tables the
            region reads, or None: ``keep(table)`` says whether each row
            of a table may still pass the tests above the region,
            ``tables()`` names the tables it holds rows of, and
            ``generation()`` changes whenever a test tightens.
        tables: Each scan alias's table name.
        first: For each alias the plan deduplicates, each row's first
            row with the same document. One answer serves every copy,
            so a document is skipped only when the hint drops them all.
    """
    region = _REGION
    previous = (region.rows, region.hint, region.tables, region.keep,
                region.first)
    held = set() if hint is None else set(hint.tables())
    region.rows, region.hint, region.keep = rows, hint, {}
    region.first = first or {}
    region.tables = {alias: table for alias, table in (tables or {}).items()
                     if table in held}
    try:
        yield rows
    finally:
        (region.rows, region.hint, region.tables, region.keep,
         region.first) = previous


@contextlib.contextmanager
def held():
    """Hold back operators' reports while the block runs.

    A node run over renumbered inputs, deduplicated or guarded, reports
    ids that are not the region's; its rows go once its whole result is
    expanded.
    """
    previous, _REGION.held = _REGION.held, True
    try:
        yield
    finally:
        _REGION.held = previous


def report(tables: Iterable[pa.Table]) -> None:
    """Report finished rows, each table tagged by ``finished_table``."""
    rows = finished_rows()
    if rows is None:
        return
    for table in tables:
        rows.report(table)


def report_outputs(node, result) -> None:
    """Report each output of a finished node that the region's rows read.

    A node's outputs are final once it finishes; rows its operator
    reported before are skipped.
    """
    rows = finished_rows()
    if rows is None:
        return
    ports = {port.name: port for port in node.outputs}
    for name, value in result.outputs.items():
        if rows.wants(node.node_id, name):
            rows.report(finished_table(
                node.node_id, name, output_table(node, ports[name], value)))


def finished_table(node_id: str, port: str, table: pa.Table) -> pa.Table:
    """Tag a table of finished rows with the node and port that wrote them."""
    metadata = {**(table.schema.metadata or {}),
                b"quail.node": node_id.encode(), b"quail.port": port.encode()}
    return table.replace_schema_metadata(metadata)


def region_hint():
    """Return the running region's ``InputHint``, or None."""
    return _REGION.hint


def hinted_tables() -> dict | None:
    """Each scan alias's table name, for the tables the hint holds rows of.

    None when the running region has no hint.
    """
    return None if _REGION.hint is None else dict(_REGION.tables)


def first_rows(columns: dict, aliases) -> dict:
    """Each alias's first row with the same document, for each row.

    Args:
        columns: Each scan's value table, by alias.
        aliases: The aliases the plan deduplicates.
    """
    return {alias: columns[alias].column(FIRST_DOCUMENT).to_numpy()
            for alias in aliases if alias in columns}


def keeps(alias, document: int) -> bool:
    """Whether the model should ask about a document of the alias.

    False when the hint, as it is now, rejects the document's row of
    its table; the hint is read again whenever it has tightened.
    """
    region = _REGION
    table = region.tables.get(alias) if isinstance(alias, str) else None
    if table is None:
        return True
    generation = region.hint.generation()
    found = region.keep.get(alias)
    if found is None or found[0] != generation:
        keep = np.asarray(region.hint.keep(table), dtype=bool)
        first = region.first.get(alias)
        if first is not None:
            # a document is kept while the hint keeps one of its copies
            any_copy = np.zeros(len(keep), dtype=bool)
            np.logical_or.at(any_copy, first, keep)
            keep = any_copy[first]
        found = region.keep[alias] = (generation, keep)
    return bool(found[1][document])


def hinting() -> bool:
    """Whether the region's hint holds the rows of a table the region reads."""
    return bool(_REGION.tables)


def hinted(key) -> bool:
    """Whether the region's hint drops the document of an arena key.

    Args:
        key: An arena key, (alias, document) for a document.
    """
    return (isinstance(key, tuple) and len(key) == 2
            and not keeps(key[0], int(key[1])))


def hinted_out(keys: Sequence) -> Callable[[int], bool] | None:
    """Callable(anchor) -> whether the region's hint drops that document.

    Args:
        keys: Each anchor's arena key, (alias, document) for a document.

    Returns:
        None when the hint holds the rows of no table the region reads.
    """
    if not hinting():
        return None
    return lambda anchor: hinted(keys[anchor])
