"""Where the region running on this process reports the rows it finishes.

Rust's ``FinishedRows`` builds a region's rows from the rows each
operator reports, each tagged with the node and port that wrote it, and
hands each region row on once every operator writing it has reported
it. The operators and the runner read it from module state: the region
runs on one thread, and the operators sit several calls below the code
that starts it, as with ``quail.progress.answer_sink``.

The region's hint, Rust's ``InputHint`` over the rows of the tables the
region reads, lives here too: an operator asks ``keeps`` before the
model reads a document.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence

import pyarrow as pa

_ROWS = None
_HELD = False
_HINT = None
# alias -> the table the hint holds rows of
_TABLES = {}
# table -> (the hint's generation, whether each row is kept)
_KEEP = {}


def finished_rows():
    """Return the FinishedRows the running region reports to, or None."""
    return None if _HELD else _ROWS


@contextlib.contextmanager
def reporting(rows, hint=None, tables: dict | None = None):
    """Report to rows, and skip the rows of tables by hint, while the block runs.

    Args:
        rows: The region's FinishedRows, or None to report nowhere.
        hint: Rust's ``InputHint`` over the rows of the tables the
            region reads, or None: ``keep(table)`` says whether each row
            of a table may still pass the tests above the region,
            ``tables()`` names the tables it holds rows of, and
            ``generation()`` changes whenever a test tightens.
        tables: Each scan alias's table name.
    """
    global _ROWS, _HINT, _TABLES, _KEEP
    previous = _ROWS, _HINT, _TABLES, _KEEP
    held = set() if hint is None else set(hint.tables())
    _ROWS, _HINT, _KEEP = rows, hint, {}
    _TABLES = {alias: table for alias, table in (tables or {}).items()
               if table in held}
    try:
        yield rows
    finally:
        _ROWS, _HINT, _TABLES, _KEEP = previous


@contextlib.contextmanager
def held():
    """Hold back operators' reports while the block runs.

    A node run over renumbered inputs, deduplicated or guarded, reports
    ids that are not the region's; its rows go once its whole result is
    expanded.
    """
    global _HELD
    previous, _HELD = _HELD, True
    try:
        yield
    finally:
        _HELD = previous


def report(tables: Iterable[pa.Table]) -> None:
    """Report finished rows, each table tagged by ``finished_table``."""
    rows = finished_rows()
    if rows is None:
        return
    for table in tables:
        rows.report(table)


def finished_table(node_id: str, port: str, table: pa.Table) -> pa.Table:
    """Tag a table of finished rows with the node and port that wrote them."""
    metadata = {**(table.schema.metadata or {}),
                b"quail.node": node_id.encode(), b"quail.port": port.encode()}
    return table.replace_schema_metadata(metadata)


def region_hint():
    """Return the running region's ``InputHint``, or None."""
    return _HINT


def keeps(alias, document: int) -> bool:
    """Whether the model should ask about a document of the alias.

    False when the hint, as it is now, rejects the document's row of
    its table; the hint is read again whenever it has tightened or the
    table has grown.
    """
    table = _TABLES.get(alias) if isinstance(alias, str) else None
    if table is None:
        return True
    generation = _HINT.generation()
    found = _KEEP.get(table)
    if found is None or found[0] != generation or document >= len(found[1]):
        found = _KEEP[table] = (generation, _HINT.keep(table))
    return document >= len(found[1]) or bool(found[1][document])


def hinting() -> bool:
    """Whether the region's hint holds the rows of a table the region reads."""
    return bool(_TABLES)


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
