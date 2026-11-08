"""Where the region running on this process reports the rows it finishes.

Rust's ``FinishedRows`` builds a region's rows from the rows each
operator reports, each tagged with the node and port that wrote it, and
hands each region row on once every operator writing it has reported
it. The operators and the runner read it from module state: the region
runs on one thread, and the operators sit several calls below the code
that starts it, as with ``quail.progress.answer_sink``.

The region's hint, Rust's ``InputHint`` over its one input, lives here
too: an operator asks ``keeps`` before the model reads a document.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable

import pyarrow as pa

_ROWS = None
_HELD = False
_HINT = None
_ALIAS = None
_GENERATION = None
_KEEP = None


def finished_rows():
    """Return the FinishedRows the running region reports to, or None."""
    return None if _HELD else _ROWS


@contextlib.contextmanager
def reporting(rows, hint=None, alias: str | None = None):
    """Report to rows, and skip input rows by hint, while the block runs.

    Args:
        rows: The region's FinishedRows, or None to report nowhere.
        hint: Rust's ``InputHint`` over the region's one input, or None:
            ``keep()`` says whether each input row may still pass the
            tests above the region, and ``generation()`` changes
            whenever a test tightens.
        alias: The alias of the scan that reads that input.
    """
    global _ROWS, _HINT, _ALIAS, _GENERATION, _KEEP
    previous = _ROWS, _HINT, _ALIAS, _GENERATION, _KEEP
    _ROWS, _HINT, _ALIAS, _GENERATION, _KEEP = rows, hint, alias, None, None
    try:
        yield rows
    finally:
        _ROWS, _HINT, _ALIAS, _GENERATION, _KEEP = previous


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


def keeps(alias: str, document: int) -> bool:
    """Whether the model should ask about a document of the alias.

    False when the hint, as it is now, rejects the document's input
    row; the hint is read again whenever it has tightened.
    """
    global _GENERATION, _KEEP
    if _HINT is None or alias != _ALIAS:
        return True
    generation = _HINT.generation()
    if generation != _GENERATION:
        _GENERATION, _KEEP = generation, _HINT.keep()
    return bool(_KEEP[document])


def hinted_alias() -> str | None:
    """The alias whose documents the hint covers, or None without a hint."""
    return None if _HINT is None else _ALIAS


def hinted_out(keys) -> Callable[[int], bool] | None:
    """Callable(anchor) -> whether the region's hint drops that document.

    Args:
        keys: Each anchor's arena key, (alias, document) for a document.

    Returns:
        None when no hint is installed.
    """
    if _HINT is None:
        return None

    def skip(anchor: int) -> bool:
        key = keys[anchor]
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        return not keeps(key[0], int(key[1]))

    return skip
