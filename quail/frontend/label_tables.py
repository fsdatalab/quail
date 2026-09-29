"""Registered label tables for AI.CLASSIFY categories."""

import pyarrow as pa

from quail.catalog import Catalog, ScanRequest
from quail.logical import CompileError


def read_label_table(catalog: Catalog, name: str) -> tuple[tuple, tuple]:
    """Return the labels and descriptions a registered table lists.

    The table has a non-null ``label`` column, an optional
    ``description`` column (null means none), and an optional unique
    integer ``ordinal`` column. Labels come in ascending ordinal order
    when that column exists, else in ascending UTF-8 order.

    Raises:
        CompileError: The table is not registered or lacks a label
            column, or its ordinals repeat.
    """
    provider = catalog.get(name)
    columns = provider.columns
    if "label" not in columns:
        raise CompileError(
            f"the label table {name!r} needs a label column; it has "
            f"{list(columns)}")
    wanted = [column for column in ("label", "description", "ordinal")
              if column in columns]
    table = pa.Table.from_batches(
        provider.scan(ScanRequest(columns=tuple(wanted))),
        schema=provider.schema.select(wanted) if hasattr(
            provider.schema, "select") else None)
    if "ordinal" in wanted:
        ordinals = table.column("ordinal").to_pylist()
        if len(set(ordinals)) != len(ordinals) or None in ordinals:
            raise CompileError(
                f"the label table {name!r} needs distinct ordinals")
        table = table.take(pa.array(sorted(
            range(len(ordinals)), key=lambda i: ordinals[i])))
    else:
        table = table.sort_by("label")
    labels = tuple(table.column("label").to_pylist())
    if any(label is None for label in labels):
        raise CompileError(f"the label table {name!r} has a null label")
    descriptions = ()
    if "description" in wanted:
        descriptions = tuple(
            "" if text is None else str(text)
            for text in table.column("description").to_pylist())
    return labels, descriptions
