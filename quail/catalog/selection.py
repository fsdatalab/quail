"""Select the rows of a scan that pass its column predicates."""

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.logical.predicates import accepted_rows


def selected_rows(columns: dict, predicates) -> pa.Array:
    """Return the positions of the rows that pass every predicate.

    Args:
        columns: Each predicate's column values, keyed by column name.
        predicates: The RegularPredicate tests, combined with AND.
    """
    mask = None
    for predicate in predicates:
        test = predicate.evaluate(columns[predicate.column.column])
        mask = test if mask is None else pc.and_kleene(mask, test)
    return pc.indices_nonzero(accepted_rows(mask))


def group_ids(columns) -> np.ndarray:
    """Number the rows so rows with equal keys share a number.

    Nulls form one group and NaNs another, like any other value.

    Args:
        columns: The key columns, Arrow arrays of one length.
    """
    codes = []
    for column in columns:
        values = pc.dictionary_encode(equality_keys(column, nan_equal=True))
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        codes.append(pc.fill_null(values.indices, -1).to_numpy())
    return np.unique(np.column_stack(codes), axis=0, return_inverse=True)[1].reshape(-1)


def equality_keys(values, *, nan_equal: bool):
    """Return values as keys that group and join by numeric equality.

    A signed zero becomes 0, and NaN either forms one group or becomes
    null, so a join or a membership test skips it. A dictionary column
    is decoded first.

    Args:
        values: An Arrow array or chunked array.
        nan_equal: Whether every NaN counts as one key.
    """
    if pa.types.is_dictionary(values.type):
        values = values.cast(values.type.value_type)
    if pa.types.is_floating(values.type):
        zero = pa.scalar(0, values.type)
        nan = pa.scalar(float("nan") if nan_equal else None, values.type)
        values = pc.if_else(pc.equal(values, zero), zero, values)
        values = pc.if_else(pc.is_nan(values), nan, values)
    return values


def document_positions(ids) -> np.ndarray:
    """Map document IDs to positions in a selection; absent IDs map to -1."""
    ids = np.asarray(ids, dtype=np.int64)
    positions = np.full(int(ids.max()) + 1 if len(ids) else 0, -1, dtype=np.int32)
    positions[ids] = np.arange(len(ids), dtype=np.int32)
    return positions
