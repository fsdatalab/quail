"""Evaluate column predicates over Arrow values.

One Arrow compute function evaluates each comparison of
quail.logical.COLUMN_COMPARISONS, over a source column at the scan
and over a label, score, answer, or span column in a Filter.
"""

from functools import partial

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc

from quail.logical import CompileError

_COMPARE = {
    "=": pc.equal, "<>": pc.not_equal,
    "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
    "like": pc.match_like,
    "ilike": partial(pc.match_like, ignore_case=True),
    "contains": pc.match_substring,
    "icontains": partial(pc.match_substring, ignore_case=True),
    "starts with": pc.starts_with,
    "ends with": pc.ends_with,
    "regex": pc.match_substring_regex,
}
_FUNCTIONS = {"length": pc.utf8_length}


def evaluate_predicate(values, predicate) -> pa.ChunkedArray:
    """Return a boolean array of the rows a column predicate accepts.

    A null compared with a literal is neither accepted nor rejected and
    comes back null; the caller treats it as rejected.

    Args:
        values: The column's values.
        predicate: A quail.logical.RegularPredicate or a
            quail.physical.Condition: its comparison, value, path, and
            function.

    Raises:
        CompileError: The literal cannot be compared with the column's
            values, such as a string against a number column.
    """
    try:
        for field in predicate.path:
            values = pc.struct_field(values, field)
        if predicate.function is not None:
            values = _FUNCTIONS[predicate.function](values)
        comparison = predicate.comparison.removeprefix("not ")
        if comparison == "is null":
            mask = pc.is_null(values)
        elif comparison == "is not null":
            mask = pc.is_valid(values)
        elif comparison == "in":
            mask = pc.is_in(values, value_set=pa.array(
                list(predicate.value), values.type))
        elif isinstance(predicate.value, str) and comparison in _COMPARE:
            mask = _COMPARE[comparison](values, predicate.value)
        else:
            mask = _COMPARE[comparison](
                values, pa.scalar(predicate.value, values.type))
        return pc.invert(mask) if comparison != predicate.comparison else mask
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
            pa.ArrowTypeError, TypeError, ValueError, OverflowError) as error:
        raise CompileError(
            f"{predicate} cannot be evaluated on a column of type "
            f"{values.type}: {error}") from error


def selected_rows(columns: dict, predicates) -> pa.Array:
    """Return the positions of the rows that pass every predicate.

    Args:
        columns: Each predicate's column values, keyed by column name.
        predicates: The RegularPredicate tests, combined with AND.
    """
    mask = None
    for predicate in predicates:
        test = evaluate_predicate(columns[predicate.column.column], predicate)
        mask = test if mask is None else pc.and_kleene(mask, test)
    return pc.indices_nonzero(accepted_rows(mask))


def accepted_rows(mask) -> pa.Array:
    """Return a predicate's mask with every null rejected, as one array."""
    mask = pc.fill_null(mask, False)
    return mask.combine_chunks() if isinstance(mask, pa.ChunkedArray) else mask


def document_positions(ids) -> np.ndarray:
    """Map document IDs to positions in a selection; absent IDs map to -1."""
    ids = np.asarray(ids, dtype=np.int64)
    positions = np.full(int(ids.max()) + 1 if len(ids) else 0, -1, dtype=np.int32)
    positions[ids] = np.arange(len(ids), dtype=np.int32)
    return positions
