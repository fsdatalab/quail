"""Evaluate column predicates over a provider's values."""

import pyarrow as pa
from pyarrow import compute as pc

from quail.logical import CompileError, RegularPredicate

_COMPARE = {
    "=": pc.equal, "<>": pc.not_equal,
    "<": pc.less, "<=": pc.less_equal,
    ">": pc.greater, ">=": pc.greater_equal,
}


def evaluate_predicate(values, predicate: RegularPredicate) -> pa.ChunkedArray:
    """Return a boolean array of the rows a column predicate accepts.

    A null compared with a literal is neither accepted nor rejected and
    comes back null; the caller treats it as rejected.

    Raises:
        CompileError: The literal cannot be compared with the column's
            values, such as a string against a number column.
    """
    try:
        if predicate.comparison == "is null":
            return pc.is_null(values)
        if predicate.comparison == "is not null":
            return pc.is_valid(values)
        if predicate.comparison == "in":
            return pc.is_in(values, value_set=pa.array(
                list(predicate.value), values.type))
        return _COMPARE[predicate.comparison](
            values, pa.scalar(predicate.value, values.type))
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
    mask = pc.fill_null(mask, False)
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    return pc.indices_nonzero(mask)
