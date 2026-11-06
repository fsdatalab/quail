"""Evaluate Quail's expressions over Arrow columns.

The expression classes are Rust's, from ``_quail_native``: a value
expression is a tree of ColumnRef, GetField, ScalarFunction, Literal,
and Alias nodes, and a condition is a RegularPredicate, or a Not, And, or
Or of conditions. Rust's planner validates them; this module computes
them with Arrow compute.
"""

from __future__ import annotations

import functools

import pyarrow as pa
from _quail_native import (
    Alias,
    And,
    ColumnRef,
    GetField,
    Literal,
    ModelCall,
    Not,
    Or,
    Prompt,
    RegularPredicate,
    ScalarFunction,
)
from pyarrow import compute as pc

from quail.common.predicates import (
    COMPARISON_KERNELS,
    accepted_rows,
    compare_literal,
    membership,
)

__all__ = [
    "Alias", "And", "ColumnRef", "GetField", "Literal", "ModelCall", "Not", "Or",
    "Prompt", "RegularPredicate", "ScalarFunction",
]

# the SQL names of the scalar functions whose Arrow compute function has
# another name; every other scalar function of Arrow compute is called by
# its own name
FUNCTION_ALIASES = {"length": "utf8_length", "lower": "utf8_lower",
                    "upper": "utf8_upper", "trim": "utf8_trim_whitespace"}
DEFAULT_SELECTIVITY = 0.2
SCORE_COMPARISONS = ("<", "<=", ">", ">=")
# the text before each label of a classification prompt
LABEL_PREFIX = " "
PROBABILITIES_SUFFIX = "_probabilities"
# an "extract" call's second result column: the answer's character
# span in the document
SPAN_SUFFIX = "_span"
# the Arrow type of one feature list
FEATURE_LIST = pa.list_(
    pa.struct([("feature", pa.int32()), ("strength", pa.float32())]))


class CompileError(ValueError):
    """Raised when a query is malformed or outside the language."""


@functools.cache
def arrow_function(name: str):
    """Return the Arrow compute function a scalar function name calls.

    Args:
        name: A scalar function of Arrow compute, such as ``abs`` or
            ``utf8_reverse``, or a SQL name in FUNCTION_ALIASES.

    Raises:
        CompileError: Arrow compute has no such function, or it is an
            aggregate or vector function, which reads more than one row.
    """
    arrow = FUNCTION_ALIASES.get(name, name)
    try:
        kind = pc.get_function(arrow).kind
    except KeyError as error:
        raise CompileError(
            f"unknown function {name!r}; a scalar function of Arrow compute "
            f"is called by its name, such as abs or utf8_reverse") from error
    # cast is a meta function: it converts one row at a time
    if kind not in ("scalar", "meta") or not hasattr(pc, arrow) or (
            kind == "meta" and arrow != "cast"):
        raise CompileError(
            f"{name} is an Arrow {kind} function; an expression calls a "
            f"scalar function, which reads one row at a time")
    return getattr(pc, arrow)


def effective_selectivity(selectivity: float | None) -> float:
    """Return the selectivity the planner uses: the given one or the default."""
    return DEFAULT_SELECTIVITY if selectivity is None else selectivity


def evaluate_expression(expression, columns, *, qualified: bool = False):
    """Evaluate a value expression over the columns of one table.

    Args:
        expression: A column name, or a value expression.
        columns: The table, record batch, or scan input the columns come
            from, read with ``column(name)``.
        qualified: Whether result columns include their table alias.

    Raises:
        CompileError: A field does not exist, or a function does not
            take its arguments' types.
    """
    if isinstance(expression, str):
        return columns.column(expression)
    if isinstance(expression, Literal):
        # a function takes a constant as a Python value: Arrow's wrappers
        # read one in an option's position, such as round's ndigits
        if isinstance(expression.value, tuple):
            return pa.array(expression.value)
        return expression.value
    if isinstance(expression, ColumnRef):
        return columns.column(column_name(expression) if qualified
                              else expression.column)
    if isinstance(expression, Alias):
        return evaluate_expression(expression.expression, columns, qualified=qualified)
    if isinstance(expression, GetField):
        values = evaluate_expression(expression.input, columns, qualified=qualified)
        read = column_name(expression.input)
        if not pa.types.is_struct(values.type):
            raise CompileError(f"{read} has no fields; it is {values.type}")
        if values.type.get_field_index(expression.field) < 0:
            raise CompileError(
                f"{read} has no field {expression.field!r}; its fields are "
                f"{[item.name for item in values.type]}")
        return pc.struct_field(values, expression.field)
    arguments = [evaluate_expression(argument, columns, qualified=qualified)
                 for argument in expression.args]
    try:
        return arrow_function(expression.function)(*arguments)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError,
            TypeError) as error:
        raise CompileError(
            f"{column_name(expression)} cannot be evaluated on "
            f"{[str(values.type) for values in arguments]}: {error}") from error


def column_references(expression) -> tuple:
    """Return the source columns a value expression reads, in first use order."""
    if isinstance(expression, ColumnRef):
        return (expression,)
    if isinstance(expression, Alias):
        return column_references(expression.expression)
    if isinstance(expression, GetField):
        return column_references(expression.input)
    if isinstance(expression, ScalarFunction):
        return tuple(dict.fromkeys(
            ref for argument in expression.args for ref in column_references(argument)))
    return ()


def source_columns(expression) -> tuple[str, ...]:
    """Return the names of the source columns a document column reads."""
    if isinstance(expression, str):
        return (expression,)
    return tuple(dict.fromkeys(ref.column for ref in column_references(expression)))


def table_alias(expression) -> str:
    """Return the one table alias a value expression reads.

    Raises:
        CompileError: The expression reads two tables, or none.
    """
    aliases = tuple(dict.fromkeys(ref.alias for ref in column_references(expression)))
    if len(aliases) != 1:
        raise CompileError(
            f"{column_name(expression)} must read one table, not {list(aliases)}")
    return aliases[0]


def column_name(column, *, qualified: bool = True) -> str:
    """Return the name used for a source, computed, or projected column.

    Args:
        column: A column name, a value expression, or a projected or
            aggregated expression with a name.
        qualified: Whether a column reference carries its table alias.
    """
    if isinstance(column, str):
        return column
    if isinstance(column, ColumnRef):
        return (f"{column.alias}.{column.column}"
                if qualified and column.alias else column.column)
    if isinstance(column, GetField):
        return f"{column_name(column.input, qualified=qualified)}.{column.field}"
    if isinstance(column, ScalarFunction):
        arguments = ", ".join(column_name(argument, qualified=qualified)
                              for argument in column.args)
        return f"{column.function}({arguments})"
    if isinstance(column, Literal):
        return repr(column.value)
    return column.name


def condition_tests(condition) -> tuple:
    """Return the column tests under a condition, in written order."""
    if isinstance(condition, RegularPredicate):
        return (condition,)
    return tuple(test for operand in condition.operands
                 for test in condition_tests(operand))


def condition_columns(condition) -> tuple[str, ...]:
    """Return the names of the columns a condition's tests read, unqualified."""
    return tuple(dict.fromkeys(
        reference.column for test in condition_tests(condition)
        for reference in column_references(test.expression)))


def evaluate_condition(condition, columns, *, qualified: bool = False):
    """Return a boolean array of the rows a condition accepts.

    A null compared with a literal is neither accepted nor rejected and
    comes back null, and Arrow's Kleene kernels follow SQL's
    three-valued logic: ``NOT NULL`` is null and ``NULL OR TRUE`` is
    true. ``accepts`` treats a null as rejected.

    Args:
        condition: A RegularPredicate, or a Not, And, or Or of conditions.
        columns: The table, record batch, or rows holding the tested
            columns, read with ``column(name)``.
        qualified: Whether the columns are named with their table alias,
            as result rows name them.

    Raises:
        CompileError: A literal cannot be compared with its column's
            values, such as a string against a number column.
    """
    if isinstance(condition, Not):
        return pc.invert(evaluate_condition(condition.operand, columns,
                                            qualified=qualified))
    if isinstance(condition, (And, Or)):
        combine = pc.and_kleene if isinstance(condition, And) else pc.or_kleene
        return functools.reduce(combine, [
            evaluate_condition(operand, columns, qualified=qualified)
            for operand in condition.operands])
    values = evaluate_expression(condition.expression, columns, qualified=qualified)
    comparison, value = condition.comparison, condition.value
    try:
        if comparison == "is null":
            return pc.is_null(values)
        if comparison == "is not null":
            return pc.is_valid(values)
        if comparison == "is true":
            if not pa.types.is_boolean(values.type):
                raise TypeError(f"a test is true or false, got {values.type}")
            return values
        if comparison == "in":
            return membership(values, value)
        if isinstance(value, str):
            return COMPARISON_KERNELS[comparison](values, value)
        return compare_literal(values, comparison, value)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError,
            pa.ArrowTypeError, TypeError, ValueError, OverflowError) as error:
        raise CompileError(
            f"{column_name(condition.expression)} {comparison} {value!r} cannot "
            f"be evaluated on a column of type {values.type}: {error}") from error


def accepts(condition, columns, *, qualified: bool = False) -> pa.Array:
    """Return the rows a condition accepts, with every null rejected."""
    return accepted_rows(evaluate_condition(condition, columns, qualified=qualified))
