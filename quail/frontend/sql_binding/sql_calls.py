"""Translate the SQL functions sqlglot parses to their own nodes into Arrow calls.

sqlglot parses many SQL functions and operators to a node of their own,
such as ``LOG10(x)`` to ``Log(10, x)`` or ``SUBSTRING(x, 1, 2)`` to
``Substring``. Each translation here returns the Arrow compute call
with SQL's meaning: ``SUBSTRING`` counts from 1, ``ROUND`` rounds half
away from zero, ``STRPOS`` returns 0 when the text is absent, and
``CASE`` takes the first true branch, and ``/`` of two integers keeps
the fraction. A node with no translation is
called by its name, as any scalar function of Arrow compute is.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from quail.logical import CompileError


@dataclass(frozen=True)
class Call:
    """An Arrow compute call over SQL nodes, constants, and other calls."""

    name: str
    args: tuple


@dataclass(frozen=True)
class Const:
    """A constant argument, such as an option in its Python position."""

    value: object


# CAST's SQL types, by sqlglot's type name, as the Arrow type they cast to
_CAST_TYPES = {
    "TINYINT": "int8", "SMALLINT": "int16", "INT": "int32", "BIGINT": "int64",
    "FLOAT": "float32", "DOUBLE": "float64", "DECIMAL": "float64",
    "TEXT": "string", "VARCHAR": "string", "CHAR": "string", "NVARCHAR": "string",
    "BOOLEAN": "bool", "DATE": "date32", "DATETIME": "timestamp[us]",
    "TIMESTAMP": "timestamp[us]",
}
# EXTRACT's parts, as the Arrow function that reads each
_EXTRACT_PARTS = {
    "YEAR": "year", "QUARTER": "quarter", "MONTH": "month", "DAY": "day",
    "HOUR": "hour", "MINUTE": "minute", "SECOND": "second",
    "DAYOFYEAR": "day_of_year", "ISOWEEK": "iso_week",
}
# DATE_TRUNC's units, as floor_temporal names them
_TRUNC_UNITS = ("year", "quarter", "month", "week", "day", "hour", "minute",
                "second")
# the nodes that call one Arrow function with the node's arguments in order
_RENAMED = {
    exp.Pow: "power", exp.Reverse: "utf8_reverse", exp.Coalesce: "coalesce",
    exp.Like: "match_like", exp.StartsWith: "starts_with",
    exp.EndsWith: "ends_with",
    exp.Replace: "replace_substring", exp.RegexpReplace: "replace_substring_regex",
    exp.Split: "split_pattern", exp.Initcap: "utf8_capitalize",
}


def _literal_int(node, what: str) -> int:
    if not (isinstance(node, exp.Literal) and not node.is_string
            and str(node.this).lstrip("-").isdigit()):
        raise CompileError(f"{what} is an integer literal, got {node.sql()}")
    return int(str(node.this))


def _arguments(node) -> tuple:
    """Return a node's arguments in the order sqlglot declares them."""
    return tuple(
        argument for key in node.arg_types
        for argument in (node.args[key] if isinstance(node.args.get(key), list)
                         else (node.args.get(key),))
        if isinstance(argument, exp.Expression))


def _substring(node) -> Call:
    start = _literal_int(node.args["start"], "SUBSTRING's start")
    if start < 1:
        raise CompileError("SUBSTRING's start counts from 1")
    length = node.args.get("length")
    if length is None:
        return Call("utf8_slice_codeunits", (node.this, Const(start - 1)))
    stop = start - 1 + _literal_int(length, "SUBSTRING's length")
    return Call("utf8_slice_codeunits", (node.this, Const(start - 1), Const(stop)))


def _log(node) -> Call:
    # sqlglot puts the base first in every dialect
    base, value = node.this, node.expression
    if value is None:
        return Call("ln", (base,))
    if isinstance(base, exp.Literal) and str(base.this) in ("10", "2"):
        return Call(f"log{base.this}", (value,))
    return Call("logb", (value, base))


def _case(node) -> Call:
    subject = node.this
    conditions, values = [], []
    for branch in node.args["ifs"]:
        condition = branch.this
        if subject is not None:
            condition = exp.EQ(this=subject.copy(), expression=condition)
        conditions.append(condition)
        values.append(branch.args["true"])
    default = node.args.get("default")
    if default is not None:
        values.append(default)
    return Call("case_when", (Call("make_struct", tuple(conditions)), *values))


def _cast(node) -> Call:
    kind = node.to.this.name
    if kind not in _CAST_TYPES:
        raise CompileError(
            f"CAST to {node.to.sql()} is not supported; cast to one of "
            f"{sorted(_CAST_TYPES)}")
    return Call("cast", (node.this, Const(_CAST_TYPES[kind])))


def _trim(node) -> Call:
    position = (node.args.get("position") or "").upper()
    characters = node.expression
    side = {"LEADING": "l", "TRAILING": "r"}.get(position, "")
    if characters is None:
        return Call(f"utf8_{side}trim_whitespace", (node.this,))
    return Call(f"utf8_{side}trim", (node.this, characters))


def _extract(node) -> Call:
    part = node.this.name.upper()
    if part not in _EXTRACT_PARTS:
        raise CompileError(
            f"EXTRACT({part} ...) is not supported; extract one of "
            f"{sorted(_EXTRACT_PARTS)}")
    return Call(_EXTRACT_PARTS[part], (node.expression,))


def _date_trunc(node) -> Call:
    unit = node.args["unit"]
    name = (unit.this if isinstance(unit, exp.Literal) else unit.name).lower()
    if name not in _TRUNC_UNITS:
        raise CompileError(f"DATE_TRUNC's unit is one of {list(_TRUNC_UNITS)}")
    value = node.this
    if isinstance(value, exp.Literal):
        raise CompileError(f"DATE_TRUNC truncates a column, got {value.sql()}")
    return Call("floor_temporal", (value, Const(1), Const(name)))


def _pad(node) -> Call:
    side = "lpad" if node.args.get("is_left") else "rpad"
    fill = node.args.get("fill_pattern")
    return Call(f"utf8_{side}", (node.this, node.expression,
                                 *(() if fill is None else (fill,))))


def _not(node) -> Call:
    inner = node.this.unnest() if isinstance(node.this, exp.Paren) else node.this
    if isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null):
        return Call("is_valid", (inner.this,))
    return Call("invert", (node.this,))


def _in(node) -> Call:
    if node.args.get("query") is not None or not node.expressions:
        raise CompileError(f"IN takes a list of literals, got {node.sql()}")
    found = Call("is_in", (node.this, Const(tuple(_constant(item)
                                                  for item in node.expressions))))
    # NULL IN (...) is NULL in SQL; Arrow's is_in says false
    return Call("if_else", (Call("is_null", (node.this,)), Const(None), found))


def _extreme(name: str):
    """Translate GREATEST or LEAST, which is NULL when any argument is."""
    def translate_extreme(node) -> Call:
        arguments = _arguments(node)
        nulls = [Call("is_null", (argument,)) for argument in arguments]
        any_null = nulls[0]
        for null in nulls[1:]:
            any_null = Call("or", (any_null, null))
        return Call("if_else", (any_null, Const(None), Call(name, arguments)))
    return translate_extreme


def _constant(node):
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        text = str(node.this)
        return float(text) if "." in text or "e" in text.lower() else int(text)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        return -_constant(node.this)
    raise CompileError(f"IN lists literals, got {node.sql()}")


def _is(node) -> Call:
    if not isinstance(node.expression, exp.Null):
        raise CompileError(f"IS tests NULL, got {node.sql()}")
    return Call("is_null", (node.this,))


_TRANSLATIONS = {
    exp.Substring: _substring,
    exp.Left: lambda node: Call("utf8_slice_codeunits", (
        node.this, Const(0), Const(_literal_int(node.expression, "LEFT's length")))),
    exp.Right: lambda node: Call("utf8_slice_codeunits", (
        node.this, Const(-_literal_int(node.expression, "RIGHT's length")))),
    exp.Log: _log,
    exp.Round: lambda node: Call("round", (
        node.this, node.args.get("decimals") or Const(0),
        Const("half_towards_infinity"))),
    exp.Concat: lambda node: Call("binary_join_element_wise",
                                  (*node.expressions, Const(""))),
    exp.DPipe: lambda node: Call("binary_join_element_wise",
                                 (node.this, node.expression, Const(""))),
    exp.Case: _case,
    exp.If: lambda node: Call("if_else", (
        node.this, node.args["true"], node.args.get("false") or Const(None))),
    exp.Nullif: lambda node: Call("if_else", (
        Call("equal", (node.this, node.expression)), Const(None), node.this)),
    exp.Cast: _cast,
    exp.Trim: _trim,
    exp.Extract: _extract,
    exp.DateTrunc: _date_trunc,
    exp.TimestampTrunc: _date_trunc,
    exp.StrPosition: lambda node: Call("add", (
        Call("find_substring", (node.this, node.args["substr"])), Const(1))),
    exp.Pad: _pad,
    exp.ILike: lambda node: Call("match_like", (
        node.this, node.expression, Const(True))),
    exp.Between: lambda node: Call("and_kleene", (
        Call("greater_equal", (node.this, node.args["low"])),
        Call("less_equal", (node.this, node.args["high"])))),
    exp.And: lambda node: Call("and_kleene", (node.this, node.expression)),
    exp.Or: lambda node: Call("or_kleene", (node.this, node.expression)),
    exp.Xor: lambda node: Call("xor", (node.this, node.expression)),
    exp.Not: _not,
    # SQL's / divides integers exactly; Arrow's truncates them
    exp.Div: lambda node: Call("divide", (
        Call("cast", (node.this, Const("float64"))), node.expression)),
    exp.Greatest: _extreme("max_element_wise"),
    exp.Least: _extreme("min_element_wise"),
    exp.In: _in,
    exp.Is: _is,
    **{kind: (lambda name: lambda node: Call(name, _arguments(node)))(name)
       for kind, name in _RENAMED.items()},
}
# the nodes translate() takes besides functions: operators and tests
TRANSLATED_NODES = (*_TRANSLATIONS, exp.RegexpLike, exp.TryCast, exp.Mod,
                    exp.IntDiv)


def translate(node, dialect: str = "bq") -> Call | None:
    """Return the Arrow call a SQL node makes, or None to call it by name.

    Args:
        node: The node.
        dialect: "snowflake" or "bq"; Snowflake's REGEXP_LIKE matches
            the whole value, BigQuery's REGEXP_CONTAINS any part.

    Raises:
        CompileError: The node's arguments cannot be passed to Arrow,
            such as a SUBSTRING whose start is a column.
    """
    if isinstance(node, exp.TryCast):
        raise CompileError(
            f"{node.sql()} returns NULL where the cast fails, which Arrow's "
            f"cast does not; use CAST")
    if isinstance(node, (exp.Mod, exp.IntDiv)):
        raise CompileError(
            f"{node.sql()}: Arrow compute has no remainder or integer "
            f"division function")
    if isinstance(node, exp.RegexpLike):
        pattern = node.expression
        if dialect == "snowflake":
            if not (isinstance(pattern, exp.Literal) and pattern.is_string):
                raise CompileError("REGEXP_LIKE's pattern is a string literal")
            pattern = Const(f"^(?:{pattern.this})$")
        return Call("match_substring_regex", (node.this, pattern))
    found = _TRANSLATIONS.get(type(node))
    return None if found is None else found(node)
