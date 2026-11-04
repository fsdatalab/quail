"""Parse the plain tests of WHERE and ON: tests that ask no model.

A plain test compares a value expression with literals, or is a
boolean expression on its own, such as ``UTF8_IS_UPPER(r.lang)`` or
``r.stars > r.min_stars``.

A plain test reads a source column, applied at the scan, or a column
a column operator computes, named by its AS alias as DuckDB allows
(``WHERE lease.end_date IS NOT NULL``) or by repeating the call.
Either way the model runs once.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from quail.common.expressions import (
    SPAN_SUFFIX,
    column_expression,
    column_references,
)
from quail.frontend.sql_binding.sql_calls import TRANSLATED_NODES
from quail.logical import And, ColumnRef, CompileError, Not, Or, RegularPredicate

_PLAIN_COMPARISONS = {
    exp.EQ: "=",
    exp.NEQ: "<>",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
}

_FLIPPED = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "<>": "<>"}

# text tests sqlglot parses to one node in both dialects; the
# function's case rule is the dialect's
_TEXT_TESTS = {
    exp.Like: "like",
    exp.ILike: "ilike",
    exp.StartsWith: "starts with",
    exp.EndsWith: "ends with",
    exp.RegexpLike: "regex",
}
_SUPPORTED = (
    "=, <>, <, <=, >, >=, IN (literals), IS NULL, IS NOT NULL, LIKE, "
    "ILIKE, CONTAINS, STARTSWITH, ENDSWITH, REGEXP_LIKE, or a boolean "
    "column or scalar function of columns")


def sql_literal(node):
    """Return a Python value for a SQL literal, or raise."""
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        text = str(node.this)
        return float(text) if "." in text or "e" in text.lower() else int(text)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if (
        isinstance(node, exp.Neg)
        and isinstance(node.this, exp.Literal)
        and not node.this.is_string
    ):
        return -sql_literal(node.this)
    raise CompileError(
        f"a column is compared with a number, string, or boolean "
        f"literal, got {node.sql()}"
    )


@dataclass(frozen=True)
class PlainTest:
    """One parsed plain test, before its operand is resolved.

    Attributes:
        operand: The tested column, as written: a Column, or an
            AI_EXTRACT call with a field (a Dot).
        comparison: One of quail.logical.COLUMN_COMPARISONS; ``is true``
            for a boolean operand tested on its own.
        value: The literal or tuple of literals compared with.
        function: The function applied to the column, or None.
        negated: Whether the test is written negated, as ``NOT LIKE``.
    """

    operand: exp.Expression
    comparison: str
    value: object = None
    function: str | None = None
    negated: bool = False


# the SQL functions of one column, by the name the expression tree uses
SQL_FUNCTIONS = {exp.Length: "length", exp.Lower: "lower", exp.Upper: "upper",
                 exp.Trim: "trim"}
COLUMN_NODES = (exp.Column, *SQL_FUNCTIONS)


def unwrap_function(node) -> tuple[exp.Expression, str | None]:
    """Return the node under a column function and the function's name."""
    if isinstance(node, exp.Trim) and (node.expression is not None
                                       or node.args.get("position")):
        return node, None
    function = SQL_FUNCTIONS.get(type(node))
    return (node.this, function) if function else (node, None)


# the arithmetic of value expressions, by the Arrow compute function it calls
ARITHMETIC = {exp.Add: "add", exp.Sub: "subtract", exp.Mul: "multiply",
              exp.Div: "divide"}
# the comparisons of value expressions, by the Arrow compute function
# they call when no side is a literal, as in r.stars > r.min_stars
COMPARISON_FUNCTIONS = {exp.EQ: "equal", exp.NEQ: "not_equal", exp.LT: "less",
                        exp.LTE: "less_equal", exp.GT: "greater",
                        exp.GTE: "greater_equal"}
# the binary operators of value expressions, by the function they call
OPERATORS = {**ARITHMETIC, **COMPARISON_FUNCTIONS}


def column_call(node) -> str | None:
    """Return the binder method for a column-operator call, or None."""
    if isinstance(node, exp.AIClassify):
        return "bind_ai_classify"
    if isinstance(node, exp.Anonymous):
        return {"AI_EXTRACT": "bind_ai_extract", "AI_MAP": "bind_ai_map",
                "AI_SAE_FEATURES": "bind_ai_sae_features"}.get(str(node.this).upper())
    return None


def computed_value(node) -> bool:
    """Return whether a node is a function or arithmetic over columns and literals.

    An AI call, an aggregate, and a window are not computed values.
    """
    if not isinstance(node, (exp.Func, exp.Paren, exp.Neg, *OPERATORS,
                             *TRANSLATED_NODES)) or isinstance(
            node, (exp.AggFunc, exp.AIClassify, exp.Window)):
        return False
    return not any(
        isinstance(inner, exp.Anonymous) and str(inner.this).upper().startswith("AI_")
        for inner in node.walk()) and node.find(exp.Column) is not None


def _operand(node) -> tuple[exp.Expression, str | None] | None:
    """Return the column a test reads and the function around it, or None.

    A function or arithmetic over columns and literals, such as
    ``ABS(r.stars - 3)``, is returned whole, with no function.
    """
    inner, function = unwrap_function(node)
    if isinstance(inner, (exp.Column, exp.Dot)) or column_call(inner) is not None:
        return inner, function
    if computed_value(node):
        return node, None
    return None


def unwrap_hint(b, term) -> tuple:
    """Split ``HINT(test, {'selectivity': fraction})`` into the test and its hint.

    Args:
        b: The expression binder, which parses the option object.
        term: The term.

    Returns:
        (term, selectivity), with None for a term without the hint.

    Raises:
        CompileError: The hint has no option object, or an option other
            than selectivity.
    """
    if not (isinstance(term, exp.Anonymous) and str(term.this).upper() == "HINT"):
        return term, None
    if len(term.expressions) != 2:
        raise CompileError(
            "HINT takes a test and an option object like {'selectivity': 0.4}")
    inner, options = term.expressions
    return inner, b.parse_options(options, {"selectivity"}).get("selectivity")


def parse_plain_test(term, dialect: str) -> PlainTest:
    """Parse one WHERE or ON term as a test of a column against literals.

    Args:
        term: The term.
        dialect: "snowflake" or "bq"; it decides whether CONTAINS
            ignores case and whether a regex matches the whole value.

    Returns:
        The test. A comparison of two value expressions, or a boolean
        expression alone, is tested ``is true``.

    Raises:
        CompileError: The term is not a supported plain test.
    """
    node, negated = term, False
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        found = _operand(node.this)
        if found is not None:
            return PlainTest(found[0], "is null", function=found[1])
    if isinstance(node, exp.In) and node.expressions:
        found = _operand(node.this)
        if found is not None:
            return PlainTest(found[0], "in",
                             tuple(sql_literal(item) for item in node.expressions),
                             found[1])
    comparison = _text_comparison(node, dialect)
    if comparison is not None:
        tested, pattern = node.this, node.expression
        if comparison == "icontains" and isinstance(tested, exp.Lower):
            # BigQuery's CONTAINS_SUBSTR parses as CONTAINS(LOWER(a), LOWER(b))
            tested, pattern = tested.this, pattern.this
        if node.args.get("negate"):
            negated = not negated
        found = _operand(tested)
        if found is None:
            raise CompileError(f"{term.sql()} tests a column")
        pattern = sql_literal(pattern)
        if not isinstance(pattern, str):
            raise CompileError(f"{comparison.upper()} compares with a string")
        if comparison == "regex" and dialect == "snowflake":
            # Snowflake's REGEXP_LIKE and RLIKE match the whole value
            pattern = f"^(?:{pattern})$"
        return PlainTest(found[0], comparison, pattern, found[1], negated)
    comparison = _PLAIN_COMPARISONS.get(type(node))
    if comparison is not None:
        left, right = _operand(node.this), _operand(node.expression)
        if left is not None and right is None:
            return PlainTest(left[0], comparison, sql_literal(node.expression), left[1])
        if right is not None and left is None:
            return PlainTest(right[0], _FLIPPED[comparison], sql_literal(node.this),
                             right[1])
        if left is not None and right is not None:
            # two value expressions: an Arrow comparison, tested true;
            # bind_condition reads one between two tables' columns as a
            # join condition
            return PlainTest(node, "is true")
    if isinstance(node, (exp.Column, exp.Dot)) or computed_value(node):
        return PlainTest(node, "is true")
    raise CompileError(
        f"{term.sql()} is not supported; a regular predicate is {_SUPPORTED}")


def bind_condition(b, term, selected: dict, *, join_keys: bool = False):
    """Bind a WHERE or ON term that asks no model as a condition tree.

    NOT, AND, and OR become Not, And, and Or; every other term is one
    plain test, of a source column or of an AI.EXTRACT field.

    Args:
        b: The expression binder, which resolves the columns.
        term: The term.
        selected: The AI calls in SELECT, by AS name.
        join_keys: Whether the term is in ON, where a comparison of two
            tables' columns, outside NOT, AND, and OR, is a join key.

    Returns:
        The condition, or None for a join key, which the caller reads
        as a join condition.

    Raises:
        CompileError: A term is not a supported test.
    """
    node = term.unnest()
    if isinstance(node, exp.Not):
        return Not(_bind_operand(b, node.this, selected))
    if isinstance(node, (exp.And, exp.Or)):
        kind = And if isinstance(node, exp.And) else Or
        return kind(tuple(_bind_operand(b, part, selected) for part in node.flatten()))
    test = parse_plain_test(node, b.dialect)
    if join_keys and _join_condition(b, test):
        return None
    found = column_test(b, test, selected)
    predicate = found[1] if found is not None else bind_plain_test(b, test)
    return Not(predicate) if test.negated else predicate


def _join_condition(b, test: PlainTest) -> bool:
    """Return whether a test compares one table's values with another's.

    Each side is a column or an expression of one table's columns, such
    as ``LOWER(r.lang) = u.lang``.
    """
    node = test.operand
    if type(node) not in COMPARISON_FUNCTIONS:
        return False
    try:
        sides = [b.bind_value(side) for side in (node.this, node.expression)]
    except CompileError:
        return False
    aliases = [{reference.alias for reference in column_references(side)}
               for side in sides]
    return all(len(found) == 1 for found in aliases) and aliases[0] != aliases[1]


def _bind_operand(b, term, selected: dict):
    """Bind a term under NOT, AND, or OR."""
    return bind_condition(b, term, selected)


def _text_comparison(node, dialect: str) -> str | None:
    """Return the comparison name of a text test node, or None."""
    if isinstance(node, exp.Contains):
        # BigQuery's CONTAINS_SUBSTR ignores case; Snowflake's CONTAINS
        # does not
        return "icontains" if dialect == "bq" else "contains"
    return _TEXT_TESTS.get(type(node))


def column_path(column: exp.Column) -> tuple[str, ...]:
    """Return the parts of a column reference as written, such as (a, b, c)."""
    parts = [column.args.get(part) for part in ("catalog", "db", "table", "this")]
    return tuple(str(part.name) for part in parts if part is not None)


def ai_aliases(tree: exp.Select) -> dict:
    """Return the AS names of the column-operator calls in SELECT."""
    return {item.alias: item.this for item in tree.expressions
            if isinstance(item, exp.Alias) and column_call(item.this) is not None}


def bind_plain_test(b, test: PlainTest, tables=None) -> RegularPredicate:
    """Bind a plain test of a source column.

    Args:
        b: The expression binder, which resolves the column.
        test: The parsed test.
        tables: (alias, provider) pairs to resolve against instead of
            the tables in scope.
    """
    if computed_value(test.operand):
        predicate = RegularPredicate(b.bind_value(test.operand, tables),
                                     test.comparison, test.value)
        predicate.validate()
        return predicate
    if not isinstance(test.operand, exp.Column):
        raise CompileError(
            f"{test.operand.sql()} reads a field of an AI.EXTRACT call "
            f"that SELECT does not return; name the call in SELECT")
    predicate = RegularPredicate(b.resolve_value(test.operand, tables, test.function),
                                 test.comparison, test.value)
    predicate.validate()
    return predicate


def column_test(b, test: PlainTest, selected: dict) -> tuple | None:
    """Resolve a plain test of a column-operator result."""
    method = column_call(test.operand)
    if method is None:
        return extract_test(b, test, selected)
    if method == "bind_ai_extract":
        raise CompileError("test a field of AI.EXTRACT, such as lease.end_date")
    call = getattr(b, method)(test.operand)
    if method == "bind_ai_classify":
        call, _, _ = call
    if len(call.aliases()) != 1:
        raise CompileError("a filter on a label tests a one-document classification")
    (alias,) = call.aliases()
    name = next((name for name, node in selected.items() if node == test.operand),
                f"__column_{alias}_{len(b.tested_calls)}")
    name = b.tested_calls.setdefault(call, name)
    predicate = RegularPredicate(
        column_expression(ColumnRef(alias, dict(b.tables)[alias], name),
                          (), test.function), test.comparison, test.value)
    if call.kind == "features":
        if (test.function is not None
                or test.comparison not in ("is null", "is not null")):
            raise CompileError(
                "AI.SAE_FEATURES returns a list; test IS NULL or IS NOT NULL, "
                "or UNNEST it to test its fields")
        predicate.validate()
    else:
        predicate.validate_text_column()
    return alias, predicate


def extract_test(b, test: PlainTest, selected: dict) -> tuple | None:
    """Resolve a plain test of an AI.EXTRACT field named in SELECT.

    The field is written as ``name.field`` for the answer, as
    ``name_span.field.start`` or ``name_span.field.end`` for its span,
    or as the repeated call with ``.field``.

    Returns:
        (alias, RegularPredicate), or None when the test reads no
        extraction.

    Raises:
        CompileError: The struct name is both a table alias and an AS
            name, or the field is not one the call returns.
    """
    operand = test.operand
    if not isinstance(operand, (exp.Column, exp.Dot)):
        return None
    if isinstance(operand, exp.Dot):
        call, field, path = operand.this, str(operand.expression.name), ()
        if not (isinstance(call, exp.Anonymous)
                and str(call.this).upper() == "AI_EXTRACT"):
            raise CompileError(f"{operand.sql()} is not a supported column")
        name = next((alias for alias, node in selected.items() if node == call),
                    None)
        if name is None:
            raise CompileError(
                f"{call.sql()} in WHERE is not returned by SELECT; name it "
                f"there and test it by its AS name")
    else:
        parts = column_path(operand)
        if len(parts) < 2:
            return None
        name, field, path = parts[0], parts[1], parts[2:]
        if name.endswith(SPAN_SUFFIX) and name[:-len(SPAN_SUFFIX)] in selected:
            name = name[:-len(SPAN_SUFFIX)]
            if tuple(path) not in (("start",), ("end",)):
                raise CompileError(
                    f"{operand.sql()} reads a span; test its start or end, "
                    f"such as {name}{SPAN_SUFFIX}.{field}.start")
            if test.function is not None:
                raise CompileError(
                    f"{operand.sql()} is a number; {test.function.upper()}() "
                    f"takes text")
        elif name in selected:
            if path:
                raise CompileError(f"{operand.sql()} reads a field under an answer")
        else:
            return None
        if name in dict(b.tables):
            raise CompileError(
                f"{name!r} is both a table alias and an AS name in SELECT; "
                f"rename one")
    call = selected[name]
    if not (isinstance(call, exp.Anonymous)
            and str(call.this).upper() == "AI_EXTRACT"):
        raise CompileError(f"{name!r} is not an AI.EXTRACT column")
    struct, members = b.bind_ai_extract(call, name)
    if field not in struct.fields:
        raise CompileError(
            f"AI.EXTRACT {name!r} has no field {field!r}; its fields are "
            f"{list(struct.fields)}")
    member = next(member for member in members if member.name == struct.member(field))
    (alias,) = member.expression.aliases()
    column = struct.member(field) + (SPAN_SUFFIX if path else "")
    predicate = RegularPredicate(
        column_expression(ColumnRef(alias, dict(b.tables)[alias], column),
                          tuple(path), test.function),
        test.comparison, test.value)
    if path:
        predicate.validate()
    else:
        predicate.validate_text_column()
    return alias, predicate
