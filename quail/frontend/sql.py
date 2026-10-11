"""Parse SQL with SQLGlot and construct a Quail logical plan."""

from enum import StrEnum

import sqlglot
from sqlglot import exp

from quail.catalog import Catalog
from quail.frontend.sql_binding.expressions import (
    ExpressionBinder,
)
from quail.frontend.sql_binding.relations import (
    JoinBinder,
    bind_where,
    check_join_coverage,
)
from quail.frontend.sql_binding.select import (
    SelectBinder,
    _parse_distinct,
    _parse_limit,
    _parse_offset,
    bind_order,
)
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    LogicalPlan,
    LogicalPlanBuilder,
    is_score,
)

FORBIDDEN = (
    (exp.Qualify, "QUALIFY"),
    (exp.Window, "window functions"),
    (exp.Union, "UNION"),
    (exp.Except, "EXCEPT"),
    (exp.Intersect, "INTERSECT"),
)


class SQLDialect(StrEnum):
    """SQL dialects accepted by the Quail front end."""

    SNOWFLAKE = "snowflake"
    BQ = "bq"


def _sqlglot_dialect(dialect: SQLDialect) -> str:
    return "bigquery" if dialect is SQLDialect.BQ else "snowflake"


def _normalize_ai_calls(sql: str, dialect: SQLDialect) -> str:
    """Lower dotted AI calls to internal function names."""
    tokenizer = sqlglot.Dialect.get_or_raise(_sqlglot_dialect(dialect)).tokenizer()
    try:
        tokens = tokenizer.tokenize(sql)
    except sqlglot.errors.SqlglotError as error:
        raise CompileError(f"parse error: {error}") from error
    edits = []
    for index in range(len(tokens) - 3):
        head, dot, name, left = tokens[index : index + 4]
        function = name.text.upper()
        replacement = None
        if function == "SCORE":
            replacement = "AI_SCORE("
        elif function == "CLASSIFY":
            replacement = "AI_CLASSIFY("
        elif function == "EXTRACT":
            replacement = "AI_EXTRACT("
        elif function == "IF" and dialect is SQLDialect.BQ:
            replacement = "AI_FILTER("
        if (
            head.text.upper() == "AI"
            and dot.text == "."
            and left.text == "("
            and replacement is not None
        ):
            edits.append((head.start, left.end + 1, replacement))
    for start, end, replacement in reversed(edits):
        sql = sql[:start] + replacement + sql[end:]
    return sql


def _reject_forbidden(tree) -> None:
    for cls, name in FORBIDDEN:
        if list(tree.find_all(cls)):
            raise CompileError(
                f"{name} is outside the language: Quail runs the "
                f"semantic part; do the relational part in the "
                f"database the ids came from"
            )
    if list(tree.find_all(exp.Or)):
        raise CompileError(
            "OR between AI predicates is not supported: a disjunction "
            "belongs inside one prompt's text, where the model "
            "evaluates it"
        )
    for fn in tree.find_all(exp.Anonymous):
        name = str(fn.this).upper()
        if name.startswith("AI_") and name not in {
            "AI_FILTER",
            "AI_SCORE",
            "AI_EXTRACT",
        }:
            raise CompileError(
                f"{name} is not supported; supported AI functions are "
                "AI_FILTER, AI_SCORE, AI_CLASSIFY, and AI_EXTRACT"
            )
    # subqueries are legal only as the EXISTS form, checked
    # structurally; any other subquery is refused here
    for sub in tree.find_all(exp.Subquery):
        raise CompileError(
            "subqueries other than [NOT] EXISTS (SELECT 1 ...) are not supported"
        )


def _validate_query(
    b: ExpressionBinder, joins: JoinBinder, columns: tuple[ColumnRef | Alias, ...]
) -> None:
    """Validate output names and supported combinations of AI operators."""
    projected = tuple(column for column in columns if isinstance(column, Alias))
    projected_scores = tuple(
        column for column in projected if column.expression.kind == "score"
    )
    names_by_prompt = {}
    for score in projected:
        if score.name in dict(b.tables):
            raise CompileError(
                f"output name {score.name!r} is also a table "
                f"alias; pick another AS name"
            )
        names = names_by_prompt.setdefault(score.expression.prompt, [])
        if names:
            raise CompileError(
                f"the same AI expression is projected as "
                f"{names[0]!r} and {score.name!r}; project it once"
            )
        names.append(score.name)
    for alias, equalities in joins.conditions.items():
        if any(alias in score.expression.aliases() for score in projected_scores):
            raise CompileError(
                f"JOIN {alias} ON {equalities[0]} projects a pair "
                f"AI.SCORE; a projected pair score runs over CROSS JOIN, "
                f"and an ON equality is supported when the score is "
                f"compared in ON"
            )
        raise CompileError(
            f"JOIN {alias} ON {equalities[0]} has no AI predicate over "
            f"its pairs; a plain join belongs in the database the ids "
            f"came from"
        )
    check_join_coverage(b, joins.aliases, projected_scores)

    filter_predicates = [
        predicate for predicates in b.filters.values() for predicate in predicates
    ]
    score_flags = (
        [is_score(predicate.expression) for predicate in filter_predicates]
        + [is_score(join.predicate) for join in b.joins]
        + [True for _ in projected_scores]
    )
    if any(score_flags) and not all(score_flags):
        raise CompileError(
            "AI.SCORE cannot be mixed with generative AI predicates in one query"
        )

    if not b.joins and not b.filters and not b.label_tests and not projected:
        raise CompileError(
            "the query has no AI predicate; a plain "
            "scan belongs in the database the ids came "
            "from"
        )


def _bind_label_names(b: ExpressionBinder, columns: tuple[ColumnRef | Alias, ...]):
    """Assign names to classifications used in projections and filters."""
    projected = tuple(column for column in columns if isinstance(column, Alias))
    # a filter on a label tests the projected column of the same call;
    # a classification only tested is named after its first test's
    # position among the alias's filters
    named = {
        column.expression: column.name
        for column in projected
        if column.expression.kind == "label"
    }
    labels = {}
    for alias, tests in b.label_tests.items():
        asks = len(b.filters.get(alias, ()))
        for index, (call, _, _) in enumerate(tests):
            labels.setdefault(alias, {}).setdefault(
                call, named.get(call, f"__label_{alias}_{asks + index}")
            )
    for call, name in named.items():
        if len(call.aliases()) == 1:
            labels.setdefault(call.aliases()[0], {}).setdefault(call, name)

    return named, labels


def compile_sql(
    sql: str,
    catalog: Catalog,
    tokenizer=None,
    dialect: SQLDialect | str = SQLDialect.SNOWFLAKE,
    turn: tuple[str, str] = ("", ""),
    layout: str = "ai-if",
) -> LogicalPlan:
    """Compile AI SQL text into a LogicalPlan.

    Args:
        sql: A SELECT statement in the chosen dialect.
        catalog: Registered tables and their schemas.
        tokenizer: Tokenizer used to bind AI prompts.
        dialect: SQL dialect used to parse the statement.
        turn: Model chat text placed before and after each prompt.
        layout: Prompt layout used by the model.

    Returns:
        The bound logical query plan.

    Raises:
        CompileError: The query is invalid or uses unsupported SQL.
        ValueError: The dialect is not supported.
    """
    try:
        dialect = SQLDialect(dialect)
    except ValueError as error:
        raise ValueError(
            f"unsupported SQL dialect {dialect!r}; expected snowflake or bq"
        ) from error
    sql = _normalize_ai_calls(sql, dialect)
    try:
        tree = sqlglot.parse_one(sql, dialect=_sqlglot_dialect(dialect))
    except sqlglot.errors.SqlglotError as e:
        raise CompileError(f"parse error: {e}") from e
    if not isinstance(tree, exp.Select):
        raise CompileError("the query must be a single SELECT")
    _reject_forbidden(tree)

    limit = _parse_limit(tree)
    offset = _parse_offset(tree)
    distinct = _parse_distinct(tree)

    b = ExpressionBinder(catalog, tokenizer, turn, layout)

    joins = JoinBinder(b)
    joins.bind_tables(tree)
    bind_where(b, tree, joins)

    selection = SelectBinder(b).bind(tree)
    columns, aggregation = selection.columns, selection.aggregation
    order = bind_order(tree, b, columns, aggregation)
    _validate_query(b, joins, columns)
    named, labels = _bind_label_names(b, columns)
    extracts = {}
    for column in columns:
        if isinstance(column, Alias) and column.expression.kind == "extract":
            extracts.setdefault(column.expression.aliases()[0], []).append(
                (column.expression, column.name)
            )

    logical = LogicalPlanBuilder()
    for alias, provider in b.tables:
        logical.add_scan(
            alias,
            provider,
            b.doc_columns.get(alias, ""),
            tuple(b.filters.get(alias, ())),
            regular_predicates=tuple(b.regular_predicates.get(alias, ())),
            labels=tuple(labels.get(alias, {}).items()),
            label_filters=tuple(
                (labels[alias][call], accepted, selectivity)
                for call, accepted, selectivity in b.label_tests.get(alias, ())
            ),
            extracts=tuple(extracts.get(alias, ())),
        )
    for join in b.joins:
        logical.add_join(join)
    for alias in joins.aliases:
        if alias not in joins.claimed:
            logical.add_cross_join(alias)
    for call, name in named.items():
        if len(call.aliases()) == 2:
            logical.add_classify(call, name)
    return logical.project(
        tuple(columns),
        limit,
        order=order,
        offset=offset,
        distinct=distinct,
        aggregation=aggregation,
    )
