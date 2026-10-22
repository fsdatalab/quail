"""Parse SQL with SQLGlot and construct a Quail logical plan."""

from collections.abc import Callable
from enum import StrEnum

import sqlglot
from sqlglot import exp

from quail.catalog import Catalog, MemoryTableProvider
from quail.frontend.prompts import PromptBinder
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
    Cte,
    LogicalPlan,
    LogicalPlanBuilder,
    WithPlan,
    column_name,
    is_score,
)
from quail.logical.expressions import aliased_call

FORBIDDEN = (
    (exp.Qualify, "QUALIFY"),
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
    b: ExpressionBinder, joins: JoinBinder, columns: tuple[ColumnRef | Alias, ...],
    structs: tuple = (),
) -> None:
    """Validate output names and supported combinations of AI operators."""
    for struct in structs:
        if struct.name in dict(b.tables):
            raise CompileError(
                f"output name {struct.name!r} is also a table "
                f"alias; pick another AS name"
            )
    for column in columns:
        name = column_name(column)
        if isinstance(column, Alias) and name in dict(b.tables):
            raise CompileError(
                f"output name {name!r} is also a table alias; pick another AS name")
        if isinstance(column, Alias) and any(
                name in b.catalog.get(provider).columns for _, provider in b.tables):
            raise CompileError(
                f"output name {name!r} is also a column of a table in the query; "
                f"pick another AS name")
    projected = tuple(column for column in columns
                      if aliased_call(column) is not None)
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
    projected = tuple(column for column in columns
                      if aliased_call(column) is not None)
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
    dialect: SQLDialect | str = SQLDialect.SNOWFLAKE,
    turn: tuple[str, str] = ("", ""),
    layout: str = "ai-if",
    prompts: PromptBinder | None = None,
    cte_schema: Callable | None = None,
) -> LogicalPlan | WithPlan:
    """Compile AI SQL text into a LogicalPlan, or a WithPlan for a WITH query.

    Args:
        sql: A SELECT statement in the chosen dialect, with optional
            common table expressions (CTEs) in a WITH clause.
        catalog: Registered tables and their schemas.
        dialect: SQL dialect used to parse the statement.
        turn: Model chat text placed before and after each prompt.
        layout: Prompt layout used by the model.
        prompts: Model-aware prompt binder. Overrides turn and layout.
        cte_schema: Returns the Arrow schema of a CTE's result from its
            LogicalPlan and the catalog it reads, so the queries reading
            the CTE are compiled against its columns. Required for a
            WITH query.

    Returns:
        The bound logical query plan: a WithPlan when the statement has
        a WITH clause, a LogicalPlan otherwise.

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
    prompts = prompts or PromptBinder(turn=turn, layout=layout)
    with_ = tree.args.get("with_")
    if with_ is None:
        return _compile_select(tree, catalog, prompts, dialect)
    return _compile_with(tree, with_, catalog, prompts, dialect, cte_schema)


def _compile_with(tree, with_, catalog: Catalog, prompts, dialect,
                  cte_schema) -> WithPlan:
    """Compile each CTE against the tables and CTEs before it, then the SELECT."""
    if with_.recursive:
        raise CompileError(
            "WITH RECURSIVE is not supported; a CTE reads registered tables "
            "and the CTEs before it")
    if cte_schema is None:
        raise CompileError(
            "a WITH query needs the schema of each CTE's result; compile it "
            "through Session.sql()")
    tree.set("with_", None)
    tables = Catalog(dict(catalog.providers))
    ctes = []
    for cte in with_.expressions:
        name = cte.alias
        if cte.args["alias"].columns:
            raise CompileError(
                f"WITH {name} (...) column names are not supported; name "
                f"the columns with AS inside the CTE")
        body = cte.this
        if not isinstance(body, exp.Select):
            raise CompileError(f"the CTE {name!r} must be a SELECT")
        if body.args.get("with_") is not None:
            raise CompileError(
                f"the CTE {name!r} has its own WITH; list every CTE in the "
                f"outer WITH, each reading the ones before it")
        if name in catalog:
            raise CompileError(
                f"the CTE {name!r} is also a registered table; pick another name")
        if name in tables:
            raise CompileError(f"the CTE {name!r} is defined twice")
        plan = _compile_select(body, tables, prompts, dialect)
        ctes.append(Cte(name, plan, _cte_reads(plan, tables, catalog)))
        tables.register(name, MemoryTableProvider(
            cte_schema(plan, tables).empty_table(), id_col="", identity=f"cte:{name}"))
    query = _compile_select(tree, tables, prompts, dialect)
    return WithPlan(tuple(ctes), query, _cte_reads(query, tables, catalog))


def _cte_reads(plan: LogicalPlan, tables: Catalog, catalog: Catalog) -> tuple:
    """Return the CTE names the plan scans: its providers not in the catalog."""
    return tuple(dict.fromkeys(
        scan.provider for scan in plan.operators().scans
        if scan.provider in tables and scan.provider not in catalog))


def _compile_select(tree: exp.Select, catalog: Catalog, prompts: PromptBinder,
                    dialect: SQLDialect) -> LogicalPlan:
    """Compile one SELECT against the tables in the catalog."""
    limit = _parse_limit(tree)
    offset = _parse_offset(tree)
    distinct = _parse_distinct(tree)

    b = ExpressionBinder(catalog, prompts, dialect=dialect.value)

    joins = JoinBinder(b)
    joins.bind_tables(tree)
    bind_where(b, tree, joins)

    selection = SelectBinder(b).bind(tree)
    columns, aggregation = selection.columns, selection.aggregation
    order = bind_order(tree, b, columns, aggregation, selection.structs,
                       selection.windows)
    _validate_query(b, joins, columns, selection.structs)
    named, labels = _bind_label_names(b, columns)
    extracts = {}
    for column in columns:
        if aliased_call(column, "extract") is not None:
            extracts.setdefault(column.expression.aliases()[0], []).append(
                (column.expression, column.name)
            )

    logical = LogicalPlanBuilder()
    for alias, provider in b.tables:
        logical.add_scan(
            alias,
            provider,
            b.document_columns.get(alias, ""),
            tuple(b.filters.get(alias, ())),
            regular_predicates=tuple(b.regular_predicates.get(alias, ())),
            labels=tuple(labels.get(alias, {}).items()),
            label_filters=tuple(
                (labels[alias][call], accepted, selectivity)
                for call, accepted, selectivity in b.label_tests.get(alias, ())
            ),
            extracts=tuple(extracts.get(alias, ())),
            extract_filters=tuple(b.extract_tests.get(alias, ())),
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
        structs=selection.structs,
        windows=selection.windows,
        output=selection.output,
    )
