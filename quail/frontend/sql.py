"""Parse SQL with SQLGlot and construct a Quail logical plan."""

import re
from collections import Counter
from enum import StrEnum

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.qualify_columns import qualify_columns
from sqlglot.optimizer.qualify_tables import qualify_tables
from sqlglot.optimizer.scope import Scope, ScopeType, traverse_scope
from sqlglot.schema import MappingSchema

from quail.catalog import Catalog, MemoryTableProvider
from quail.frontend.prompts import PromptBinder
from quail.frontend.sql_binding.expressions import (
    ExpressionBinder,
    _conjuncts,
)
from quail.frontend.sql_binding.predicates import SQL_FUNCTIONS, column_path
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
from quail.frontend.sql_binding.unnest import list_fields, take_unnests
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    LogicalPlan,
    LogicalPlanBuilder,
    WithCTE,
    column_name,
    cte_schema,
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
        elif function == "SAE_FEATURES":
            replacement = "AI_SAE_FEATURES("
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
            "AI_SAE_FEATURES",
        }:
            raise CompileError(
                f"{name} is not supported; supported AI functions are "
                "AI_FILTER, AI_SCORE, AI_CLASSIFY, AI_EXTRACT, and "
                "AI_SAE_FEATURES"
            )
    # a subquery is legal in FROM, where it runs as its own stage, and
    # as the EXISTS form, checked structurally; any other is refused
    for sub in tree.find_all(exp.Subquery):
        if not isinstance(sub.parent, (exp.From, exp.Join)):
            raise CompileError(
                "subqueries other than in FROM and [NOT] EXISTS (SELECT 1 ...) "
                "are not supported")


def _validate_query(
    b: ExpressionBinder, joins: JoinBinder, columns: tuple[ColumnRef | Alias, ...],
    structs: tuple = (), *, allow_relational: bool = False,
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

    if (not allow_relational and not b.joins and not b.filters
            and not b.label_tests and not projected):
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
) -> LogicalPlan:
    """Compile AI SQL text into one logical plan.

    Args:
        sql: A SELECT statement in the chosen dialect, with optional
            common table expressions (CTEs) in WITH clauses and
            subqueries in FROM.
        catalog: Registered tables and their schemas.
        dialect: SQL dialect used to parse the statement.
        turn: Model chat text placed before and after each prompt.
        layout: Prompt layout used by the model.
        prompts: Model-aware prompt binder. Overrides turn and layout.

    Returns:
        The bound logical query plan, including any CTE definitions.

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
    for with_ in tree.find_all(exp.With):
        if with_.recursive:
            raise CompileError(
                "WITH RECURSIVE is not supported; a CTE reads registered "
                "tables and the CTEs before it")
    prompts = prompts or PromptBinder(turn=turn, layout=layout)
    if len(_scopes(tree)) == 1:
        return _compile_select(tree, catalog, prompts, dialect)
    for select in tree.find_all(exp.Select):
        for item in select.selects:
            if isinstance(item, exp.Alias):
                item.meta["user_alias"] = True
    tables = Catalog(dict(catalog.providers))
    ctes = []
    plan = _compile_stages(_qualified(tree, catalog, dialect), tables, prompts,
                           dialect, ctes)
    result = (LogicalPlan(WithCTE(plan.root, tuple(
        (name, definition.root) for name, definition in ctes))) if ctes else plan)
    if not any(stage.operators().prompts for _, stage in result.stages()):
        raise CompileError("the query has no AI predicate")
    return result


def _qualified(tree: exp.Select, catalog: Catalog, dialect: SQLDialect) -> exp.Select:
    """Qualify every column with its table, as sqlglot's merge requires."""
    # Catalog names keep their spelling in both dialects; sqlglot otherwise
    # uppercases the implicit aliases it adds for Snowflake tables.
    for table in tree.find_all(exp.Table):
        if not table.alias and isinstance(table.this, exp.Identifier):
            table.set("alias", exp.TableAlias(this=table.this.copy()))
    schema = MappingSchema(
        {name: dict.fromkeys(provider.columns, "UNKNOWN")
         for name, provider in catalog.providers.items()},
        dialect=_sqlglot_dialect(dialect), normalize=False)
    try:
        return qualify_columns(
            qualify_tables(tree, dialect=_sqlglot_dialect(dialect)),
            schema=schema, dialect=_sqlglot_dialect(dialect))
    except OptimizeError as error:
        raise CompileError(str(error)) from error


def _compile_stages(tree: exp.Select, tables: Catalog, prompts, dialect,
                    ctes: list) -> LogicalPlan:
    """Compile a statement's stages into ctes and return the plan of its SELECT.

    A CTE or FROM subquery runs as its own stage when an AI function
    reads a column it computes, when two queries read it, or when
    sqlglot's merge_subqueries cannot inline it; every other one is
    merged into the query that reads it, so the statement becomes as
    few plans as its model calls allow.
    """
    tree = eliminate_ctes(tree)
    scopes = _scopes(tree)
    staged = _staged(scopes)
    # a merge appends a CTE's WHERE after its reader's; the CTE's runs first
    for position, scope in enumerate(scopes):
        for term in _conjuncts(scope.expression.args["where"].this) \
                if scope.expression.args.get("where") else ():
            term.meta["stage"] = position
    # the merged CTEs, which a detached stage may still read
    pool = [cte for cte in tree.find_all(exp.CTE)
            if not any(cte is scope.expression.parent for scope in staged)]
    for scope in staged:
        if any(candidate.expression is scope.expression
               for candidate in _staged(_scopes(tree))):
            _compile_stage(scope, pool, tables, prompts, dialect, ctes)
    tree = merge_subqueries(eliminate_ctes(tree))
    # Merging the right side of a join moves its WHERE into ON. Restore
    # those filters so they still run as document operators before the join.
    for join in tree.args.get("joins") or ():
        if join.args.get("on") is None:
            continue
        predicates = []
        for term in _conjuncts(join.args["on"]):
            if "stage" in term.meta:
                tree = tree.where(term, copy=False)
            else:
                predicates.append(term)
        join.set("on", exp.and_(*predicates, copy=False) if predicates else None)
    if tree.args.get("where"):
        tree.set("where", exp.Where(this=exp.and_(*sorted(
            _conjuncts(tree.args["where"].this),
            key=lambda term: term.meta.get("stage", 0)), copy=False)))
    for scope in _scopes(tree):
        if not scope.is_root:
            _compile_stage(scope, (), tables, prompts, dialect, ctes)
    tree.set("with_", None)
    # qualifying wrote a struct field path as Dot nodes over the column;
    # the binders read the path as one column
    for dot in list(tree.find_all(exp.Dot)):
        parts, node = [], dot
        while isinstance(node, exp.Dot):
            parts.insert(0, node.expression.name)
            node = node.this
        if isinstance(node, exp.Column):
            path = [*column_path(node), *parts]
            if len(path) > 4:
                raise CompileError(
                    f"field path {dot.sql()} has more than two nested fields")
            dot.replace(exp.column(*reversed(path)))
    # qualifying named every select item; a column keeps its own name
    for item in tree.selects:
        if isinstance(item, exp.Alias) and not item.meta.get("user_alias") and (
                (isinstance(item.this, exp.Column) and item.alias == item.this.name)
                or AUTO_ALIAS.fullmatch(item.alias)):
            item.replace(item.this)
    return _compile_select(tree, tables, prompts, dialect, allow_relational=True)


def _compile_stage(scope: Scope, pool, tables: Catalog, prompts, dialect,
                   ctes: list) -> None:
    """Detach one stage from its statement, compile it, and add it to ctes.

    The statement reads the stage's result as a table named by the
    CTE name or the subquery alias.
    """
    # the CTE name, or the subquery alias sqlglot gave or qualifying added
    name = scope.expression.parent.alias
    if name in tables:
        raise CompileError(
            f"{name!r} names a registered table or another CTE; pick "
            f"another name")
    owner = scope.expression.parent
    body = scope.expression
    if isinstance(owner, exp.CTE):
        owner.pop()
    else:
        owner.replace(exp.table_(name, alias=name))
    body.set("with_", exp.With(expressions=[cte.copy() for cte in pool])
             if pool else None)
    plan = _compile_stages(body, tables, prompts, dialect, ctes)
    ctes.append((name, plan))
    tables.register(name, MemoryTableProvider(
        cte_schema(plan).empty_table(), id_col="", identity=f"cte:{name}"))


def _scopes(tree) -> list:
    """Return the CTE, FROM subquery, and statement scopes in dependency order."""
    # an EXISTS body and an UNNEST are bound with their query, not as stages
    return [scope for scope in traverse_scope(tree)
            if scope.scope_type not in (ScopeType.SUBQUERY, ScopeType.UDTF)]


AI_FUNCTIONS = {"PROMPT", "AI_FILTER", "AI_SCORE", "AI_EXTRACT"}
# the name sqlglot gives a select item that is not a column
AUTO_ALIAS = re.compile(r"_col_\d+", re.IGNORECASE)


def _staged(scopes) -> list:
    """Return the scopes that must run as stages, in dependency order.

    A CTE or FROM subquery is a stage when an AI function reads a
    column it computes, so the text does not exist before the model has
    run, when two queries read it, so it runs once, or when it reads an
    UNNEST, which sqlglot's merge drops the alias of.
    """
    readers = Counter()
    forced = set()
    for scope in dict.fromkeys(child for root in scopes for child in root.traverse()):
        for _, source in scope.selected_sources.values():
            if isinstance(source, Scope):
                readers[id(source)] += 1
        for column in scope.columns:
            owner = scope
            while owner.parent is not None and column.table not in owner.sources:
                owner = owner.parent
            source = owner.sources.get(column.table)
            ancestor = column.find_ancestor(
                exp.Anonymous, exp.AIClassify, *SQL_FUNCTIONS)
            if (isinstance(source, Scope) and ancestor is not None
                    and (isinstance(ancestor, (exp.AIClassify, *SQL_FUNCTIONS))
                         or str(ancestor.this).upper() in AI_FUNCTIONS)
                    and column.name in _computed(source)):
                forced.add(id(source))
    for scope in scopes:
        if any(isinstance(join.this, exp.Unnest)
               for join in scope.expression.args.get("joins") or ()):
            forced.add(id(scope))
    return [scope for scope in scopes if not scope.is_root
            and (readers[id(scope)] > 1 or id(scope) in forced)]


def _computed(scope: Scope) -> set:
    """Return the names of a scope's result columns an AI function computes.

    A column passed through from another CTE's computed column counts.
    """
    names = set()
    for item in scope.expression.selects:
        value = item.unalias()
        if any(isinstance(node, exp.AIClassify) or (
                isinstance(node, exp.Anonymous)
                and str(node.this).upper() in AI_FUNCTIONS) for node in value.walk()):
            names.add(item.alias_or_name)
        else:
            for column in value.find_all(exp.Column):
                source = scope.sources.get(column.table)
                if isinstance(source, Scope) and column.name in _computed(source):
                    names.add(item.alias_or_name)
                    break
    return names


def _compile_select(tree: exp.Select, catalog: Catalog, prompts: PromptBinder,
                    dialect: SQLDialect, *,
                    allow_relational: bool = False) -> LogicalPlan:
    """Compile one SELECT against the tables in the catalog."""
    unnests = take_unnests(tree)
    limit = _parse_limit(tree)
    offset = _parse_offset(tree)
    distinct = _parse_distinct(tree)

    b = ExpressionBinder(catalog, prompts, dialect=dialect.value)

    for names, _, _ in unnests:
        for name in names:
            if name in b.unnested:
                raise CompileError(f"two UNNEST lists are named {name!r}")
            b.unnested[name] = list_fields()
    joins = JoinBinder(b)
    joins.bind_tables(tree)
    for name in b.unnested:
        if name in dict(b.tables):
            raise CompileError(
                f"{name!r} is both a table alias and an UNNEST list name; "
                f"rename one")
    bind_where(b, tree, joins)

    selection = SelectBinder(b).bind(tree)
    columns, aggregation = selection.columns, selection.aggregation
    columns = (*columns, *(Alias(b.bind_ai_sae_features(call), name)
                           for names, calls, _ in unnests
                           for name, call in zip(names, calls)))
    order = bind_order(tree, b, columns, aggregation, selection.structs,
                       selection.windows)
    _validate_query(b, joins, columns, selection.structs,
                    allow_relational=allow_relational)
    named, labels = _bind_label_names(b, columns)
    extracts = {}
    feature_columns = {}
    for column in columns:
        if aliased_call(column, "extract") is not None:
            extracts.setdefault(column.expression.aliases()[0], []).append(
                (column.expression, column.name)
            )
        if aliased_call(column, "features") is not None:
            feature_columns.setdefault(
                column.expression.aliases()[0], []).append(
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
            sae_features=tuple(feature_columns.get(alias, ())),
            schema=catalog.get(provider).schema(),
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
        unnests=tuple(names for names, _, _ in unnests),
        unnest_outer=tuple(outer for _, _, outer in unnests),
        unnest_tests=tuple(b.unnest_tests),
        unnest_aliases=tuple(b.unnest_aliases),
    )
