"""The AI SQL front end: sqlglot (Snowflake dialect), a validator, a
binder.

sqlglot parses AI_FILTER(...) and PROMPT(...) as generic function
nodes (exp.Anonymous). We match them by name in the two legal
positions - WHERE conjuncts and JOIN ON - plus the [NOT] EXISTS form,
and reject everything else by node class: the rejection list below is
literally a list of forbidden sqlglot classes, so new SQL surface
cannot creep in silently.

Each multi-table AI_FILTER(PROMPT(...)) is one join predicate,
evaluated over the cross product of the tables its prompt references
with every document in one model call. A query may have several. A
predicate is written either Snowflake style (on a JOIN's ON; a bare
JOIN leaves its table to another predicate) or BigQuery style (tables
comma-joined or CROSS JOINed in FROM, the predicate a WHERE term):

    FROM reviews a JOIN threads b JOIN products c
      ON AI_FILTER(PROMPT('... {0} ... {1} ... {2}',
                          a.review, b.thread, c.description))

    FROM reviews a, threads b, products c
    WHERE AI_FILTER(PROMPT('... {0} ... {1}', a.review, b.thread))
      AND AI_FILTER(PROMPT('... {0} ... {1}', b.thread, c.description))

Coverage rule: every JOINed table must appear in at least one join
predicate, and the predicates' tables must form one connected graph
with the FROM table - otherwise some table would never be compared
with the rest of the query.
"""

import sqlglot
from sqlglot import exp

from quail.catalog import Catalog
from quail.logical import (ColumnRef, CompileError, FilterPredicate,
                           JoinSpec, LogicalPlan, QueryDesc,
                           assemble_plan, bind_join_prompt, bind_prompt)

# Every relational operator except the projection, named and refused.
# OR is rejected separately with its own message.
FORBIDDEN = (
    (exp.Group, "GROUP BY"),
    (exp.Order, "ORDER BY"),
    (exp.Distinct, "DISTINCT"),
    (exp.Having, "HAVING"),
    (exp.Qualify, "QUALIFY"),
    (exp.Window, "window functions"),
    (exp.Union, "UNION"),
    (exp.Except, "EXCEPT"),
    (exp.Intersect, "INTERSECT"),
    (exp.Offset, "OFFSET"),
)

# one option surface: anchor is rejected after parsing when the
# predicate turns out to be a one-provider filter
JOIN_OPTION_KEYS = {"selectivity", "anchor"}


def _is_call(node, name: str) -> bool:
    return (isinstance(node, exp.Anonymous)
            and str(node.this).upper() == name)


def _reject_forbidden(tree) -> None:
    for cls, name in FORBIDDEN:
        if list(tree.find_all(cls)):
            raise CompileError(
                f"{name} is outside the language: Quail runs the "
                f"semantic part; do the relational part in the "
                f"database the ids came from")
    if list(tree.find_all(exp.Or)):
        raise CompileError(
            "OR between AI predicates is not supported: a disjunction "
            "belongs inside one prompt's text, where the model "
            "evaluates it")
    for fn in tree.find_all(exp.Anonymous):
        name = str(fn.this).upper()
        if name.startswith("AI_") and name != "AI_FILTER":
            raise CompileError(f"{name} is not supported; AI_FILTER "
                               f"is the only AI function")
    # subqueries are legal only as the EXISTS form, checked
    # structurally; any other subquery is refused here
    for sub in tree.find_all(exp.Subquery):
        raise CompileError("subqueries other than "
                           "[NOT] EXISTS (SELECT 1 ...) are not "
                           "supported")


class _Binder:
    def __init__(self, catalog: Catalog, tokenizer):
        self.catalog = catalog
        self.tokenizer = tokenizer
        self.tables = []          # (alias, provider) in appearance order
        self.doc_columns = {}     # alias -> document column
        self.filters = {}         # alias -> [FilterPredicate]
        self.joins = []           # [JoinSpec]

    # ---- scope ------------------------------------------------------

    def add_table(self, table: exp.Table) -> str:
        name = table.name
        alias = table.alias or name
        self.catalog.get(name)    # unknown provider -> CompileError
        if alias in dict(self.tables):
            raise CompileError(f"duplicate table alias {alias!r}")
        self.tables.append((alias, name))
        return alias

    def resolve_column(self, col: exp.Column, scope=None) -> ColumnRef:
        scope = dict(self.tables) if scope is None else dict(scope)
        column = col.name
        if col.table:
            if col.table not in scope:
                raise CompileError(f"unknown table alias {col.table!r} "
                                   f"in column {col.sql()}")
            provider = scope[col.table]
            if column not in self.catalog.get(provider).columns:
                raise CompileError(
                    f"column {column!r} not in provider {provider!r} "
                    f"(schema: {self.catalog.get(provider).columns})")
            return ColumnRef(alias=col.table, provider=provider,
                             column=column)
        owners = [(a, p) for a, p in scope.items()
                  if column in self.catalog.get(p).columns]
        if len(owners) != 1:
            raise CompileError(
                f"column {column!r} is {'ambiguous' if owners else 'unknown'};"
                f" qualify it with a table alias")
        return ColumnRef(alias=owners[0][0], provider=owners[0][1],
                         column=column)

    def note_doc_column(self, ref: ColumnRef) -> None:
        seen = self.doc_columns.get(ref.alias)
        if seen and seen != ref.column:
            raise CompileError(
                f"alias {ref.alias!r} is referenced through two "
                f"columns ({seen!r}, {ref.column!r}); predicates over "
                f"one table must share one document column so its KV "
                f"is computed once")
        self.doc_columns[ref.alias] = ref.column

    # ---- AI_FILTER parsing -------------------------------------------

    def parse_options(self, node, allowed: set) -> dict:
        if node is None:
            return {}
        if not isinstance(node, exp.Struct):
            raise CompileError(
                f"the second argument to AI_FILTER must be an option "
                f"object like {{'selectivity': 0.3}}, got {node.sql()}")
        out = {}
        for prop in node.expressions:
            if not isinstance(prop, exp.PropertyEQ):
                raise CompileError(f"malformed option {prop.sql()}")
            key = str(prop.this.name)
            if key not in allowed:
                raise CompileError(
                    f"unknown option key {key!r}; allowed: "
                    f"{sorted(allowed)}")
            value = prop.expression
            if key == "selectivity":
                if not (isinstance(value, exp.Literal)
                        and not value.is_string):
                    raise CompileError("selectivity must be a number")
                out[key] = float(value.this)
            else:   # anchor
                if not (isinstance(value, exp.Literal)
                        and value.is_string):
                    raise CompileError("anchor must be a table alias "
                                       "string")
                out[key] = str(value.this)
        return out

    def parse_ai_filter(self, node, allowed: set, scope=None,
                        join=None):
        """(prompt, options, provider aliases referenced). join: True
        binds the join layout (a static question in the anchor frame
        and labeled partner blocks), False the filter layout, None
        decides by how many providers the prompt references."""
        if not _is_call(node, "AI_FILTER"):
            raise CompileError(
                f"only AI_FILTER(PROMPT(...)) predicates are "
                f"supported here, got: {node.sql()}")
        args = node.expressions
        if not args or not _is_call(args[0], "PROMPT"):
            raise CompileError("AI_FILTER's first argument must be "
                               "PROMPT('template', columns...)")
        if len(args) > 2:
            raise CompileError("AI_FILTER takes PROMPT and at most one "
                               "option object")
        options = self.parse_options(args[1] if len(args) == 2 else None,
                                     allowed)
        p_args = args[0].expressions
        if not p_args or not (isinstance(p_args[0], exp.Literal)
                              and p_args[0].is_string):
            raise CompileError("PROMPT's first argument must be a "
                               "string template")
        template = str(p_args[0].this)
        refs = []
        for a in p_args[1:]:
            if not isinstance(a, exp.Column):
                raise CompileError(f"PROMPT arguments must be column "
                                   f"references, got {a.sql()}")
            refs.append(self.resolve_column(a, scope))
        aliases = []
        for r in refs:
            if r.alias not in aliases:
                aliases.append(r.alias)
        if join is None:
            join = len(aliases) > 1
        binder = bind_join_prompt if join else bind_prompt
        prompt = binder(template, tuple(refs), self.tokenizer)
        for r in refs:
            self.note_doc_column(r)
        return prompt, options, aliases


def _from_clause(select):
    # sqlglot renamed the arg key "from" to "from_" across versions
    return select.args.get("from_") or select.args.get("from")


def _where_terms(where) -> list:
    """Flatten the WHERE conjunction; anything not reachable through
    AND alone is rejected by the term handlers."""
    if where is None:
        return []
    terms, stack = [], [where.this]
    while stack:
        n = stack.pop()
        if isinstance(n, exp.And):
            stack.append(n.expression)
            stack.append(n.this)
        else:
            terms.append(n)
    return terms       # the pop order above yields written order


def _parse_limit(tree) -> int | None:
    """Extract a plain LIMIT N from the parse tree. ORDER BY ... LIMIT
    is rejected by the FORBIDDEN list (ORDER BY is still forbidden), so
    this only handles the early-termination case."""
    limit_node = tree.args.get("limit")
    if limit_node is None:
        return None
    expr = limit_node.expression
    if not isinstance(expr, exp.Literal) or expr.is_string:
        raise CompileError("LIMIT must be a positive integer")
    value = int(expr.this)
    if value <= 0:
        raise CompileError("LIMIT must be a positive integer")
    return value


def compile_sql(sql: str, catalog: Catalog,
                tokenizer=None) -> LogicalPlan:
    """AI SQL text -> LogicalPlan, or CompileError. `tokenizer` is any
    callable text -> token list, used once at bind time to split and
    count each prompt's preamble and tail."""
    try:
        tree = sqlglot.parse_one(sql, dialect="snowflake")
    except sqlglot.errors.ParseError as e:
        raise CompileError(f"parse error: {e}") from e
    if not isinstance(tree, exp.Select):
        raise CompileError("the query must be a single SELECT")
    _reject_forbidden(tree)

    limit = _parse_limit(tree)

    b = _Binder(catalog, tokenizer)

    from_ = _from_clause(tree)
    if from_ is None or not isinstance(from_.this, exp.Table):
        raise CompileError("FROM must name one registered provider")
    b.add_table(from_.this)

    joined_aliases = []       # tables brought in by JOIN clauses
    on_preds = []
    for join in tree.args.get("joins") or []:
        if join.side or (join.kind
                         and join.kind.upper() not in ("INNER",
                                                       "CROSS")):
            raise CompileError(
                f"only plain JOIN is supported, got "
                f"{join.side or ''} {join.kind or ''} JOIN".strip())
        if not isinstance(join.this, exp.Table):
            raise CompileError("JOIN must name one registered provider")
        joined_aliases.append(b.add_table(join.this))
        on = join.args.get("on")
        if on is None:
            continue    # a bare/cross-joined table: some join
            #             predicate must cover it (checked below)
        on_preds.append(b.parse_ai_filter(on, JOIN_OPTION_KEYS,
                                          join=True))

    claimed = set()           # joined tables already carried by a spec

    def add_join_spec(prompt, options, aliases):
        joinable = {b.tables[0][0], *joined_aliases}
        outside = [a for a in aliases if a not in joinable]
        if outside:
            raise CompileError(
                f"the join prompt references {outside}, which are not "
                f"the FROM table or JOINed tables of this query "
                f"({sorted(joinable)})")
        anchor = options.get("anchor")
        if anchor is not None and anchor not in aliases:
            raise CompileError(
                f"anchor {anchor!r} is not a table of this join "
                f"({aliases})")
        # each spec carries the joined tables its prompt references
        # that no earlier spec carried, so assemble_plan folds every
        # table into the tree exactly once
        news = tuple(a for a in joined_aliases
                     if a in aliases and a not in claimed)
        claimed.update(news)
        b.joins.append(JoinSpec(aliases=news,
                                prompt=prompt, semantics="full",
                                selectivity=options.get("selectivity"),
                                anchor=anchor))

    for pred in on_preds:
        add_join_spec(*pred)

    for term in _where_terms(tree.args.get("where")):
        anti = False
        node = term
        if isinstance(node, exp.Not):
            node, anti = node.this, True
        if isinstance(node, exp.Exists):
            _compile_exists(b, node, anti)
            continue
        if anti:
            raise CompileError(f"NOT is only supported as NOT EXISTS, "
                               f"got NOT {node.sql()}")
        prompt, options, aliases = b.parse_ai_filter(
            term, JOIN_OPTION_KEYS)
        if len(aliases) == 1:
            if "anchor" in options:
                raise CompileError(
                    "anchor is a join option; a one-provider "
                    "AI_FILTER takes only selectivity")
            b.filters.setdefault(aliases[0], []).append(
                FilterPredicate(prompt=prompt,
                                selectivity=options.get("selectivity")))
            continue
        # a multi-provider WHERE predicate is a join predicate,
        # BigQuery style: tables cross-joined in FROM, filtered here
        add_join_spec(prompt, options, aliases)

    _check_join_coverage(b, joined_aliases)

    columns = _compile_projection(b, tree.expressions)

    if not b.joins and not b.filters:
        raise CompileError("the query has no AI predicate; a plain "
                           "scan belongs in the database the ids came "
                           "from")

    desc = QueryDesc(
        tables=tuple(b.tables),
        doc_columns=dict(b.doc_columns),
        filters={a: tuple(v) for a, v in b.filters.items()},
        joins=tuple(b.joins),
        columns=tuple(columns),
        limit=limit)
    return assemble_plan(desc)


def _check_join_coverage(b: _Binder, joined_aliases: list) -> None:
    """The coverage rule: every JOINed table must appear in at least
    one join predicate, and the predicates' tables must form one
    connected graph with the FROM table. A table outside that graph
    would never be compared with the rest of the query, so its cross
    product would pass through unfiltered."""
    if not joined_aliases:
        return
    preds = [{r.alias for r in j.prompt.args}
             for j in b.joins if j.semantics == "full"]
    uncovered = [a for a in joined_aliases
                 if not any(a in p for p in preds)]
    if uncovered:
        raise CompileError(
            f"JOINed tables {uncovered} appear in no join predicate: "
            f"every JOINed table must appear in at least one "
            f"AI_FILTER(PROMPT(...)) join predicate - on a JOIN's ON "
            f"or as a WHERE term")
    root = b.tables[0][0]
    reached = {root}
    grew = True
    while grew:
        grew = False
        for p in preds:
            if p & reached and not p <= reached:
                reached |= p
                grew = True
    disconnected = sorted(a for a in joined_aliases
                          if a not in reached)
    if disconnected:
        raise CompileError(
            f"join predicates do not connect {disconnected} to the "
            f"FROM table {root!r}: the predicates' tables must form "
            f"one connected graph with it")


def _compile_exists(b: _Binder, node: exp.Exists, anti: bool) -> None:
    inner = node.this
    if not isinstance(inner, exp.Select):
        raise CompileError("EXISTS must wrap SELECT 1 FROM provider "
                           "WHERE AI_FILTER(...)")
    inner_from = _from_clause(inner)
    if inner_from is None or not isinstance(inner_from.this, exp.Table):
        raise CompileError("the EXISTS subquery must scan one "
                           "registered provider")
    if (inner.args.get("joins") or inner.args.get("group")
            or len(inner.expressions) != 1):
        raise CompileError("the EXISTS subquery must be exactly "
                           "SELECT 1 FROM provider WHERE AI_FILTER(...)")
    alias = b.add_table(inner_from.this)
    terms = _where_terms(inner.args.get("where"))
    if len(terms) != 1:
        raise CompileError("the EXISTS subquery takes exactly one "
                           "AI_FILTER predicate")
    prompt, options, aliases = b.parse_ai_filter(terms[0],
                                                 JOIN_OPTION_KEYS,
                                                 join=True)
    if len(aliases) != 2 or alias not in aliases:
        raise CompileError(
            "the EXISTS predicate must reference the inner provider "
            "and exactly one outer provider")
    outer = next(a for a in aliases if a != alias)
    anchor = options.get("anchor")
    if anchor is not None and anchor != outer:
        raise CompileError(
            f"exists/anti always anchor on the outer table {outer!r} "
            f"- the gate applies to its documents - got anchor "
            f"{anchor!r}")
    b.joins.append(JoinSpec(aliases=(alias,), prompt=prompt,
                            semantics="anti" if anti else "exists",
                            selectivity=options.get("selectivity"),
                            anchor=outer))


def _compile_projection(b: _Binder, expressions) -> list:
    columns = []
    for e in expressions:
        if isinstance(e, exp.Star):
            for alias, provider in b.tables:
                for c in b.catalog.get(provider).columns:
                    columns.append(ColumnRef(alias=alias,
                                             provider=provider,
                                             column=c))
            continue
        if isinstance(e, exp.Alias):
            e = e.this
        if not isinstance(e, exp.Column):
            raise CompileError(
                f"the SELECT list is column selection only, got "
                f"{e.sql()}: nothing computed, per the projection "
                f"contract")
        columns.append(b.resolve_column(e))
    return columns
