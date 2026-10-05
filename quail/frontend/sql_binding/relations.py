"""Bind FROM, JOIN, and WHERE clauses."""

from dataclasses import dataclass, field

from sqlglot import exp

from quail.frontend.sql_binding.expressions import (
    JOIN_OPTION_KEYS,
    SCORE_OPTION_KEYS,
    ExpressionBinder,
    _column_predicate,
    _conjuncts,
    _is_ai_score_comparison,
    _is_call,
    _label_test,
)
from quail.logical import (
    Alias,
    CompileError,
    Equality,
    FilterPredicate,
    JoinSpec,
)


def _parse_equality(b, term, joined_alias: str) -> Equality:
    """Parse one ordinary ON condition: two columns compared with =."""
    if (
        not isinstance(term, exp.EQ)
        or not isinstance(term.this, exp.Column)
        or not isinstance(term.expression, exp.Column)
    ):
        raise CompileError(
            f"ON supports column = column and AI predicates, got {term.sql()}"
        )
    left = b.resolve_column(term.this)
    right = b.resolve_column(term.expression)
    sides = {left.alias, right.alias}
    if joined_alias not in sides or len(sides) != 2:
        raise CompileError(
            f"JOIN {joined_alias} ON {term.sql()} must relate "
            f"{joined_alias!r} to a table already in the query"
        )
    return Equality(left, right)


def _from_clause(select):
    return select.args.get("from_")


@dataclass
class JoinBinder:
    """Bind table references and assign predicates to joins."""

    binder: ExpressionBinder
    aliases: list[str] = field(default_factory=list)
    conditions: dict[str, list[Equality]] = field(default_factory=dict)
    claimed: set[str] = field(default_factory=set)

    def bind_tables(self, tree: exp.Select) -> None:
        """Bind FROM and JOIN clauses in table order."""
        b = self.binder
        from_ = _from_clause(tree)
        if from_ is None or not isinstance(from_.this, exp.Table):
            raise CompileError("FROM must name one registered provider")
        b.add_table(from_.this)

        on_preds = []
        for join in tree.args.get("joins") or []:
            if join.side or (join.kind and join.kind.upper() not in ("INNER", "CROSS")):
                raise CompileError(
                    f"only plain JOIN is supported, got "
                    f"{join.side or ''} {join.kind or ''} JOIN".strip()
                )
            if not isinstance(join.this, exp.Table):
                raise CompileError("JOIN must name one registered provider")
            alias = b.add_table(join.this)
            self.aliases.append(alias)
            on = join.args.get("on")
            if on is None:
                continue  # a bare/cross-joined table: some join
                #             predicate must cover it (checked below)
            # ON: equalities choose the pairs; at most one AI predicate
            equalities, predicates = [], []
            for term in _conjuncts(on):
                if _is_call(term, "AI_FILTER"):
                    predicates.append(
                        b.bind_ai_filter(term, JOIN_OPTION_KEYS, join=True)
                    )
                elif _is_ai_score_comparison(term):
                    predicates.append(
                        b.bind_ai_score(term, SCORE_OPTION_KEYS, join=True)
                    )
                else:
                    if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                        raise CompileError(
                            "AI.SCORE must be compared with <, <=, >, or >="
                        )
                    plain = _column_predicate(b, term)
                    if plain is not None:
                        b.column_predicates.setdefault(plain.column.alias, []).append(
                            plain
                        )
                        continue
                    equalities.append(_parse_equality(b, term, alias))
            if len(predicates) > 1:
                raise CompileError(
                    f"JOIN {alias} has {len(predicates)} AI predicates in "
                    f"ON; ask one question per JOIN"
                )
            if equalities:
                self.conditions[alias] = equalities
            on_preds.extend(predicates)

        for predicate in on_preds:
            self.add_predicate(*predicate)

    def add_predicate(self, predicate, options, aliases) -> None:
        """Assign a join predicate to the tables it introduces."""
        b = self.binder
        joinable = {b.tables[0][0], *self.aliases}
        outside = [a for a in aliases if a not in joinable]
        if outside:
            raise CompileError(
                f"the join prompt references {outside}, which are not "
                f"the FROM table or JOINed tables of this query "
                f"({sorted(joinable)})"
            )
        anchor = options.get("anchor")
        if anchor is not None and anchor not in aliases:
            raise CompileError(
                f"anchor {anchor!r} is not a table of this join ({aliases})"
            )
        # each spec carries the joined tables its prompt references
        # that no earlier spec carried, so the logical builder adds
        # every table to the join tree exactly once
        news = tuple(a for a in self.aliases if a in aliases and a not in self.claimed)
        self.claimed.update(news)
        on = tuple(condition for a in news for condition in self.conditions.pop(a, ()))
        if on and len(aliases) != 2:
            raise CompileError(
                f"join conditions are supported on two-table AI "
                f"predicates; this one names {aliases}"
            )
        for condition in on:
            if not set(condition.aliases()) <= set(aliases):
                raise CompileError(
                    f"join condition {condition} names a table the AI "
                    f"predicate over {aliases} does not"
                )
        b.joins.append(
            JoinSpec(
                aliases=news,
                predicate=predicate,
                semantics="full",
                selectivity=options.get("selectivity"),
                anchor=anchor,
                on=on,
            )
        )


def bind_where(b: ExpressionBinder, tree: exp.Select, joins: JoinBinder) -> None:
    """Bind WHERE conditions to scans and joins."""
    where = tree.args.get("where")
    for term in _conjuncts(where.this) if where else []:
        plain = _column_predicate(b, term)
        if plain is not None:
            b.column_predicates.setdefault(plain.column.alias, []).append(plain)
            continue
        anti = False
        node = term
        if isinstance(node, exp.Not):
            node, anti = node.this, True
        if isinstance(node, exp.Exists):
            _compile_exists(b, node, anti)
            continue
        if anti:
            raise CompileError(
                f"NOT is only supported as NOT EXISTS, got NOT {node.sql()}"
            )
        tested = _label_test(node)
        if tested is not None:
            classify_node, items = tested
            call, options, aliases = b.bind_ai_classify(classify_node)
            if len(aliases) != 1:
                raise CompileError(
                    f"a filter on a label tests a one-document "
                    f"classification; this one classifies pairs of {aliases}"
                )
            accepted = []
            for item in items:
                if not (isinstance(item, exp.Literal) and item.is_string):
                    raise CompileError("AI.CLASSIFY is compared with string literals")
                accepted.append(str(item.this))
            if len(set(accepted)) != len(accepted):
                raise CompileError("a filter on a label lists a label twice")
            b.label_tests.setdefault(aliases[0], []).append(
                (call, tuple(accepted), options.get("selectivity"))
            )
            continue
        if any(isinstance(call, exp.AIClassify) for call in term.walk()):
            raise CompileError(
                "AI.CLASSIFY in WHERE is tested with = 'label' or IN ('label', ...)"
            )
        if _is_ai_score_comparison(term):
            predicate, options, aliases = b.bind_ai_score(term, SCORE_OPTION_KEYS)
        else:
            if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                raise CompileError("AI.SCORE must be compared with <, <=, >, or >=")
            predicate, options, aliases = b.bind_ai_filter(term, JOIN_OPTION_KEYS)
        if len(aliases) == 1:
            if "anchor" in options:
                raise CompileError(
                    "anchor is a join option; a one-provider "
                    "AI_FILTER takes only selectivity"
                )
            b.filters.setdefault(aliases[0], []).append(
                FilterPredicate(
                    expression=predicate, selectivity=options.get("selectivity")
                )
            )
            continue
        # a multi-provider WHERE predicate is a join predicate,
        # BigQuery style: tables cross-joined in FROM, filtered here
        joins.add_predicate(predicate, options, aliases)


def check_join_coverage(
    b: ExpressionBinder,
    joined_aliases: list,
    projected_scores: tuple[Alias, ...] = (),
) -> None:
    """Check that the join predicates cover and connect every JOINed table.

    Every JOINed table must appear in a join predicate, and all
    predicates must form one connected graph with the FROM table.
    """
    if not joined_aliases:
        return
    preds = [{r.alias for r in j.prompt.args} for j in b.joins if j.semantics == "full"]
    preds.extend(
        set(score.expression.aliases())
        for score in projected_scores
        if len(score.expression.aliases()) > 1
    )
    uncovered = [a for a in joined_aliases if not any(a in p for p in preds)]
    if uncovered:
        raise CompileError(
            f"JOINed tables {uncovered} appear in no join predicate: "
            f"every JOINed table must appear in at least one "
            f"AI predicate on a JOIN's ON "
            f"or as a WHERE term"
        )
    root = b.tables[0][0]
    reached = {root}
    grew = True
    while grew:
        grew = False
        for p in preds:
            if p & reached and not p <= reached:
                reached |= p
                grew = True
    disconnected = sorted(a for a in joined_aliases if a not in reached)
    if disconnected:
        raise CompileError(
            f"join predicates do not connect {disconnected} to the "
            f"FROM table {root!r}: the predicates' tables must form "
            f"one connected graph with it"
        )


def _compile_exists(b: ExpressionBinder, node: exp.Exists, anti: bool) -> None:
    inner = node.this
    if not isinstance(inner, exp.Select):
        raise CompileError(
            "EXISTS must wrap SELECT 1 FROM provider WHERE AI_FILTER(...)"
        )
    inner_from = _from_clause(inner)
    if inner_from is None or not isinstance(inner_from.this, exp.Table):
        raise CompileError("the EXISTS subquery must scan one registered provider")
    if (
        inner.args.get("joins")
        or inner.args.get("group")
        or len(inner.expressions) != 1
    ):
        raise CompileError(
            "the EXISTS subquery must be exactly "
            "SELECT 1 FROM provider WHERE AI_FILTER(...)"
        )
    alias = b.add_table(inner_from.this)
    where = inner.args.get("where")
    terms = _conjuncts(where.this) if where else []
    if len(terms) != 1:
        raise CompileError("the EXISTS subquery takes exactly one AI_FILTER predicate")
    predicate, options, aliases = b.bind_ai_filter(
        terms[0], JOIN_OPTION_KEYS, join=True
    )
    if len(aliases) != 2 or alias not in aliases:
        raise CompileError(
            "the EXISTS predicate must reference the inner provider "
            "and exactly one outer provider"
        )
    outer = next(a for a in aliases if a != alias)
    anchor = options.get("anchor")
    if anchor is not None and anchor != outer:
        raise CompileError(
            f"exists/anti always anchor on the outer table {outer!r} "
            f"- the gate applies to its documents - got anchor "
            f"{anchor!r}"
        )
    b.joins.append(
        JoinSpec(
            aliases=(alias,),
            predicate=predicate,
            semantics="anti" if anti else "exists",
            selectivity=options.get("selectivity"),
            anchor=outer,
        )
    )
