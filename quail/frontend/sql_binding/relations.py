"""Bind FROM, JOIN, and WHERE clauses."""

from dataclasses import dataclass, field

from sqlglot import exp

from quail.frontend.sql_binding.expressions import (
    JOIN_OPTION_KEYS,
    SCORE_OPTION_KEYS,
    ExpressionBinder,
    _conjuncts,
    _is_ai_score_comparison,
    _is_call,
    _label_test,
)
from quail.frontend.sql_binding.predicates import (
    COLUMN_NODES,
    ai_aliases,
    bind_condition,
    unwrap_hint,
)
from quail.logical import (
    CompileError,
    Equality,
    RegularPredicate,
    Scan,
)
from quail.logical.naive import Exists, Hint


def _parse_equality(b, term, joined_alias: str) -> Equality:
    """Parse one ordinary ON condition: two columns compared with =."""
    if not isinstance(term, exp.EQ) or not all(
            isinstance(side, COLUMN_NODES)
            for side in (term.this, term.expression)):
        raise CompileError(
            f"ON supports column = column and AI predicates, got {term.sql()}"
        )
    equality = Equality(b.resolve_value(term.this), b.resolve_value(term.expression))
    if joined_alias not in equality.aliases() or len(set(equality.aliases())) != 2:
        raise CompileError(
            f"JOIN {joined_alias} ON {term.sql()} must relate "
            f"{joined_alias!r} to a table already in the query"
        )
    return equality


def _from_clause(select):
    return select.args.get("from_")


@dataclass
class JoinBinder:
    """Bind table references and assign predicates to joins."""

    binder: ExpressionBinder
    aliases: list[str] = field(default_factory=list)
    conditions: dict[str, list[Equality]] = field(default_factory=dict)

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
                    predicates.append(_written(
                        *b.bind_ai_filter(term, JOIN_OPTION_KEYS, join=True)))
                elif _is_ai_score_comparison(term):
                    predicates.append(_written(
                        *b.bind_ai_score(term, SCORE_OPTION_KEYS, join=True)))
                else:
                    if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                        raise CompileError(
                            "AI.SCORE must be compared with <, <=, >, or >="
                        )
                    condition = (None if _has_ai_call(term)
                                 else bind_condition(b, term, {}))
                    if condition is not None:
                        b.parts.append(condition)
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
        # the ON predicates come before WHERE's, in JOIN order
        b.parts.extend(on_preds)


def _written(predicate, options, aliases):
    """Return a bound AI predicate with the options written on it."""
    return Hint(predicate, options.get("selectivity"), options.get("anchor"))


def _has_ai_call(term) -> bool:
    """Return whether a term asks the model, counting AI.EXTRACT only bare."""
    return any(
        _is_call(node, "AI_FILTER")
        or _is_call(node, "AI_SCORE")
        or isinstance(node, (exp.AIClassify, exp.Exists))
        or (_is_call(node, "AI_EXTRACT") and not isinstance(node.parent, exp.Dot))
        for node in term.walk()
    )


def _name_select_labels(b: ExpressionBinder, term, selected: dict):
    """Replace a bare column named after an AI.CLASSIFY in SELECT with the call.

    Raises:
        CompileError: The name is also a column of a table in scope.
    """
    for column in list(term.find_all(exp.Column)):
        if column.table or column.name not in selected:
            continue
        if not isinstance(selected[column.name], exp.AIClassify):
            continue
        if any(column.name in b.catalog.get(provider).columns
               for _, provider in b.tables):
            raise CompileError(
                f"{column.name!r} is both a column of a table in scope and "
                f"an AS name in SELECT; qualify or rename it")
        if column is term:
            return selected[column.name].copy()
        column.replace(selected[column.name].copy())
    return term


def _bind_plain_term(b: ExpressionBinder, term, selected: dict,
                     selectivity: float | None) -> bool:
    """Bind a term that asks no model as a condition, placed later.

    Args:
        b: The binder.
        term: The term.
        selected: The AI calls in SELECT, by AS name.
        selectivity: The term's SELECTIVITY hint, or None.

    Returns:
        Whether the term was a condition; False leaves an AI term, or a
        comparison of two columns, to the AI binders.
    """
    if _has_ai_call(term):
        return False
    condition = bind_condition(b, term, selected)
    if condition is None:
        return False
    b.parts.append(condition if selectivity is None
                   else Hint(condition, selectivity))
    return True


def bind_where(b: ExpressionBinder, tree: exp.Select) -> None:
    """Bind each WHERE term as a part of the query's condition, in written order.

    A part is a condition, a model call or a test of one, or an EXISTS
    gate; place_relation decides where each runs.
    """
    where = tree.args.get("where")
    selected = ai_aliases(tree)
    for term in _conjuncts(where.this) if where else []:
        term, hint = unwrap_hint(b, term)
        term = _name_select_labels(b, term, selected)
        if _bind_plain_term(b, term, selected, hint):
            continue
        if isinstance(term.unnest(), exp.Or):
            raise CompileError(
                f"OR combines tests of source columns; {term.sql()} asks "
                f"the model, so put the disjunction inside one prompt's text")
        anti = False
        node = term
        if isinstance(node, exp.Not):
            node, anti = node.this, True
        if isinstance(node, exp.Exists):
            b.parts.append(_bind_exists(b, node, anti))
            continue
        if anti:
            raise CompileError(
                f"NOT is only supported as NOT EXISTS, got NOT {node.sql()}"
            )
        tested = _label_test(node)
        if tested is not None:
            classify_node, items = tested
            call, options, _ = b.bind_ai_classify(classify_node)
            accepted = []
            for item in items:
                if not (isinstance(item, exp.Literal) and item.is_string):
                    raise CompileError("AI.CLASSIFY is compared with string literals")
                accepted.append(str(item.this))
            if len(set(accepted)) != len(accepted):
                raise CompileError("a filter on a label lists a label twice")
            b.parts.append(Hint(
                RegularPredicate(call, "in", tuple(accepted)),
                options.get("selectivity") if hint is None else hint))
            continue
        if any(isinstance(call, exp.AIClassify) for call in term.walk()):
            raise CompileError(
                "AI.CLASSIFY in WHERE is tested with = 'label' or IN ('label', ...)"
            )
        if any(_is_call(call, "AI_EXTRACT") for call in term.walk()):
            raise CompileError(
                "an AI.EXTRACT call is not a predicate; test one of its "
                "fields, such as AI.EXTRACT(...).end_date IS NOT NULL, or "
                "its AS name from SELECT")
        if _is_ai_score_comparison(term):
            b.parts.append(_written(*b.bind_ai_score(term, SCORE_OPTION_KEYS)))
        else:
            if any(_is_call(call, "AI_SCORE") for call in term.walk()):
                raise CompileError("AI.SCORE must be compared with <, <=, >, or >=")
            # a predicate over two tables is a join predicate, BigQuery
            # style: tables cross-joined in FROM, filtered here
            b.parts.append(_written(*b.bind_ai_filter(term, JOIN_OPTION_KEYS)))


def _bind_exists(b: ExpressionBinder, node: exp.Exists, anti: bool) -> Hint:
    """Bind EXISTS (SELECT 1 FROM t WHERE AI_FILTER(...)) as a gate part."""
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
    predicate, options, _ = b.bind_ai_filter(terms[0], JOIN_OPTION_KEYS, join=True)
    scan = Scan(provider=dict(b.tables)[alias], alias=alias, column="")
    return Hint(Exists(predicate, scan, anti), options.get("selectivity"),
                options.get("anchor"))
