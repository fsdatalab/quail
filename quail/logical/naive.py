"""The relation tree a frontend binds, and the rule batch that places it.

The SQL frontend resolves names and writes the FROM, JOIN, and WHERE
part of a query as one tree, without deciding where anything runs: a
Scan per table, a Join per JOIN with its ON equalities, and one Filter
above them whose condition holds every ON and WHERE part as written,
model calls included. place_relation reads that tree and makes each
decision about where a part runs, as Spark's optimizer rules do after
its analyzer:

- An AI.IF or AI.SCORE test of one table joins that table's
  SemanticFilter; one of two tables becomes a SemanticJoin, with the
  ON equalities of the tables it brings in.
- An EXISTS or NOT EXISTS over a model call becomes a gate join.
- A test of a label, an answer, or a source column goes where
  place_conditions puts it.

For example, ``SELECT r.id FROM reviews r JOIN products p ON r.asin =
p.asin AND AI.IF(...r.review, p.description...) WHERE r.stars > 3``
binds to ``Filter(AI.IF(...) AND r.stars > 3)`` over ``Join(Scan(r),
Scan(p), on=r.asin = p.asin)``; place_relation puts ``r.stars > 3`` in
the Scan of ``r`` and the AI.IF in a SemanticJoin over the Join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from quail.logical.builder import JoinSpec, LogicalPlanBuilder
from quail.logical.expressions import (
    SPAN_SUFFIX,
    And,
    ColumnRef,
    CompileError,
    FilterPredicate,
    ModelCall,
    RegularPredicate,
    aliased_call,
    is_score,
)
from quail.logical.nodes import Filter, Join, Scan
from quail.logical.placement import place_conditions


@dataclass(frozen=True)
class Hint:
    """A part of a condition with the options written on it.

    Attributes:
        condition: The part as written.
        selectivity: The fraction of rows or pairs it keeps, or None.
        anchor: The table whose KV a join keeps, or None.
    """

    condition: Any
    selectivity: Optional[float] = None
    anchor: Optional[str] = None


@dataclass(frozen=True)
class Exists:
    """An EXISTS or NOT EXISTS gate: some row of a table makes a model call true.

    Attributes:
        call: The AI.IF call over the outer table and ``relation``.
        relation: The Scan of the subquery's table.
        anti: Whether the gate keeps the outer rows no row makes true.
    """

    call: ModelCall
    relation: Scan
    anti: bool = False


def relation_tree(scans, equalities, parts):
    """Return the tree a frontend binds for the FROM, JOIN, and WHERE part.

    Args:
        scans: The Scan of each table in FROM and JOIN order, FROM first.
        equalities: alias -> the ON equalities of the JOIN of that table.
        parts: The ON and WHERE parts in written order, the ON parts
            first: conditions, model calls and tests of them, Exists
            gates, each in a Hint when options are written on it.
    """
    node = scans[0]
    for scan in scans[1:]:
        node = Join(node, scan, on=tuple(equalities.get(scan.alias, ())))
    if not parts:
        return node
    return Filter(node, parts[0] if len(parts) == 1 else And(tuple(parts)))


@dataclass
class PlacedRelation:
    """The placed tables and joins, and what the result rows still test.

    Attributes:
        builder: The LogicalPlanBuilder holding the placed tree; its
            project() builds the rest of the plan.
        rows: The conditions on result rows, for project().
        filters: alias -> the table's AI.IF and AI.SCORE predicates.
        joins: The JoinSpecs, in written order.
        labels: alias -> the label tests of the table.
    """

    builder: LogicalPlanBuilder
    rows: tuple = ()
    filters: dict = field(default_factory=dict)
    joins: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)


def place_relation(relation, columns, catalog, unnested=(), *,
                   allow_relational: bool = False) -> PlacedRelation:
    """Place each part of a bound relation tree and build the logical plan below it.

    Args:
        relation: The tree relation_tree returns.
        columns: The projected columns, with their model calls.
        catalog: The registered tables, for their schemas and columns.
        unnested: The aliases of UNNEST relations.
        allow_relational: Whether a query with no model call is allowed,
            as a stage of a larger statement is.

    Raises:
        CompileError: A part cannot be placed: an anchor on a one-table
            predicate, a label test of a pair classification, an ON
            equality with no AI predicate over its pairs, a JOINed table
            that no predicate connects, AI.SCORE mixed with generative
            predicates, or no model call at all.
    """
    condition = relation.condition if isinstance(relation, Filter) else None
    scans = _scans(relation.input if isinstance(relation, Filter) else relation)
    equalities = _equalities(relation)
    parts = (() if condition is None else condition.operands
             if isinstance(condition, And) else (condition,))
    joined = [scan.alias for scan in scans[1:]]
    filters, joins, label_tests, conditions = {}, [], {}, []
    claimed = set()
    for part in parts:
        hint = part if isinstance(part, Hint) else Hint(part)
        body = hint.condition
        if isinstance(body, Exists):
            scans.append(body.relation)
            joins.append(_gate(body, hint))
        elif not _asks_model(body):
            conditions.append((body, hint.selectivity))
        elif _is_label_test(body):
            (alias,) = _one_alias(body.expression, "a filter on a label tests a "
                                  "one-document classification; this one "
                                  "classifies pairs of {}")
            label_tests.setdefault(alias, []).append(
                (body.expression, body.value, hint.selectivity))
        else:
            aliases = _call(body).aliases()
            if len(aliases) == 1:
                if hint.anchor is not None:
                    raise CompileError(
                        "anchor is a join option; a one-provider "
                        "AI_FILTER takes only selectivity")
                filters.setdefault(aliases[0], []).append(
                    FilterPredicate(expression=body, selectivity=hint.selectivity))
            else:
                joins.append(_join(body, hint, aliases, scans[0].alias, joined,
                                   claimed, equalities))
    projected = tuple(column for column in columns
                      if aliased_call(column) is not None)
    _check_joins(equalities, scans[0].alias, joined, joins, projected)
    _check_calls(filters, joins, label_tests, projected,
                 allow_relational=allow_relational)
    named, labels = label_names(label_tests, filters, projected)
    extracts, features = {}, {}
    for column in projected:
        call = column.expression
        if call.kind == "extract":
            extracts.setdefault(call.aliases()[0], []).append((call, column.name))
        if call.kind == "features":
            features.setdefault(call.aliases()[0], []).append((call, column.name))
    providers = {scan.alias: scan.provider for scan in scans}
    # a test of a label is a condition on the column its classification adds
    conditions.extend(
        (RegularPredicate(ColumnRef(alias, providers[alias], labels[alias][call]),
                          "in", tuple(accepted)), selectivity)
        for alias, tests in label_tests.items()
        for call, accepted, selectivity in tests)
    computed = {(alias, column): name for alias, calls in extracts.items()
                for _, name in calls for column in (name, name + SPAN_SUFFIX)}
    computed.update({(alias, name): name for alias, named_calls in labels.items()
                     for name in named_calls.values()})
    placed = place_conditions(conditions, computed, unnested)
    builder = LogicalPlanBuilder()
    for scan in scans:
        builder.add_scan(
            scan.alias,
            scan.provider,
            scan.column,
            tuple(filters.get(scan.alias, ())),
            regular_predicates=placed.scans.get(scan.alias, ()),
            labels=tuple(labels.get(scan.alias, {}).items()),
            extracts=tuple(extracts.get(scan.alias, ())),
            computed_filters=placed.computed.get(scan.alias, ()),
            sae_features=tuple(features.get(scan.alias, ())),
            schema=catalog.get(scan.provider).schema(),
        )
    for join in joins:
        builder.add_join(join)
    for alias in joined:
        if alias not in claimed:
            builder.add_cross_join(alias)
    for call, name in named.items():
        if len(call.aliases()) == 2:
            builder.add_classify(call, name)
    return PlacedRelation(builder, placed.rows, filters, joins, label_tests)


def label_names(label_tests, filters, projected) -> tuple[dict, dict]:
    """Name each classification that SELECT returns or WHERE tests.

    A classification SELECT returns takes its AS name. One only tested
    is named after its first test's position among the table's
    filters, as ``__label_<alias>_<position>``.

    Returns:
        (call -> name for the classifications SELECT returns,
        alias -> call -> name for every one-table classification).
    """
    named = {column.expression: column.name for column in projected
             if column.expression.kind == "label"}
    labels = {}
    for alias, tests in label_tests.items():
        asks = len(filters.get(alias, ()))
        for index, (call, _, _) in enumerate(tests):
            labels.setdefault(alias, {}).setdefault(
                call, named.get(call, f"__label_{alias}_{asks + index}"))
    for call, name in named.items():
        if len(call.aliases()) == 1:
            labels.setdefault(call.aliases()[0], {}).setdefault(call, name)
    return named, labels


def _scans(node) -> list:
    """Return the Scans of a Join tree, left to right."""
    if isinstance(node, Join):
        return [*_scans(node.left), *_scans(node.right)]
    return [node]


def _equalities(relation) -> dict:
    """Return alias -> the ON equalities of the JOIN that brings the table in."""
    node = relation.input if isinstance(relation, Filter) else relation
    found = {}
    while isinstance(node, Join):
        if node.on:
            found[node.right.alias] = list(node.on)
        node = node.left
    return found


def _call(body) -> ModelCall:
    return body.expression if isinstance(body, RegularPredicate) else body


def _asks_model(body) -> bool:
    return isinstance(body, ModelCall) or (
        isinstance(body, RegularPredicate) and isinstance(body.expression, ModelCall))


def _is_label_test(body) -> bool:
    return (isinstance(body, RegularPredicate)
            and isinstance(body.expression, ModelCall)
            and body.expression.kind == "label")


def _one_alias(call, message: str) -> tuple[str]:
    aliases = call.aliases()
    if len(aliases) != 1:
        raise CompileError(message.format(aliases))
    return aliases


def _gate(gate: Exists, hint: Hint) -> JoinSpec:
    """Return the gate join an EXISTS or NOT EXISTS over a model call becomes."""
    inner = gate.relation.alias
    aliases = gate.call.aliases()
    if len(aliases) != 2 or inner not in aliases:
        raise CompileError(
            "the EXISTS predicate must reference the inner provider "
            "and exactly one outer provider")
    outer = next(alias for alias in aliases if alias != inner)
    if hint.anchor is not None and hint.anchor != outer:
        raise CompileError(
            f"exists/anti always anchor on the outer table {outer!r} "
            f"- the gate applies to its documents - got anchor "
            f"{hint.anchor!r}")
    return JoinSpec(aliases=(inner,), predicate=gate.call,
                    semantics="anti" if gate.anti else "exists",
                    selectivity=hint.selectivity, anchor=outer)


def _join(body, hint: Hint, aliases, base: str, joined: list, claimed: set,
          equalities: dict) -> JoinSpec:
    """Return the join a model call over two or more tables becomes.

    The join brings in the JOINed tables its prompt reads that no
    earlier join brought in, with their ON equalities.
    """
    joinable = {base, *joined}
    outside = [alias for alias in aliases if alias not in joinable]
    if outside:
        raise CompileError(
            f"the join prompt references {outside}, which are not "
            f"the FROM table or JOINed tables of this query "
            f"({sorted(joinable)})")
    if hint.anchor is not None and hint.anchor not in aliases:
        raise CompileError(
            f"anchor {hint.anchor!r} is not a table of this join ({aliases})")
    news = tuple(alias for alias in joined if alias in aliases and alias not in claimed)
    claimed.update(news)
    on = tuple(condition for alias in news
               for condition in equalities.pop(alias, ()))
    if on and len(aliases) != 2:
        raise CompileError(
            f"join conditions are supported on two-table AI "
            f"predicates; this one names {aliases}")
    for condition in on:
        if not set(condition.aliases()) <= set(aliases):
            raise CompileError(
                f"join condition {condition} names a table the AI "
                f"predicate over {aliases} does not")
    return JoinSpec(aliases=news, predicate=body, semantics="full",
                    selectivity=hint.selectivity, anchor=hint.anchor, on=on)


def _check_joins(equalities, root: str, joined, joins, projected) -> None:
    """Refuse JOINed tables no predicate asks about or connects.

    Raises:
        CompileError: An ON equality has no AI predicate over its
            pairs, a JOINed table appears in no join predicate, or the
            predicates do not connect a table to the FROM table.
    """
    scores = [column for column in projected if column.expression.kind == "score"]
    for alias, conditions in equalities.items():
        if any(alias in score.expression.aliases() for score in scores):
            raise CompileError(
                f"JOIN {alias} ON {conditions[0]} projects a pair "
                f"AI.SCORE; a projected pair score runs over CROSS JOIN, "
                f"and an ON equality is supported when the score is "
                f"compared in ON")
        raise CompileError(
            f"JOIN {alias} ON {conditions[0]} has no AI predicate over "
            f"its pairs; a plain join belongs in the database the ids "
            f"came from")
    if not joined:
        return
    predicates = [{reference.alias for reference in join.prompt.args}
                  for join in joins if join.semantics == "full"]
    predicates.extend(set(score.expression.aliases()) for score in scores
                      if len(score.expression.aliases()) > 1)
    uncovered = [alias for alias in joined
                 if not any(alias in found for found in predicates)]
    if uncovered:
        raise CompileError(
            f"JOINed tables {uncovered} appear in no join predicate: "
            f"every JOINed table must appear in at least one "
            f"AI predicate on a JOIN's ON "
            f"or as a WHERE term")
    reached = {root}
    grew = True
    while grew:
        grew = False
        for found in predicates:
            if found & reached and not found <= reached:
                reached |= found
                grew = True
    disconnected = sorted(alias for alias in joined if alias not in reached)
    if disconnected:
        raise CompileError(
            f"join predicates do not connect {disconnected} to the "
            f"FROM table {root!r}: the predicates' tables must form "
            f"one connected graph with it")


def _check_calls(filters, joins, label_tests, projected, *,
                 allow_relational: bool) -> None:
    """Refuse AI.SCORE mixed with generative predicates, and a query with no model call.

    Raises:
        CompileError: Some predicates compare AI.SCORE and others ask
            the model a question, or nothing asks the model.
    """
    flags = ([is_score(predicate.expression)
              for predicates in filters.values() for predicate in predicates]
             + [is_score(join.predicate) for join in joins]
             + [True for column in projected if column.expression.kind == "score"])
    if any(flags) and not all(flags):
        raise CompileError(
            "AI.SCORE cannot be mixed with generative AI predicates in one query")
    if not (allow_relational or joins or filters or label_tests or projected):
        raise CompileError(
            "the query has no AI predicate; a plain "
            "scan belongs in the database the ids came "
            "from")
