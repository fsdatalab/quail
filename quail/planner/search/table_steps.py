"""The steps of one table's sequence, priced for ordering.

A step is an AI.IF ask, a one-table classification with the label
filters that test it, or an extraction with the filters that test its
answers. The order search (quail.planner.search.filter_order) ranks the
steps by cost per rejected document, pricing a step as reading the
documents' KV resident when the step before it leaves that KV in the
pipeline, and as reading the documents afresh otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from itertools import groupby
from typing import Any, Callable

from quail.logical import (
    Apply,
    Filter,
    LogicalPlan,
    SemanticClassify,
    SemanticExtract,
    SemanticFilter,
    SemanticJoin,
    effective_selectivity,
    model_call,
)
from quail.planner.operators.classify import classify_builder, classify_scoring
from quail.planner.operators.extract import extract_spec
from quail.planner.operators.filter import prepare_filter_costs
from quail.planner.search.filter_order import order_filters_indexed
from quail.planner.validation import ClassifyRefusedError

ASK, CLASSIFY, EXTRACT = "ask", "classify", "extract"
APPLY, TEST = "apply", "test"


@dataclass(frozen=True)
class TableStep:
    """One step of a table's sequence and its costs.

    Attributes:
        kind: ASK, CLASSIFY, EXTRACT, APPLY, or TEST.
        key: The ask's written position, the classification's call, or
            the extraction's name, the Apply node, or the ColumnTest.
        selectivity: The fraction of documents the step keeps: the
            product of its tests' selectivities, 1 without a test.
        first: Seconds at a live count, the documents read afresh.
        later: Seconds at a live count, the documents' KV resident.
        reuse_key: Identity of a result an earlier step computed, or None.
        head: The token ids the step shows the document under.
        tests: Tests attached to a model call, or None for all its tests.
        joins: Written positions of joins that must precede the step.
        position: The step's position in the logical traversal.
    """

    kind: str
    key: Any
    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Any = None
    head: Any = None
    tests: tuple | None = None
    joins: frozenset[int] = frozenset()
    position: int = 0


def resident_after(previous: TableStep, step: TableStep) -> bool:
    """Return whether a step reads the KV the step before it left.

    The steps under one head run in one pipeline: each rewinds the KV
    past what the step before it appended and asks its question
    (the can_follow rules of the physical nodes).
    """
    return previous.joins == step.joins and previous.head == step.head


def ready_table_steps(pending, completed):
    """Remove and yield consecutive table steps whose input joins are complete."""
    ready = sorted(
        ((step.position, alias, step)
         for alias, steps in pending.items() for step in steps
         if step.joins <= completed), key=lambda item: item[0])
    for alias, group in groupby(ready, key=lambda item: item[1]):
        steps = [item[2] for item in group]
        for step in steps:
            pending[alias].remove(step)
        yield alias, steps


class PlanPricing:
    """Price a table's steps with the node estimators of the physical planner.

    Args:
        context: The planning context, with the model, device, and
            tokenizer.
        statistics: The plan's PlanStatistics.
        alias: The table.
    """

    def __init__(self, context, statistics, alias: str):
        self.context = context
        self.statistics = statistics
        self.alias = alias

    def asks(self, predicates) -> tuple:
        """Return the PredicateCosts of the table's AI.IF predicates."""
        stats = self.statistics.stats[self.alias]
        return prepare_filter_costs(
            predicates, prefix_tokens=self.statistics.pre + stats.mean_doc_tokens,
            model=self.context.model, device=self.context.device,
            chunk_tokens=self.statistics.chunk, context=self.context)

    def classify(self, call, name: str) -> tuple:
        """Return a classification's (first, later) costs at a live count."""
        return tuple(partial(_classify_seconds, self.context, self.alias, call,
                             name, resident) for resident in (False, True))

    def extract(self, node) -> tuple:
        """Return an extraction's (first, later) costs at a live count."""
        return tuple(partial(_extract_seconds, self.context, node, resident)
                     for resident in (False, True))

    def head(self, prompt):
        """Return the token ids a prompt shows the document under."""
        return tuple(self.context.lowered(prompt).preamble_token_ids)


def table_steps(operators, alias: str, asks, pricing) -> list[TableStep]:
    """Return a table's steps in written order, priced.

    Args:
        operators: The plan's Operators.
        alias: The table.
        asks: The written positions of the table's AI.IF predicates.
        pricing: Prices the steps: asks(predicates), classify(call,
            name), extract(node), and head(prompt), as PlanPricing does.
    """
    steps = []
    predicates = operators.filters.get(alias, ())
    costs = pricing.asks([predicates[position] for position in asks])
    for position, cost in zip(asks, costs):
        steps.append(TableStep(ASK, position, cost.selectivity, cost.first,
                               cost.later, cost.reuse_key,
                               head=pricing.head(predicates[position].prompt)))
    tests = {}
    for test in operators.label_filters.get(alias, ()):
        tests.setdefault(test.call, []).append(test.selectivity)
    for call, owner in operators.labels.calls:
        if owner != alias or len(call.aliases()) != 1:
            continue
        first, later = pricing.classify(call, operators.labels.names[call])
        steps.append(TableStep(CLASSIFY, call, _survival(tests.get(call, ())),
                               first, later, head=pricing.head(call.prompt)))
    tests = {}
    for test in operators.extract_filters.get(alias, ()):
        tests.setdefault(test.name, []).append(test.selectivity)
    for node in operators.extracts:
        if node.alias != alias:
            continue
        first, later = pricing.extract(node)
        steps.append(TableStep(EXTRACT, node.name, _survival(tests.get(node.name, ())),
                               first, later, head=pricing.head(node.call.prompt)))
    return _ordered_nodes(operators, alias, steps)


def _zero_cost(_count):
    return 0.0


def _ordered_nodes(operators, alias, steps):
    """Bind table steps to their logical positions around user functions."""
    if not any(node.ids != "pairs" for node in operators.applies):
        return steps
    by_key = {(step.kind, step.key): step for step in steps}
    tests = list((*operators.label_filters.get(alias, ()),
                  *operators.extract_filters.get(alias, ())))
    ordered, group = [], []
    join_positions = {id(node): index for index, node in enumerate(operators.joins)}

    def located(step, node, written):
        joins = frozenset(join_positions[id(child)]
                          for child in LogicalPlan(node.input).walk()
                          if isinstance(child, SemanticJoin))
        return replace(step, joins=joins, position=written)

    def append(step, node, written):
        step = located(step, node, written)
        if group and group[-1].joins != step.joins:
            ordered.extend(_group_tests(group))
            group.clear()
        group.append(step)

    position = 0
    for written, node in enumerate(operators.nodes):
        if isinstance(node, Apply):
            ordered.extend(_group_tests(group))
            group = []
            if node.ids != "pairs" and node.aliases == (alias,):
                ordered.append(located(
                    TableStep(APPLY, node, 1.0, _zero_cost, _zero_cost), node, written))
        elif isinstance(node, SemanticFilter):
            for predicate in node.predicates:
                if model_call(predicate.expression).aliases() != (alias,):
                    continue
                step = by_key.get((ASK, position))
                if step is not None:
                    append(step, node, written)
                position += 1
        elif isinstance(node, SemanticClassify) and node.call.aliases() == (alias,):
            append(replace(by_key[(CLASSIFY, node.call)],
                           selectivity=1.0, tests=()), node, written)
        elif isinstance(node, SemanticExtract) and node.alias == alias:
            append(replace(by_key[(EXTRACT, node.name)],
                           selectivity=1.0, tests=()), node, written)
        elif isinstance(node, Filter) and node.condition.column.alias == alias:
            test = next(test for test in tests if test.condition == node.condition
                        and test.selectivity == node.selectivity)
            tests.remove(test)
            append(TableStep(TEST, test, effective_selectivity(test.selectivity),
                             _zero_cost, _zero_cost), node, written)
    return [*ordered, *_group_tests(group)]


def _group_tests(steps):
    """Attach tests to their producer when no callback separates them."""
    producers = {step.key: index for index, step in enumerate(steps)
                 if step.kind in (CLASSIFY, EXTRACT)}
    for index, step in enumerate(steps):
        if step.kind != TEST:
            continue
        test = step.key
        producer = producers.get(test.call if test.call.kind == "label" else test.name)
        if producer is not None:
            parent = steps[producer]
            steps[producer] = replace(
                parent, tests=(*parent.tests, test),
                selectivity=parent.selectivity * step.selectivity)
            steps[index] = None
    return [step for step in steps if step is not None]


def order_table_steps(steps, rule, count):
    """Order model work within each interval between callbacks."""
    ordered, group = [], []
    for step in (*steps, None):
        if (step is not None and step.kind != APPLY
                and (not group or group[-1].joins == step.joins)):
            group.append(step)
            continue
        ordered.extend(group[index] for index in order_filters_indexed(
            group, rule, count=count, resident=resident_after))
        for member in group:
            count *= member.selectivity
        group = []
        if step is not None and step.kind == APPLY:
            head = (ordered[-1].head if ordered and step.key.kind == "per_batch"
                    else None)
            ordered.append(replace(step, head=head))
        elif step is not None:
            group.append(step)
    return ordered


def _survival(selectivities) -> float:
    survival = 1.0
    for selectivity in selectivities:
        survival *= effective_selectivity(selectivity)
    return survival


def _classify_seconds(context, alias, call, name, resident, live) -> float:
    """Price a classification at a live count with its cheapest scoring rule."""
    spec, _ = classify_builder(context, alias, "quail").prepare(call, name, live)
    if spec.estimated_seconds:
        return spec.estimated_seconds
    try:
        chosen = classify_scoring(context, alias).choose_scoring(
            replace(spec, expected_inputs=live), call, resident)
    except ClassifyRefusedError:
        # the refusal is raised again when the node is built
        return float("inf")
    return chosen.estimated_seconds


def _extract_seconds(context, node, resident, live) -> float:
    """Price an extraction at a live count."""
    return extract_spec(context, node.call, node.name, live,
                        shared=resident).estimated_seconds

