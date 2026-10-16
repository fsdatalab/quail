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
from typing import Any, Callable

from quail.logical import effective_selectivity
from quail.planner.operators.classify import classify_builder, classify_scoring
from quail.planner.operators.common import shares_query_head
from quail.planner.operators.extract import extract_spec
from quail.planner.operators.filter import prepare_filter_costs
from quail.planner.validation import ClassifyRefusedError

ASK, CLASSIFY, EXTRACT = "ask", "classify", "extract"


@dataclass(frozen=True)
class TableStep:
    """One step of a table's sequence and its costs.

    Attributes:
        kind: ASK, CLASSIFY, or EXTRACT.
        key: The ask's written position, the classification's call, or
            the extraction's name.
        selectivity: The fraction of documents the step keeps: the
            product of its tests' selectivities, 1 without a test.
        first: Seconds at a live count, the documents read afresh.
        later: Seconds at a live count, the documents' KV resident.
        reuse_key: Identity of a result an earlier step computed, or None.
        head: The prompt head the step shows the document under: None
            for the query's shared head, else its token ids.
    """

    kind: str
    key: Any
    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Any = None
    head: Any = None


def resident_after(previous: TableStep, step: TableStep) -> bool:
    """Return whether a step reads the KV the step before it left.

    The steps under one head run in one pipeline: each rewinds the KV
    past what the step before it appended and asks its question
    (the can_follow rules of the physical nodes).
    """
    return previous.head == step.head


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
        """Return a prompt's head: None under the query's shared head."""
        lowered = self.context.lowered(prompt)
        return (None if shares_query_head(self.context, lowered)
                else tuple(lowered.preamble_token_ids))


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
                               cost.later, cost.reuse_key))
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
    return steps


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

