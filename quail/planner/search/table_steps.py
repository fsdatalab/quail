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
from quail.planner.operators.extract import extract_spec
from quail.planner.operators.filter import prepare_filter_costs
from quail.planner.operators.map import map_spec
from quail.planner.operators.sae_features import sae_features_spec
from quail.planner.validation import ClassifyRefusedError

ASK, CLASSIFY, EXTRACT = "ask", "classify", "extract"
FEATURES = "features"
MAP = "map"


@dataclass(frozen=True)
class TableStep:
    """One step of a table's sequence and its costs.

    Attributes:
        kind: ASK, CLASSIFY, EXTRACT, or MAP.
        key: The ask's written position, the classification's call, or
            the extraction's or map's name.
        selectivity: The fraction of documents the step keeps: the
            product of its tests' selectivities, 1 without a test.
        first: Seconds at a live count, the documents read afresh.
        later: Seconds at a live count, the documents' KV resident.
        reuse_key: Identity of a result an earlier step computed, or None.
        document_head: The token ids the step shows the document under.
    """

    kind: str
    key: Any
    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Any = None
    document_head: Any = None


def resident_after(previous: TableStep, step: TableStep) -> bool:
    """Return whether a step reads the KV the step before it left.

    The steps under one head run in one pipeline: each rewinds the KV
    past what the step before it appended and asks its question
    (the can_follow rules of the physical nodes).
    """
    return previous.document_head == step.document_head


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
            predicates, mean_tokens=self.statistics.pre + stats.mean_document_tokens,
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

    def features(self, node) -> tuple:
        """Return a feature call's (first, later) costs at a live count."""
        max_pooling = (self.context.logical_plan is not None and any(
            call.alias == node.alias and call.pooling == "max"
            for call in self.context.logical_plan.operators().sae_features))
        return tuple(partial(_features_seconds, self.context, node, resident,
                             max_pooling=max_pooling)
                     for resident in (False, True))

    def map(self, node) -> tuple:
        """Return a map's (first, later) costs at a live count."""
        return tuple(partial(_map_seconds, self.context, node, resident)
                     for resident in (False, True))

    def document_head(self, prompt):
        """Return the token ids a prompt shows the document under."""
        return tuple(self.context.lowered(prompt).preamble_token_ids)


def table_steps(operators, alias: str, asks, pricing) -> list[TableStep]:
    """Return a table's steps in written order, priced.

    Args:
        operators: The plan's Operators.
        alias: The table.
        asks: The written positions of the table's AI.IF predicates.
        pricing: Prices the steps: asks(predicates), classify(call,
            name), extract(node), and document_head(prompt), as
            PlanPricing does.
    """
    steps = []
    predicates = operators.filters.get(alias, ())
    costs = pricing.asks([predicates[position] for position in asks])
    for position, cost in zip(asks, costs):
        steps.append(TableStep(ASK, position, cost.selectivity, cost.first,
                               cost.later, cost.reuse_key,
                               document_head=pricing.document_head(
                                   predicates[position].prompt)))
    tests = {}
    for test in operators.label_filters.get(alias, ()):
        tests.setdefault(test.call, []).append(test.selectivity)
    for call, owner in operators.labels.calls:
        if owner != alias or len(call.aliases()) != 1:
            continue
        first, later = pricing.classify(call, operators.labels.names[call])
        steps.append(TableStep(CLASSIFY, call, _survival(tests.get(call, ())),
                               first, later,
                               document_head=pricing.document_head(call.prompt)))
    tests = {}
    for test in operators.extract_filters.get(alias, ()):
        tests.setdefault(test.name, []).append(test.selectivity)
    for node in operators.extracts:
        if node.alias != alias:
            continue
        first, later = pricing.extract(node)
        steps.append(TableStep(EXTRACT, node.name, _survival(tests.get(node.name, ())),
                               first, later,
                               document_head=pricing.document_head(node.call.prompt)))
    for node in operators.sae_features:
        if node.alias != alias:
            continue
        first, later = pricing.features(node)
        steps.append(TableStep(
            FEATURES, node.name, 1.0, first, later,
            document_head=pricing.document_head(node.call.prompt)))
    for node in operators.maps:
        if node.alias != alias:
            continue
        first, later = pricing.map(node)
        steps.append(TableStep(MAP, node.name, 1.0, first, later,
                               document_head=pricing.document_head(node.call.prompt)))
    return steps


def _survival(selectivities) -> float:
    survival = 1.0
    for selectivity in selectivities:
        survival *= effective_selectivity(selectivity)
    return survival


def _classify_seconds(context, alias, call, name, resident, expected_rows) -> float:
    """Price a classification at a live count with its cheapest scoring rule."""
    spec, _ = classify_builder(context, alias, "quail").prepare(
        call, name, expected_rows
    )
    if spec.estimated_seconds:
        return spec.estimated_seconds
    try:
        chosen = classify_scoring(context, alias).choose_scoring(
            replace(spec, expected_rows=expected_rows), call, resident)
    except ClassifyRefusedError:
        # the refusal is raised again when the node is built
        return float("inf")
    return chosen.estimated_seconds


def _extract_seconds(context, node, resident, expected_rows) -> float:
    """Price an extraction at a live count."""
    return extract_spec(context, node.call, node.name, expected_rows,
                        resident=resident).estimated_seconds


def _features_seconds(context, node, resident, expected_rows,
                      max_pooling=False) -> float:
    """Price a feature call at a live count."""
    return sae_features_spec(context, node.call, node.name, expected_rows,
                             resident=resident,
                             max_pooling=max_pooling).estimated_seconds


def _map_seconds(context, node, resident, expected_rows) -> float:
    """Price a map at a live count."""
    return map_spec(context, node.call, node.name, expected_rows,
                    resident=resident).estimated_seconds

