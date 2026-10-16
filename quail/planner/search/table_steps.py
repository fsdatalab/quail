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
    """

    kind: str
    key: Any
    selectivity: float
    first: Callable[[float], float]
    later: Callable[[float], float]
    reuse_key: Any = None


def resident_after(previous: TableStep, step: TableStep) -> bool:
    """Return whether a step reads the KV the step before it left.

    An ask or a classification leaves the document under the query's
    shared head, which a later ask, classification, or extraction
    continues. An extraction's answer and its tests stay in the
    extraction pipeline, which only another extraction continues.
    """
    if previous.kind == EXTRACT:
        return step.kind == EXTRACT
    if step.kind == ASK:
        return previous.kind == ASK
    return True


def table_steps(operators, alias: str, *, context, statistics) -> list[TableStep]:
    """Return a table's steps in written order, priced.

    Args:
        operators: The plan's Operators.
        alias: The table.
        context: The planning context, with the model, device, and
            tokenizer.
        statistics: The plan's PlanStatistics.
    """
    stats = statistics.stats[alias]
    steps = []
    predicates = operators.filters.get(alias, ())
    asks = statistics.asks.get(alias, ())
    costs = prepare_filter_costs(
        [predicates[position] for position in asks],
        prefix_tokens=statistics.pre + stats.mean_doc_tokens,
        model=context.model, device=context.device,
        chunk_tokens=statistics.chunk, context=context)
    for position, cost in zip(asks, costs):
        steps.append(TableStep(ASK, position, cost.selectivity, cost.first,
                               cost.later, cost.reuse_key))
    tests = {}
    for test in operators.label_filters.get(alias, ()):
        tests.setdefault(test.call, []).append(test.selectivity)
    for call, owner in operators.labels.calls:
        if owner != alias or len(call.aliases()) != 1:
            continue
        name = operators.labels.names[call]
        steps.append(TableStep(
            CLASSIFY, call, _survival(tests.get(call, ())),
            partial(_classify_seconds, context, alias, call, name, False),
            partial(_classify_seconds, context, alias, call, name, True)))
    tests = {}
    for test in operators.extract_filters.get(alias, ()):
        tests.setdefault(test.name, []).append(test.selectivity)
    for node in operators.extracts:
        if node.alias != alias:
            continue
        steps.append(TableStep(
            EXTRACT, node.name, _survival(tests.get(node.name, ())),
            partial(_extract_seconds, context, node, False),
            partial(_extract_seconds, context, node, True)))
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

