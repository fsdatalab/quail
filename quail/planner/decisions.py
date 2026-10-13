"""Validate native queries and choose their cost-based operator order."""

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost.work import Work
from quail.logical import CompileError, has_score
from quail.logical.plan import Operators
from quail.planner import join_order as joinsearch
from quail.planner.filter_order import default_order_rule
from quail.planner.ordering import choose_filter_orders, choose_join_sequence
from quail.planner.plan import Refusal
from quail.planner.statistics import (
    PlanStatistics,
    cached_statistics,
    filter_works,
    live_after_filters,
    question_tokens,
)
from quail.planner.validation import (
    classification_refusal,
    extraction_refusal,
    join_input_tokens,
    score_refusal,
)


@dataclass(frozen=True)
class PlanDecisions:
    """Operator order, input statistics, and work estimates for graph construction."""

    operators: Operators
    statistics: PlanStatistics
    filter_orders: dict
    join_sequence: list
    join_records: list
    live: dict[str, float]
    filter_work: dict[str, Work]
    order_rule: str
    order_source: str


def validate_query(plan, operators, context, model, device, gpus):
    """Return a refusal for unsupported inference or hardware requirements."""
    labels, applies = operators.labels, operators.applies
    scored = has_score(plan)
    if scored:
        refusal = score_refusal(plan, context)
        if refusal is not None:
            return refusal
    if labels.calls:
        if context is None:
            return Refusal(
                reasons=("AI.CLASSIFY planning needs a planning context",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        if any(len(call.aliases()) not in (1, 2) for call, _ in labels.calls):
            return Refusal(
                reasons=("AI.CLASSIFY reads one document, or one from each "
                         "side of a join",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        refusal = classification_refusal(context)
        if refusal is not None:
            return refusal
    extracts = operators.extracts
    if extracts:
        if context is None:
            return Refusal(
                reasons=("AI.EXTRACT planning needs a planning context",),
                constraint="unsupported_extract_query",
                needed=1, available=0, unit="queries")
        refusal = extraction_refusal(context)
        if refusal is not None:
            return refusal
    join_applies = {}
    for apply in applies:
        if apply.ids == "pairs":
            join_applies.setdefault(apply.written_pos, []).append(apply)
    names = [apply.function for apply in applies]
    if len(set(names)) != len(names):
        raise CompileError(
            f"each apply() needs its own name; {names} repeat one")
    if any(len(group) > 1 for group in join_applies.values()):
        raise CompileError("a join takes one apply() returning pairs")
    if gpus > 1 and any(apply.kind == "per_batch" for apply in applies):
        return Refusal(
            reasons=("per-batch apply() requires one GPU; "
                     "use apply_table() or set gpus=1",),
            constraint="per_batch_apply_needs_one_gpu",
            needed=1, available=gpus, unit="gpus")

    weight_gpus = budgets.minimum_weight_gpus(model, device)
    if weight_gpus > 1:
        return Refusal(
            reasons=(
                f"the model needs {weight_gpus} GPUs of memory, "
                f"but Quail loads one complete copy per GPU",
            ),
            constraint="weights_need_more_cards",
            needed=weight_gpus, available=1, unit="cards")
    return None


def choose_plan(plan, operators, model, device, doc_tokens, order,
                pair_fractions, context, scan_fractions):
    """Choose operator order and estimate work without constructing graph nodes."""
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    statistics = cached_statistics(
        plan, {} if context is None else context.memo, model=model,
        device=device, doc_tokens=doc_tokens, pair_fractions=pair_fractions,
        scan_fractions=scan_fractions, context=context)
    stats, chunk, pre = statistics.stats, statistics.chunk, statistics.pre
    specs = statistics.specs
    asks = statistics.asks
    ask_filters = {alias: [filters[alias][position] for position in positions]
                   for alias, positions in asks.items() if positions}
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(operators.all_filters(), joins)
    orders = choose_filter_orders(
        plan, statistics, model=model, device=device, rule=rule,
        context=context)

    live0 = live_after_filters(plan, statistics)
    works = filter_works(plan, statistics, model, orders)

    seq = [(specs[position], anchor) for position, anchor in
           choose_join_sequence(plan, statistics, orders, model=model,
                                device=device, rule=rule)]
    _, stage_records = joinsearch.walk(
        seq, live0, statistics.lengths, set(operators.all_filters()), pre,
        model, device)
    anchors = {spec["written_pos"]: anchor for spec, anchor in seq}
    for s in scans:
        fq = max((question_tokens(p.prompt, model.canvas_tokens)
                  for p in ask_filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_doc_tokens + fq
        if need > chunk:
            return Refusal(
                reasons=(f"a document in {s.alias!r} needs {need} tokens "
                         f"with its prompt, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")
    for spec in specs:
        if "cost" in spec:
            continue
        anchor = anchors[spec["written_pos"]]
        need = join_input_tokens(
            spec, anchor, {alias: stats[alias].max_doc_tokens
                           for alias in spec["aliases"]}, pre)
        if need > chunk:
            return Refusal(
                reasons=(f"one join pair anchored on {anchor!r} needs "
                         f"{need} tokens, but the forward pass budget "
                         f"is {chunk} tokens",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    return PlanDecisions(operators, statistics, orders, seq, stage_records,
                         live0, works, rule, source)
