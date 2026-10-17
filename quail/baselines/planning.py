"""Planning for the request backends: one physical request plan per region."""

from __future__ import annotations

from quail.baselines.lowering import (
    lower_classifications,
    lower_extractions,
    lower_filters,
    lower_joins,
    request_preamble,
)
from quail.baselines.nodes import RequestExecution
from quail.cost import budgets
from quail.logical import (
    SPAN_SUFFIX,
    Alias,
    Apply,
    classified_above_joins,
    column_name,
    effective_selectivity,
)
from quail.logical import Aggregate as LogicalAggregate
from quail.physical import (
    Limit,
    PortRef,
    Project,
    Recombine,
    Scan,
)
from quail.physical.base import input_ports
from quail.planner import (
    default_order_rule,
    hash_join_nodes,
    order_filters_indexed,
    preamble_tokens,
)
from quail.planner import (
    join_specs as logical_join_specs,
)
from quail.planner.operators.filter import prepare_filter_costs
from quail.planner.optimizer.candidates import PhysicalCandidate
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
from quail.planner.prompt_lowering import answer_row_offsets, true_false_token_ids
from quail.planner.search.join_order import search_joins
from quail.planner.statistics.tables import summarize_alias
from quail.planner.validation import extraction_refusal, has_extract, has_label


def plan_request_backend(
    region,
    context,
    *,
    backend_name: str,
    filter_submission: str,
    join_submission: str,
    scores_labels: bool = False,
) -> tuple[PhysicalCandidate, ...]:
    """Build one physical request plan for a request engine.

    Args:
        region: The logical model region to plan.
        context: The planning context.
        backend_name: The backend the plan names.
        filter_submission: How filter stages become requests.
        join_submission: How join tuples become requests.
        scores_labels: Whether the engine decodes an answer as text,
            which AI.CLASSIFY and AI.EXTRACT need.
    """
    # a decision model scores one option block per label
    decides = context.model.role == "decision"
    refusal = _refusal(region, context, backend_name=backend_name,
                       scores_labels=scores_labels, decides=decides)
    if refusal is not None:
        return (PhysicalCandidate(
            graph=None, plan=refusal, estimated_seconds=float("inf")),)
    operators = region.logical_plan.operators()
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    ask_filters = {alias: list(ps) for alias, ps in filters.items() if ps}
    summaries, stats = _corpus_stats(context)
    nodes = []
    input_refs = []
    aliases = tuple(scan.alias for scan in scans)
    for scan in scans:
        if scan.alias not in stats:
            raise ValueError(f"no document tokens for alias {scan.alias!r}")
        summary = stats[scan.alias]
        n_docs = summaries[scan.alias].count
        node = Scan(
            node_id=f"scan:{scan.alias}",
            alias=scan.alias,
            input_id=scan.alias,
            n_docs=n_docs,
            total_tokens=summaries[scan.alias].total,
            shard_ranges=((0, n_docs),),
            shard_token_loads=(summary.total_tokens,),
            predicates=scan.predicates,
            expected_docs=float(summary.n_docs) if scan.predicates else None,
        )
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"ids:{scan.alias}"))

    for node in hash_join_nodes(joins, context.pair_fractions, input_refs):
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"pairs:{node.written_pos}"))

    rule = context.order or default_order_rule(ask_filters, joins)[0]
    filter_orders, join_sequence = _order_filters_and_joins(
        context, operators, ask_filters, stats, summaries, rule)
    first_anchor = join_sequence[0][1] if join_sequence else None
    filter_specs = lower_filters(
        context, aliases, filters, ask_filters, filter_orders, first_anchor)
    classify_specs = lower_classifications(context, operators, decides)
    join_specs = lower_joins(context, joins, join_sequence)
    extract_specs = lower_extractions(context, operators.extracts)
    prompts = [
        *(predicate.prompt for spec in filter_specs
          for predicate in ask_filters[spec.alias]),
        *(call.prompt for call, _ in operators.labels.calls),
        *(joins[written_pos].prompt for written_pos, _ in join_sequence),
    ]
    preamble, preamble_text = request_preamble(context, prompts)

    request_node = RequestExecution(
        node_id="request-model",
        inputs=input_ports(tuple(input_refs)),
        backend_name=backend_name,
        aliases=aliases,
        preamble_token_ids=tuple(preamble),
        preamble_text=preamble_text,
        filters=tuple(filter_specs),
        joins=tuple(join_specs),
        classifies=tuple(classify_specs),
        extracts=tuple(extract_specs),
    )
    nodes.append(request_node)

    nodes.extend(_result_nodes(
        region, request_node, aliases, join_specs, extract_specs,
        operators.labels))

    true_ids, false_ids = true_false_token_ids(context.tokenizer)
    plan = PhysicalPlan(
        model=context.model.name,
        device=context.device.name,
        workers=1,
        backend=backend_name,
        estimated_seconds=0.0,
        nodes=tuple(nodes),
        settings={
            "filter_submission": filter_submission,
            "join_submission": join_submission,
            "true_ids": true_ids,
            "false_ids": false_ids,
            "order_rule": rule,
            **({"decision_offsets": list(answer_row_offsets(
                context.model.prompt_layout, context.tokenizer,
                context.model.turn_suffix))} if decides else {}),
        },
    )
    return (PhysicalCandidate(
        graph=plan.graph,
        plan=plan,
        estimated_seconds=plan.estimated_seconds,
    ),)


def _refusal(region, context, *, backend_name: str, scores_labels: bool,
             decides: bool) -> Refusal | None:
    """Return why the request backends cannot run the region, or None."""
    if any(isinstance(node, Apply) for node in region.logical_plan.walk()):
        return Refusal(
            reasons=("apply() functions run on the Quail backend only",),
            constraint="apply_needs_quail_backend",
            needed=1, available=0, unit="backends")
    root = region.logical_plan.result
    if root.order or root.distinct or root.offset \
            or isinstance(root.input, LogicalAggregate):
        return Refusal(
            reasons=("ORDER BY, DISTINCT, OFFSET, and GROUP BY run on "
                     "the Quail backend only",),
            constraint="sort_needs_quail_backend",
            needed=1, available=0, unit="backends")
    if has_extract(region.logical_plan):
        refusal = extraction_refusal(context)
        if refusal is None and not (scores_labels
                                    and not context.model.canvas_tokens):
            refusal = Refusal(
                reasons=(f"AI.EXTRACT on {backend_name} decodes each answer "
                         f"as text, which its engine does not return for "
                         f"{context.model.name!r}",),
                constraint="extract_needs_quail_backend",
                needed=1, available=0, unit="backends")
        if refusal is None and region.logical_plan.operators().extract_filters:
            refusal = Refusal(
                reasons=(f"a filter on an AI.EXTRACT answer runs on the "
                         f"Quail backend only, not {backend_name}",),
                constraint="extract_filter_needs_quail_backend",
                needed=1, available=0, unit="backends")
        if refusal is not None:
            return refusal
    if any(isinstance(column, Alias)
           and getattr(column.expression, "probabilities", False)
           for column in region.logical_plan.projection.columns):
        return Refusal(
            reasons=(f"AI.CLASSIFY on {backend_name} decodes the label "
                     f"as text and has no probability for the others",),
            constraint="classify_probabilities_need_quail_backend",
            needed=1, available=0, unit="backends")
    # a diffusion model's vLLM canvas is sized for a filter's one
    # answer token, not a label
    scores_labels = scores_labels and not context.model.canvas_tokens
    if has_label(region.logical_plan) and not ((scores_labels or decides)
                                               and context.tokenizer is not None):
        reason = (f"AI.CLASSIFY on {backend_name} decodes each answer as "
                  f"text, which its engine does not return for "
                  f"{context.model.name!r}"
                  if context.tokenizer is not None
                  else "AI.CLASSIFY planning needs the model's tokenizer")
        return Refusal(
            reasons=(reason,),
            constraint="classify_needs_quail_backend",
            needed=1, available=0, unit="backends")
    if classified_above_joins(region.logical_plan.root):
        return Refusal(
            reasons=(f"{backend_name} runs single-document classification "
                     "before joins",),
            constraint="classify_placement_not_supported",
            needed=1, available=0, unit="placements")
    return None


def _corpus_stats(context) -> tuple[dict, dict]:
    """Summarize each alias's corpus and the fraction its scan keeps.

    Returns:
        The token summary of each alias, and its CorpusStats after the
        regular predicates.
    """
    # the regular predicates keep an estimated fraction of each table, as
    # the Quail planner's statistics do
    summaries = {
        alias: summarize_alias(lengths, context.model.sliding_window, memo=context.memo)
        for alias, lengths in context.document_tokens.items()
    }
    stats = {
        alias: CorpusStats(
            n_docs=round(summary.count * context.scan_fractions.get(alias, 1.0)),
            total_tokens=round(summary.total
                               * context.scan_fractions.get(alias, 1.0)),
            max_doc_tokens=summary.maximum,
        )
        for alias, summary in summaries.items()
    }
    return summaries, stats


def _order_filters_and_joins(context, operators, ask_filters, stats,
                             summaries, rule) -> tuple[dict, list]:
    """Order each alias's AI.IF filters and search the join sequence.

    Returns:
        The written positions of each alias's filters in run order, and
        the (written position, anchor) sequence of the joins.
    """
    joins = operators.joins
    chunk_tokens = budgets.chunk_budget(context.model, context.device)
    preamble_count = preamble_tokens(
        context.lowered(prompt) for prompt in (
            *(p.prompt for fs in ask_filters.values() for p in fs),
            *(j.prompt for j in joins)))
    filter_orders = {
        alias: order_filters_indexed(prepare_filter_costs(
            predicates,
            prefix_tokens=(
                preamble_count + stats[alias].mean_doc_tokens
            ),
            model=context.model,
            device=context.device,
            chunk_tokens=chunk_tokens,
            context=context,
        ), rule)
        for alias, predicates in ask_filters.items()
    }
    live = {
        alias: float(summary.n_docs) for alias, summary in stats.items()
    }
    for alias, predicates in operators.all_filters().items():
        for predicate in predicates:
            live[alias] *= effective_selectivity(predicate.selectivity)
    search_specs = logical_join_specs(joins, context.pair_fractions, context=context)
    join_search = search_joins(
        search_specs,
        live,
        summaries,
        {},
        preamble_count,
        chunk_tokens,
        context.model,
        context.device,
        fixed_order=rule == "as_written",
    )
    join_sequence = (
        list(join_search.seq) if join_search is not None else [
            (position, spec.anchor or spec.aliases[0])
            for position, spec in enumerate(search_specs)
        ]
    )
    return filter_orders, join_sequence


def _result_nodes(region, request_node, aliases, join_specs, extract_specs,
                  label_plan) -> list:
    """Return the nodes that turn the request node's answers into result rows."""
    nodes = []
    full_joins = [spec for spec in join_specs if spec.semantics == "full"]
    if len(full_joins) == 1 and len(join_specs) == 1:
        # one full join and no gates: its true pairs are the result rows
        sink_input = PortRef(
            request_node.node_id, f"join_answers:{full_joins[0].written_pos}")
    elif full_joins:
        result_aliases = tuple(dict.fromkeys(
            alias for spec in full_joins for alias in spec.aliases
        ))
        edges = tuple(
            PortRef(request_node.node_id, f"join_answers:{spec.written_pos}")
            for spec in full_joins
        ) + tuple(
            PortRef(request_node.node_id, f"ids:{alias}")
            for alias in result_aliases
        )
        nodes.append(Recombine(
            node_id="recombine",
            inputs=input_ports(edges),
            alias_order=result_aliases,
        ))
        sink_input = PortRef("recombine", "tuples")
    else:
        sink_input = PortRef(request_node.node_id, f"ids:{aliases[0]}")

    label_ports = tuple(
        PortRef(request_node.node_id, f"label_answers:{name}")
        for name in label_plan.projected.values()
    ) + tuple(
        PortRef(request_node.node_id, f"extract_answers:{spec.output}")
        for spec in extract_specs)
    columns = []
    for column in region.logical_plan.projection.columns:
        columns.append(column_name(column))
        if isinstance(column, Alias) and column.expression.kind == "extract":
            columns.append(column.name + SPAN_SUFFIX)
    nodes.append(Project(
        node_id="project",
        inputs=input_ports((sink_input, *label_ports)),
        columns=tuple(columns),
        structs=region.logical_plan.projection.packing(),
    ))
    if region.logical_plan.result.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=region.logical_plan.result.limit,
        ))
    return nodes
