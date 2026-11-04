"""Planning for the request backends: one physical request plan per region."""

from __future__ import annotations

from quail.baselines.lowering import (
    lower_answers,
    lower_classifications,
    lower_filters,
    lower_joins,
    request_preamble,
)
from quail.baselines.nodes import (
    RequestExecution,
    RequestExtractSpec,
    RequestMapSpec,
)
from quail.common.expressions import aliased_call, packed_columns
from quail.common.prompts import PROMPT_LAYOUTS, answer_row_offsets
from quail.logical import (
    SPAN_SUFFIX,
    RegularPredicate,
    column_name,
    columns_above_joins,
    has_extract,
    has_label,
)
from quail.logical import Aggregate as LogicalAggregate
from quail.logical import Window as LogicalWindow
from quail.physical import (
    Limit,
    PortRef,
    Project,
    Recombine,
    Scan,
    Sort,
)
from quail.physical.base import input_ports
from quail.physical.lowering import answer_ids
from quail.physical.plan import PhysicalPlan, Refusal
from quail.planner.candidates import PhysicalCandidate
from quail.planner.native import request_order


def plan_request_backend(
    region,
    context,
    *,
    backend_name: str,
    filter_submission: str,
    join_submission: str,
    answer_submission: str = "operator-at-a-time",
    scores_labels: bool = False,
) -> tuple[PhysicalCandidate, ...]:
    """Build one physical request plan for a request engine.

    Args:
        region: The logical model region to plan.
        context: The planning context.
        backend_name: The backend the plan names.
        filter_submission: How filter stages become requests.
        join_submission: How join tuples become requests.
        answer_submission: How each document's extraction and map prompts
            become requests.
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
    order = request_order(region.logical_plan, context)
    rule, tables = order["rule"], order["tables"]
    filter_orders, join_sequence = order["filter_orders"], order["join_sequence"]
    nodes = []
    input_refs = []
    aliases = tuple(scan.alias for scan in scans)
    for scan in scans:
        table = tables[scan.alias]
        num_documents = table["num_documents"]
        node = Scan(
            node_id=f"scan:{scan.alias}",
            alias=scan.alias,
            input_id=scan.alias,
            num_documents=num_documents,
            total_tokens=table["total_tokens"],
            shard_ranges=((0, num_documents),),
            shard_token_loads=(table["kept_tokens"],),
            predicates=tuple(scan.predicates),
            expected_rows=(float(table["kept_documents"]) if scan.predicates
                           else None),
        )
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"rows:{scan.alias}"))

    for node in order["hash_joins"]:
        nodes.append(node)
        input_refs.append(PortRef(node.node_id, f"pairs:{node.written_position}"))

    first_anchor = join_sequence[0][1] if join_sequence else None
    filter_specs = lower_filters(
        context, aliases, filters, ask_filters, filter_orders, first_anchor)
    classify_specs = lower_classifications(context, operators, decides)
    join_specs = lower_joins(context, joins, join_sequence)
    extract_specs = lower_answers(
        context, operators.extracts, RequestExtractSpec,
        question=lambda node: node.call.question)
    map_specs = lower_answers(
        context, operators.maps, RequestMapSpec,
        max_answer_tokens=lambda node: node.call.max_answer_tokens)
    prompts = [
        *(predicate.prompt for spec in filter_specs
          for predicate in ask_filters[spec.alias]),
        *(call.prompt for call, _ in operators.labels.calls),
        *(joins[written_position].prompt for written_position, _ in join_sequence),
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
        maps=tuple(map_specs),
    )
    nodes.append(request_node)

    nodes.extend(_result_nodes(
        region, request_node, aliases, join_specs,
        (*extract_specs, *map_specs), operators.labels))

    true_ids, false_ids = answer_ids(context.model, context.tokenizer)
    plan = PhysicalPlan(
        model=context.model.name,
        device=context.device.name,
        gpus=1,
        backend=backend_name,
        estimated_seconds=0.0,
        nodes=tuple(nodes),
        settings={
            "filter_submission": filter_submission,
            "join_submission": join_submission,
            "answer_submission": answer_submission,
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


def _extraction_refusal(context, function: str) -> Refusal | None:
    """Return why the model cannot write AI.EXTRACT or AI.MAP answers, or None."""
    model = context.model
    if context.tokenizer is None:
        reason = f"{function} planning needs the model's tokenizer"
    elif model.role in ("reranker", "decision") or model.canvas_tokens:
        reason = (f"{function} decodes the model's next tokens, which "
                  f"{model.name!r} does not give")
    elif PROMPT_LAYOUTS[model.prompt_layout].read_end is not None:
        # the model answers by scoring options and document positions
        if function == "AI.EXTRACT":
            return None
        reason = (f"{function} writes text, and {model.name!r} answers only "
                  f"by scoring options")
    else:
        return None
    return Refusal(reasons=(reason,),
                   constraint=f"unsupported_{function.removeprefix('AI.').lower()}_query",
                   needed=1, available=0, unit="queries")


def _refusal(region, context, *, backend_name: str, scores_labels: bool,
             decides: bool) -> Refusal | None:
    """Return why the request backends cannot run the region, or None."""
    root = region.logical_plan.result
    projected = {column_name(column)
                 for column in region.logical_plan.projection.columns}
    if any(key.name not in projected for key in root.order) \
            or isinstance(root.input, (LogicalAggregate, LogicalWindow)):
        return Refusal(
            reasons=("GROUP BY, window functions, and ORDER BY on a column "
                     "the query does not return run on the Quail backend only",),
            constraint="sort_needs_quail_backend",
            needed=1, available=0, unit="backends")
    operators = region.logical_plan.operators()
    if any(test.call.kind == "label" and not (
            isinstance(test.condition, RegularPredicate)
            and test.condition.comparison == "in" and test.condition.function is None)
           for tests in operators.column_filters.values() for test in tests):
        return Refusal(
            reasons=("plain label tests other than IN run on the Quail backend only",),
            constraint="label_test_needs_quail_backend",
            needed=1, available=0, unit="backends")
    for function, present in (("AI.EXTRACT", has_extract(region.logical_plan)),
                              ("AI.MAP", bool(operators.maps))):
        if not present:
            continue
        refusal = _extraction_refusal(context, function)
        if refusal is None and not (scores_labels
                                    and not context.model.canvas_tokens):
            refusal = Refusal(
                reasons=(f"{function} on {backend_name} decodes each answer "
                         f"as text, which its engine does not return for "
                         f"{context.model.name!r}",),
                constraint=f"{function[3:].lower()}_needs_quail_backend",
                needed=1, available=0, unit="backends")
        kind = function[3:].lower()
        tested = any(test.call.kind == kind
                     for tests in operators.column_filters.values() for test in tests)
        if refusal is None and tested:
            refusal = Refusal(
                reasons=(f"a filter on an {function} answer runs on the "
                         f"Quail backend only, not {backend_name}",),
                constraint=f"{kind}_filter_needs_quail_backend",
                needed=1, available=0, unit="backends")
        if refusal is not None:
            return refusal
    if any(getattr(aliased_call(column), "probabilities", False)
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
    if any(kind == "label"
           for _, kind in columns_above_joins(region.logical_plan.root)):
        return Refusal(
            reasons=(f"{backend_name} runs single-document classification "
                     "before joins",),
            constraint="classify_placement_not_supported",
            needed=1, available=0, unit="placements")
    return None


def _result_nodes(region, request_node, aliases, join_specs, answer_specs,
                  label_plan) -> list:
    """Return the nodes that turn the request node's answers into result rows."""
    nodes = []
    full_joins = [spec for spec in join_specs if spec.semantics == "full"]
    if len(full_joins) == 1 and len(join_specs) == 1:
        # one full join and no gates: its true pairs are the result rows
        sink_input = PortRef(
            request_node.node_id, f"join_answers:{full_joins[0].written_position}")
    elif full_joins:
        result_aliases = tuple(dict.fromkeys(
            alias for spec in full_joins for alias in spec.aliases
        ))
        edges = tuple(
            PortRef(request_node.node_id, f"join_answers:{spec.written_position}")
            for spec in full_joins
        ) + tuple(
            PortRef(request_node.node_id, f"rows:{alias}")
            for alias in result_aliases
        )
        nodes.append(Recombine(
            node_id="recombine",
            inputs=input_ports(edges),
            alias_order=result_aliases,
        ))
        sink_input = PortRef("recombine", "tuples")
    else:
        sink_input = PortRef(request_node.node_id, f"rows:{aliases[0]}")

    label_ports = tuple(
        PortRef(request_node.node_id, f"label_answers:{name}")
        for name in label_plan.projected.values()
    ) + tuple(
        PortRef(request_node.node_id,
                f"{'map' if isinstance(spec, RequestMapSpec) else 'extract'}"
                f"_answers:{spec.output}")
        for spec in answer_specs)
    columns = []
    for column in region.logical_plan.projection.columns:
        columns.append(column_name(column))
        if aliased_call(column, "extract") is not None:
            columns.append(column.name + SPAN_SUFFIX)
    structs = region.logical_plan.projection.packing()
    nodes.append(Project(
        node_id="project",
        inputs=input_ports((sink_input, *label_ports)),
        columns=tuple(columns),
        structs=structs,
    ))
    root = region.logical_plan.result
    if root.order or root.distinct or root.offset:
        # the coordinator sorts the returned rows, as on the Quail backend
        nodes.append(Sort(
            node_id="sort",
            inputs=input_ports((PortRef("project", "rows"),)),
            keys=tuple((key.name, key.descending, key.nulls_first)
                       for key in root.order),
            columns=packed_columns(columns, structs), distinct=root.distinct,
            offset=root.offset, fetch=root.limit))
    elif root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=root.limit,
        ))
    return nodes
