"""Assemble and execute model pipelines over one table's documents."""

from quail.backends.quail.executor.operators.apply import ApplyGate, apply_part
from quail.backends.quail.executor.operators.classify import (
    ClassifyPart,
    classification_part,
    joined_classification,
)
from quail.backends.quail.executor.operators.filter import FilterPart, LabelGate
from quail.backends.quail.executor.operators.join import execute_join
from quail.backends.quail.executor.parts import (
    complete_chain,
    execution_prefix_tree,
    gated_stages,
    input_staging,
    report_chain_transitions,
)
from quail.backends.quail.executor.stages import run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.tokens import DocumentKeys, DocumentPrefixes
from quail.physical import AiClassify, AiFilter, AiJoin, Filter, Foreign


def execute_pipeline(state: QueryExecutionState, pipeline, inputs, context) -> dict:
    """Run one table's chain of per-document operators as one run.

    The members' stages are concatenated and driven by one stage
    scheduler over the documents the first member was given, so a
    document goes through every operator with its KV resident. A
    filter on a label or per-batch apply between two stages gates the
    next stage; one after the last stage runs over the documents
    that came out of it.

    Args:
        state: The loaded model and current query state.
        pipeline: The Pipeline (quail.execution.pipelines).
        inputs: Member node id -> its inputs from outside the
            pipeline, by port name.
        context: The graph's execution context.

    Returns:
        Node id -> NodeResult for every member.
    """
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    members = list(pipeline.members)
    alias = pipeline.alias
    first = members[0]
    # the first member's documents: its ids input, or its one input
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"ids:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    ids = list(source)
    gpu_inputs = {"gpu_timing": context.state.get("gpu_timing", False)}
    # the members with stages, then the gates after the last of them
    last_staged = max(index for index, member in enumerate(members)
                      if isinstance(member, (AiFilter, AiClassify, AiJoin)))
    staged, trailing = members[:last_staged + 1], members[last_staged + 1:]
    joins = [index for index, member in enumerate(staged)
             if isinstance(member, AiJoin)]
    parts = []
    for member in staged[:joins[0] if joins else -1]:
        parts.append(_part(state, member, ids, parts, inputs, context))
    # a pipeline that starts with a classification writes the
    # documents' KV under that classification's own prompt head;
    # a classification plan's settings carry no shared preamble
    head = (list(first.spec.prompt_token_parts[0])
            if isinstance(first, AiClassify) else context.state["pre"])
    prefixes = DocumentPrefixes(head, context.state["docs"][alias], ids)
    sink = staged[-1]
    results = {}
    if joins:
        # the join settles each anchor's KV; a classification of
        # the rows it keeps runs as stages after the join's
        join = staged[joins[0]]
        chain = {"documents": prefixes, "document_ids": ids,
                 "parts": parts, "after": [],
                 "pairs": {part.node.written_pos: part.rows
                           for part in parts
                           if isinstance(part, ApplyGate)
                           and part.node.ids == "pairs"}}
        join_inputs = context.model_inputs(
            join, inputs[join.node_id], context, chain)
        for member in staged[joins[0] + 1:]:
            chain["after"].append(joined_classification(
                state, member, join, ids, join_inputs, context))
        results[join.node_id] = execute_join(state, join, join_inputs)
        context.model_result(join, results[join.node_id], context)
        for part in chain["after"]:
            results[part.node.node_id] = part.result_value
    else:
        own = _part(state, sink, ids, parts, inputs, context)
        stages = gated_stages(parts + [own])
        stats = {}
        every, spans, tokens = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline, stages,
            prefixes, state.chunk_tokens,
            anchor_keys=DocumentKeys(alias, ids),
            attention_mode=getattr(first, "attention", None) or None,
            prefix_tree=execution_prefix_tree(
                first, prefixes, state.loaded_model.arena),
            stats=stats, staging=input_staging(state),
            on_chunk=lambda transitions: report_chain_transitions(
                parts + [own], transitions),
            label=f"pipeline {alias} ({len(stages)} stages)")
        complete_chain(parts + [own], every, spans, tokens, stats,
                       state.torch, gpu_inputs)
    for part in parts:
        results[part.node.node_id] = part.result_value
    if not joins:
        results[sink.node_id] = own.result_value
    # gates after the last stage read the documents that came out
    for member in trailing:
        member_inputs = dict(inputs[member.node_id])
        for port in member.inputs:
            if port.source.node_id in results:
                member_inputs[port.name] = results[
                    port.source.node_id].outputs[port.source.port]
        results[member.node_id] = context.runtimes[
            member.runtime_key].execute(member, member_inputs, context)
    return results


def _part(state: QueryExecutionState, node, ids, parts, inputs, context):
    """The pipeline part for one member, after the parts before it."""
    if isinstance(node, AiFilter):
        return FilterPart(node, ids, state.async_answers)
    if isinstance(node, AiClassify):
        return classification_part(state, node, ids, parts)
    if isinstance(node, Filter):
        labeled = next(part for part in reversed(parts)
                       if isinstance(part, ClassifyPart)
                       and part.node.spec.name == node.column)
        source = next(part for part in parts
                      if part.node.node_id == node.inputs[0].source.node_id)
        return LabelGate(node, labeled, source)
    if isinstance(node, Foreign):
        return apply_part(node, ids, parts, inputs, context)
    raise TypeError(f"{node.type_name!r} is not a pipeline operator")
