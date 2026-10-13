"""Assemble and execute model pipelines over one table's documents."""

import pyarrow as pa

from quail.backends.quail.executor.operators.apply import ApplyGate, apply_part
from quail.backends.quail.executor.operators.classify import (
    ClassifyPart,
    classification_part,
    joined_classification,
)
from quail.backends.quail.executor.operators.extract import (
    ExtractPart,
    execute_extractions,
)
from quail.backends.quail.executor.operators.filter import (
    FilterPart,
    LabelGate,
    execute_filters,
)
from quail.backends.quail.executor.operators.join import execute_joins
from quail.backends.quail.executor.parts import (
    complete_chain,
    execution_prefix_tree,
    gated_stages,
    input_staging,
    report_chain_transitions,
    settle_documents,
)
from quail.backends.quail.executor.stages import run_stages
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.tokens import DocumentKeys, DocumentPrefixes
from quail.physical import AiClassify, AiExtract, AiFilter, AiJoin, Filter, Foreign


def execute_pipeline(state: QueryExecutionState, pipeline, inputs, context) -> dict:
    """Run one table's chain of per-document operators as one run.

    The members' stages are concatenated and driven by one stage
    scheduler over the documents the first member was given, so a
    document goes through every operator with its KV resident. A
    filter on a label or per-batch apply between two stages gates the
    next stage; one after the last stage runs over the documents
    that came out of it. A pipeline of extractions runs as one stage
    list of their questions (quail.backends.quail.executor.operators.extract);
    extractions after a filter ask their questions of the documents
    the filter's stages leave resident.

    Args:
        state: The loaded model and current query state.
        pipeline: Compatible physical nodes in graph order.
        inputs: Member node id -> its inputs from outside the
            pipeline, by port name.
        context: The graph's execution context.

    Returns:
        Node id -> NodeResult for every member.
    """
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    members = list(pipeline)
    alias = pipeline[0].document_aliases[0]
    first = members[0]
    # the first member's documents: its ids input, or its one input
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"ids:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    ids = (source.column(alias).to_pylist() if isinstance(source, pa.Table)
           else list(source))
    if isinstance(first, AiExtract):
        # the extractions of one table: every document is read once
        # and asked each question while resident
        return execute_extractions(
            state, members,
            context.model_inputs(first, inputs[first.node_id], context))
    if all(isinstance(member, AiFilter) for member in members):
        prepared = context.model_inputs(first, inputs[first.node_id], context)
        return execute_filters(state, members, prepared)
    gpu_inputs = {"gpu_timing": context.state.get("gpu_timing", False)}
    # the members with stages, then the gates after the last of them
    last_staged = max(index for index, member in enumerate(members)
                      if isinstance(member, (AiFilter, AiClassify, AiJoin,
                                             AiExtract)))
    staged, trailing = members[:last_staged + 1], members[last_staged + 1:]
    joins = [index for index, member in enumerate(staged)
             if isinstance(member, AiJoin)]
    parts = []
    if joins:
        for member in staged[:joins[0]]:
            parts.append(_part(state, member, ids, parts, inputs, context))
    # a pipeline that starts with a classification writes the
    # documents' KV under that classification's own prompt head;
    # a classification plan's settings carry no shared preamble
    head = (list(first.spec.prompt_token_parts[0])
            if isinstance(first, AiClassify) else context.state["pre"])
    prefixes = DocumentPrefixes(head, context.state["docs"][alias], ids)
    results = {}
    if joins:
        # the join settles each anchor's KV; a classification of
        # the rows it keeps runs as stages after the join's
        join_nodes = [staged[index] for index in joins]
        join = join_nodes[0]
        chain = {"documents": prefixes, "document_ids": ids,
                 "parts": parts, "after": [],
                 "pairs": {part.node.written_pos: part.rows
                           for part in parts
                           if isinstance(part, ApplyGate)
                           and part.node.ids == "pairs"}}
        join_inputs = context.model_inputs(
            join, inputs, context, chain, nodes=join_nodes)
        for member in staged[joins[-1] + 1:]:
            chain["after"].append(joined_classification(
                state, member, join_nodes, ids, join_inputs, context))
        results.update(execute_joins(state, join_nodes, join_inputs))
        context.model_result(join_nodes[-1], results[join_nodes[-1].node_id], context)
        for part in chain["after"]:
            results[part.node.node_id] = part.result_value
    else:
        parts = _extract_parts(state, staged, ids, head, inputs, context)
        stages = gated_stages(parts)
        stats = {}
        appended = [part.appended_tokens for part in parts
                    if isinstance(part, ExtractPart)]
        keys = DocumentKeys(alias, ids)
        every, spans, tokens = run_stages(
            state.torch, state.loaded_model.arena, state.loaded_model.pipeline, stages,
            prefixes, state.chunk_tokens,
            anchor_keys=keys,
            on_settled=settle_documents(
                state.loaded_model.arena, keys, prefixes,
                staged[-1].keep_kv, state.chunk_tokens),
            attention_mode=getattr(first, "attention", None) or None,
            prefix_tree=execution_prefix_tree(
                first, prefixes, state.loaded_model.arena),
            stats=stats, staging=input_staging(state),
            on_chunk=lambda transitions: report_chain_transitions(
                parts, transitions),
            label=f"pipeline {alias} ({len(stages)} stages)",
            appended_tokens=max(appended) if appended else None)
        complete_chain(parts, every, spans, tokens, stats,
                       state.torch, gpu_inputs)
    for part in parts:
        if isinstance(part, ExtractPart):
            results.update(part.results)
        else:
            results[part.node.node_id] = part.result_value
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


def _extract_parts(state: QueryExecutionState, members, ids, head, inputs,
                   context):
    """Build the members' parts, consecutive extractions as one.

    Args:
        state: The loaded model and current query state.
        members: The pipeline's staged members, in order.
        ids: The pipeline's document ids.
        head: The prompt head the pipeline writes the documents under.
        inputs: Member node id -> its inputs from outside the pipeline.
        context: The graph's execution context.
    """
    parts = []
    index = 0
    while index < len(members):
        member = members[index]
        if isinstance(member, AiExtract):
            end = index
            while end < len(members) and isinstance(members[end], AiExtract):
                end += 1
            # the documents are the pipeline's, not an input port's
            parts.append(ExtractPart(state, members[index:end], ids, {
                "documents": context.state["docs"],
                "texts": context.state.get("texts", {}),
                "pre": list(head)}))
            index = end
        else:
            parts.append(_part(state, member, ids, parts, inputs, context))
            index += 1
    return parts


def _part(state: QueryExecutionState, node, ids, parts, inputs, context):
    """Build a pipeline member's execution part using preceding parts."""
    if isinstance(node, AiFilter):
        return FilterPart(node, ids, state.async_answers,
                          len(state.loaded_model.pipeline.canvas_ids))
    if isinstance(node, AiClassify):
        return classification_part(state, node, ids, parts, inputs[node.node_id])
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
