"""Run compatible physical nodes through the shared document scheduler."""

from itertools import groupby

import numpy as np

from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.operators.filter import (
    execute_set_filters,
    filter_inputs,
)
from quail.backends.quail.executor.operators.join import JoinPart, execute_set_joins
from quail.backends.quail.executor.operators.set_classify import (
    classify_passes,
    classify_result,
    joined_label_passes,
)
from quail.backends.quail.executor.operators.set_extract import (
    execute_set_extractions,
)
from quail.backends.quail.executor.parts import execute_parts, execution_prefix_tree
from quail.backends.quail.executor.state import QueryExecutionState
from quail.catalog.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.execution.runtimes.filter import FilterRuntime
from quail.physical import AiClassify, AiExtract, AiFilter, AiJoin, Filter
from quail.physical.base import ValueType


def execute_pipeline(state: QueryExecutionState, nodes, inputs, context) -> dict:
    """Prepare compatible operators, run their stages, and return node results."""
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    first = nodes[0]
    implementations = execution_parts()
    implementation = implementations[first.type_name]
    if (implementation.execute_group is not None
            and all(node.type_name == first.type_name for node in nodes)):
        prepared = context.model_inputs(first, inputs, context, nodes=nodes)
        return implementation.execute_group(state, nodes, prepared)
    if state.loaded_model.pipeline.whole_sequences:
        return execute_set_pipeline(state, nodes, inputs, context)

    alias = first.document_aliases[0]
    ids = pipeline_ids(first, inputs)
    last = max(index for index, node in enumerate(nodes)
               if node.backend is not None)

    def group_key(node):
        implementation = implementations[node.type_name]
        return node.type_name, None if implementation.combine else node.node_id

    parts = []
    for _, group in groupby(nodes, group_key):
        members = list(group)
        parts.append(implementations[members[0].type_name].prepare(
            state, members, ids, parts, inputs, context))
    prefixes = DocumentPrefixes(
        first.document_head, context.state["documents"][alias], ids
    )
    # a trailing condition gates the documents before they complete,
    # so a result limit counts only the documents that pass it
    return execute_parts(
        state, parts, prefixes, DocumentKeys(alias, ids),
        keep_kv=nodes[last].keep_kv,
        prefix_tree=execution_prefix_tree(first, prefixes, state.loaded_model.arena),
        limit=context.state.get("filter_limit"),
        attention=(first.attention or None
                   if isinstance(first, (AiFilter, AiJoin)) else None),
        label=f"pipeline {alias}")


def pipeline_ids(first, inputs) -> np.ndarray:
    """The document ids a pipeline's first operator reads."""
    alias = first.document_aliases[0]
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"rows:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    return id_positions(source.ids[alias] if isinstance(source, Rows) else source)


def _kind(node) -> str:
    """The kind of step a node is in a set-attention pipeline."""
    if isinstance(node, AiFilter):
        return "filter"
    if isinstance(node, AiClassify):
        return "label" if node.spec.partner is None else "joined"
    if isinstance(node, AiExtract):
        return "extract"
    if isinstance(node, AiJoin):
        return "join"
    if isinstance(node, Filter) and len(node.aliases) == 1:
        return "condition"
    raise ValueError(f"a set-attention pipeline does not run {node.type_name!r}")


def _node_inputs(node, results, inputs) -> dict:
    """A node's input values: from the pipeline's results, else from outside it."""
    return {port.name: (results[port.source.node_id].outputs[port.source.port]
                        if port.source.node_id in results
                        else inputs[node.node_id][port.name])
            for port in node.inputs}


def execute_set_pipeline(state, nodes, inputs, context) -> dict:
    """Run a pipeline's operators under set attention, step by step.

    The operators of one pass group (PackSetPasses) share their passes.
    A condition on their columns runs between them. A join's anchors are
    the rows the steps before it kept, and the classifications of its
    pairs follow it. Each step reads the rows the step before it kept.
    """
    alias = nodes[0].document_aliases[0]
    ids = pipeline_ids(nodes[0], inputs)
    results, join = {}, None
    # a run of one kind and one pass group (PackSetPasses) shares passes
    for (kind, _), run in groupby(
            nodes, key=lambda node: (_kind(node), getattr(node, "pass_group", -1))):
        run = list(run)
        if kind == "filter":
            run_inputs = filter_inputs(context.state, run[0], ids)
            if run[-1] is not nodes[-1]:
                run_inputs["limit"] = None
            results.update(execute_set_filters(state, run, run_inputs))
        elif kind == "extract":
            results.update(execute_set_extractions(state, run, {
                "document_ids": ids, "documents": context.state["documents"],
                "texts": context.state.get("texts", {})}))
        elif kind == "label":
            every = classify_passes(state, [node.spec for node in run], ids,
                                    context.state["documents"])
            for node, labels in zip(run, every):
                priors = [value for port, value in zip(
                    node.inputs, _node_inputs(node, results, inputs).values())
                    if port.value_type is ValueType.ROWS]
                results[node.node_id] = classify_result(node, ids, labels, priors)
        elif kind == "condition":
            for node in run:
                results[node.node_id] = FilterRuntime().execute(
                    node, _node_inputs(node, results, inputs), context)
        elif kind == "join":
            join = JoinPart.prepare(state, run, ids, [], inputs, context)
            results.update(execute_set_joins(state, join))
        else:
            for node in run:
                results[node.node_id] = joined_label_passes(state, node, join)
            continue
        last = results[run[-1].node_id].outputs.get(f"rows:{alias}")
        if last is not None:
            ids = id_positions(last.ids[alias])
    return results
