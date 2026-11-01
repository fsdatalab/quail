"""Run compatible physical nodes through the shared document scheduler."""

from itertools import groupby

import numpy as np

from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.operators.filter import (
    execute_set_filters,
    filter_inputs,
)
from quail.backends.quail.executor.operators.set_classify import (
    classify_passes,
    classify_result,
)
from quail.backends.quail.executor.operators.set_extract import (
    execute_set_extractions,
)
from quail.backends.quail.executor.parts import execute_parts, execution_prefix_tree
from quail.backends.quail.executor.state import QueryExecutionState
from quail.catalog.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.physical import AiClassify, AiExtract, AiFilter, AiJoin
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


def execute_set_pipeline(state, nodes, inputs, context) -> dict:
    """Run a pipeline of AI filters, classifications, and extractions.

    Under set attention, consecutive operators of one kind share their
    passes; each run reads the rows the run before it kept.

    Raises:
        ValueError: The pipeline holds another operator.
    """
    alias = nodes[0].document_aliases[0]
    ids = pipeline_ids(nodes[0], inputs)
    runs = []
    for node in nodes:
        if not isinstance(node, (AiFilter, AiClassify, AiExtract)):
            raise ValueError(f"a set-attention pipeline runs AI filters, "
                             f"classifications, and extractions, not "
                             f"{node.type_name!r}")
        if runs and type(runs[-1][-1]) is type(node):
            runs[-1].append(node)
        else:
            runs.append([node])
    results = {}
    for index, run in enumerate(runs):
        if isinstance(run[0], AiFilter):
            run_inputs = filter_inputs(context.state, run[0], ids)
            if index < len(runs) - 1:
                run_inputs["limit"] = None
            results.update(execute_set_filters(state, run, run_inputs))
            ids = id_positions(results[run[-1].node_id].outputs[
                f"rows:{alias}"].ids[alias])
            continue
        if isinstance(run[0], AiExtract):
            results.update(execute_set_extractions(state, run, {
                "document_ids": ids, "documents": context.state["documents"],
                "texts": context.state.get("texts", {})}))
            continue
        every = classify_passes(state, [node.spec for node in run], ids,
                                context.state["documents"])
        for node, labels in zip(run, every):
            priors = [
                results[port.source.node_id].outputs[port.source.port]
                if port.source.node_id in results
                else inputs[node.node_id][port.name]
                for port in node.inputs if port.value_type is ValueType.ROWS]
            results[node.node_id] = classify_result(node, ids, labels, priors)
    return results
