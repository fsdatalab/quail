"""Run compatible physical nodes through the shared document scheduler."""

from itertools import groupby

from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.parts import execute_parts, execution_prefix_tree
from quail.backends.quail.executor.state import QueryExecutionState
from quail.catalog.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.rows import Rows


def execute_pipeline(state: QueryExecutionState, pipeline, inputs, context) -> dict:
    """Prepare compatible operators, run their stages, and return node results."""
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    first = pipeline[0]
    implementations = execution_parts()
    implementation = implementations[first.type_name]
    if (implementation.execute_group is not None
            and all(node.type_name == first.type_name for node in pipeline)):
        prepared = context.model_inputs(first, inputs[first.node_id], context)
        return implementation.execute_group(state, pipeline, prepared)

    alias = first.document_aliases[0]
    anchor_port = next((port.name for port in first.inputs
                        if port.source.port == f"ids:{alias}"), None)
    source = (inputs[first.node_id][anchor_port] if anchor_port is not None
              else next(iter(inputs[first.node_id].values())))
    ids = id_positions(source.ids[alias] if isinstance(source, Rows) else source)
    last = max(index for index, node in enumerate(pipeline)
               if node.backend is not None)
    members, trailing = pipeline[:last + 1], pipeline[last + 1:]

    def group_key(node):
        implementation = implementations[node.type_name]
        return node.type_name, None if implementation.combine else node.node_id

    parts = []
    for _, group in groupby(members, group_key):
        nodes = list(group)
        parts.append(implementations[nodes[0].type_name].prepare(
            state, nodes, ids, parts, inputs, context))
    prefixes = DocumentPrefixes(first.document_head, context.state["docs"][alias], ids)
    results = execute_parts(
        state, parts, prefixes, DocumentKeys(alias, ids), keep_kv=members[-1].keep_kv,
        tree=execution_prefix_tree(first, prefixes, state.loaded_model.arena),
        limit=context.state.get("filter_limit"),
        attention=getattr(first, "attention", None) or None,
        label=f"pipeline {alias}")
    for member in trailing:
        member_inputs = dict(inputs[member.node_id])
        for port in member.inputs:
            if port.source.node_id in results:
                member_inputs[port.name] = results[
                    port.source.node_id].outputs[port.source.port]
        results[member.node_id] = context.runtimes[
            member.runtime_key].execute(member, member_inputs, context)
    return results
