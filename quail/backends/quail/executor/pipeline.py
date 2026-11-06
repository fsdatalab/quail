"""Run compatible physical nodes through the shared document scheduler."""

from itertools import groupby

import numpy as np

from quail.backends.quail.executor.operators import execution_parts
from quail.backends.quail.executor.operators.filter import (
    filter_inputs,
    filter_result,
)
from quail.backends.quail.executor.operators.join import JoinPart, execute_set_joins
from quail.backends.quail.executor.operators.set_classify import (
    SetLabels,
    classify_result,
    joined_label_passes,
    score_result,
)
from quail.backends.quail.executor.operators.set_extract import (
    SetReads,
    found_reads,
    read_result,
    read_texts,
)
from quail.backends.quail.executor.parts import execute_parts, execution_prefix_tree
from quail.backends.quail.executor.pass_groups import node_question, run_pass_group
from quail.backends.quail.executor.state import QueryExecutionState
from quail.common.tokens import DocumentKeys, DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.rows import Rows
from quail.execution.runtimes.filter import FilterRuntime
from quail.physical import AiClassify, AiExtract, AiFilter, AiJoin, AiScore, Filter
from quail.physical.base import ValueType


def execute_pipeline(state: QueryExecutionState, nodes, inputs, context) -> dict:
    """Prepare compatible operators, run their stages, and return node results."""
    state.gpu_timing = bool(context.state.get("gpu_timing", False))
    # a set-attention model runs every pipeline as passes over whole
    # sequences; this is the one place that choice is made
    if state.loaded_model.pipeline.whole_sequences:
        return execute_set_pipeline(state, nodes, inputs, context)
    first = nodes[0]
    implementations = execution_parts()
    implementation = implementations[first.type_name]
    if (implementation.execute_group is not None
            and all(node.type_name == first.type_name for node in nodes)):
        prepared = context.model_inputs(first, inputs, context, nodes=nodes)
        return implementation.execute_group(state, nodes, prepared)

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
    if isinstance(node, AiScore):
        return "score"
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


PACKED = ("filter", "score", "label", "extract")


def _positions(ids, chosen) -> np.ndarray:
    """The positions in ids of the chosen document ids."""
    order = np.argsort(ids, kind="stable")
    return np.sort(order[np.searchsorted(ids[order], chosen)])


def execute_set_steps(state, nodes, ids, prefixes, *, texts=None, inputs=None,
                      context=None, limit=None, stop_groups=None) -> dict:
    """Run a pipeline's steps under set attention, in order.

    The operators of one pass group (PackSetPasses) are asked in one
    pass per document when the pipeline reaches the group's first
    operator. Each step then reads its answers for the rows still kept:
    a filter keeps the rows it passes, a condition the rows it accepts,
    and a score, a classification, or an extraction answers every kept
    row. A
    question asked of a row that a filter of the same group removes is
    answered and then dropped, so packing decides which rows each step
    reads the same way as running the steps apart. The answers can
    differ: d1 lets the state read a lone question but not the questions
    of a packed pass. A join's anchors are the rows kept so far, and the
    classifications of its pairs follow it.

    Args:
        state: The query's execution state.
        nodes: The pipeline's nodes, in order.
        ids: The document ids the pipeline reads.
        prefixes: Per position in ids, the prompt head and document tokens.
        texts: The table's texts by document id, for extractions.
        inputs: Per node id, its input values from outside the pipeline.
        context: The execution context, for conditions and joins.
        limit: When the last pass group ends the pipeline, stop once this
            many rows passed its filters.
        stop_groups: Per position in ids, its stop key group, with the limit.
    """
    ids = np.asarray(ids, dtype=np.int64)
    alias = nodes[0].document_aliases[0]
    kept = np.arange(len(ids))
    results, groups, join, inputs = {}, {}, None, inputs or {}
    model_spec = state.loaded_model.model_spec
    for index, node in enumerate(nodes):
        kind = _kind(node)
        if kind in PACKED:
            if node.pass_group not in groups:
                members = [member for member in nodes if _kind(member) in PACKED
                           and member.pass_group == node.pass_group]
                ends = members[-1] is nodes[-1]
                group = run_pass_group(
                    state, [node_question(member) for member in members],
                    [prefixes[int(position)] for position in kept],
                    limit=limit if ends else None,
                    stop_groups=(None if stop_groups is None or not ends
                                 else [stop_groups[int(p)] for p in kept]),
                    label=f"pass group ({len(members)} questions)")
                groups[node.pass_group] = (members, group, kept[group.positions])
            members, group, answered = groups[node.pass_group]
            # the kept rows the group answered, and their answers
            at = np.searchsorted(answered, kept)
            hit = at < len(answered)
            hit[hit] = answered[at[hit]] == kept[hit]
            asked, values = kept[hit], group.answers[members.index(node)][at[hit]]
            lead = node is members[0]
            cost = dict(tokens=group.tokens if lead else 0,
                        gpu_s=group.gpu_s if lead else 0.0,
                        chunks=group.chunks if lead else 0)
            if kind == "filter":
                passed = values > 0
                results[node.node_id] = filter_result(
                    node, dict(enumerate([int(bit)] for bit in passed)),
                    cost["tokens"], ids[asked], gpu_s=cost["gpu_s"],
                    chunks=cost["chunks"])
                kept = asked[passed]
                continue
            suffix_tokens = group.suffix_tokens[members.index(node)]
            priors = [] if kind == "extract" else [
                value for port, value in zip(
                    node.inputs, _node_inputs(node, results, inputs).values())
                if port.value_type is ValueType.ROWS]
            if kind == "score":
                results[node.node_id] = score_result(node, ids[asked], values,
                                                     priors, cost, suffix_tokens)
            elif kind == "label":
                labels = SetLabels(
                    np.asarray(node.spec.labels, dtype=object)[values.argmax(-1)],
                    values, suffix_tokens=suffix_tokens, **cost)
                results[node.node_id] = classify_result(node, ids[asked], labels,
                                                        priors)
            else:
                if texts is None:
                    raise ValueError(f"AI.EXTRACT reads the text of {alias!r}, "
                                     f"which was not loaded")
                found, keep = found_reads(values, model_spec.read_found)
                answers, characters = read_texts(
                    values, keep, [prefixes[int(p)] for p in asked],
                    [texts[int(ids[p])] for p in asked],
                    len(node.spec.prompt_token_parts[0]),
                    state.loaded_model.tokenizer.token_bytes())
                results[node.node_id] = read_result(node, ids[asked], SetReads(
                    answers, characters, list(found), suffix_tokens=suffix_tokens,
                    **cost))
            kept = asked
            continue
        if kind == "condition":
            results[node.node_id] = FilterRuntime().execute(
                node, _node_inputs(node, results, inputs), context)
        elif kind == "join":
            if index and _kind(nodes[index - 1]) == "join":
                continue
            run = [node]
            for later in nodes[index + 1:]:
                if _kind(later) != "join":
                    break
                run.append(later)
            join = JoinPart.prepare(state, run, ids[kept], [], inputs, context)
            results.update(execute_set_joins(state, join))
            node = run[-1]
        else:
            results[node.node_id] = joined_label_passes(state, node, join)
            continue
        rows = results[node.node_id].outputs.get(f"rows:{alias}")
        if rows is not None:
            kept = _positions(ids, id_positions(rows.ids[alias]))
    return results


def execute_set_pipeline(state, nodes, inputs, context) -> dict:
    """Run a pipeline under set attention (execute_set_steps)."""
    alias = nodes[0].document_aliases[0]
    ids = pipeline_ids(nodes[0], inputs)
    head = next((node.document_head for node in nodes
                 if _kind(node) in PACKED), ())
    first_filter = next((node for node in nodes if isinstance(node, AiFilter)), None)
    stops = (None if first_filter is None else filter_inputs(
        context.state, first_filter, ids)["stop_groups"])
    return execute_set_steps(
        state, nodes, ids, DocumentPrefixes(head, context.state["documents"][alias],
                                            ids),
        texts=context.state.get("texts", {}).get(alias), inputs=inputs,
        context=context, limit=context.state.get("filter_limit"),
        stop_groups=stops)
