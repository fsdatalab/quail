"""Run a pipeline's steps under set attention, as passes over whole sequences."""

import numpy as np

from quail.backends.quail.executor.operators.filter import (
    filter_result,
    filter_stop_groups,
)
from quail.backends.quail.executor.operators.join import (
    execute_set_joins,
    upstream_join,
)
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
from quail.backends.quail.executor.pass_groups import node_question, run_pass_group
from quail.backends.quail.executor.region import pipeline_ids, pipeline_rows
from quail.common.tokens import DocumentPrefixes
from quail.execution.ids import id_positions
from quail.execution.runtimes.filter import FilterRuntime
from quail.physical import (
    AiClassify,
    AiExtract,
    AiFilter,
    AiJoin,
    AiScore,
    Filter,
    ValueType,
)


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
                      context=None, limit=None, stop_groups=None,
                      num_key_survivors=1) -> dict:
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
        num_key_survivors: How many rows of a stop key group must pass
            before the rest of the group is skipped.
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
                    num_key_survivors=num_key_survivors,
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
            join = upstream_join(state, run, ids[kept], (), inputs, context)
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
    stops = (None if first_filter is None else filter_stop_groups(
        context.state, first_filter,
        pipeline_rows(nodes[0], inputs) if first_filter is nodes[0] else ids))
    num_key_survivors = (1 if first_filter is None
                         else first_filter.num_key_survivors)
    return execute_set_steps(
        state, nodes, ids, DocumentPrefixes(head, context.state["documents"][alias],
                                            ids),
        texts=context.state.get("texts", {}).get(alias), inputs=inputs,
        context=context, limit=context.state.get("filter_limit"),
        stop_groups=stops, num_key_survivors=num_key_survivors)
