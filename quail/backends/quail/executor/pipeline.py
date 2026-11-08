"""Run a chain's steps under set attention, as passes in one GPU loop.

Under set attention every forward-pass sequence is one pass: a state,
then the questions asked of it. A chain's GPU steps are pass groups
(PackSetPasses), joins, and classifications of a join's pairs; each
asks its passes as anchors of the chain's one SetPasses run, at its own
stages. A step's passes join the loop once the step before it has
answered, since a step asks only the rows the steps before it kept.
"""

import numpy as np

from quail.backends.quail.executor.operators.filter import (
    filter_result,
    filter_stop_groups,
)
from quail.backends.quail.executor.operators.join import set_joins, upstream_join
from quail.backends.quail.executor.operators.set_classify import (
    SetLabels,
    classify_result,
    joined_label_stage,
    joined_labels,
    score_result,
)
from quail.backends.quail.executor.operators.set_extract import (
    SetReads,
    found_reads,
    read_result,
    read_texts,
)
from quail.backends.quail.executor.pass_groups import PassGroup, node_question
from quail.backends.quail.executor.readout import AsyncSlotMargins
from quail.backends.quail.executor.region import pipeline_ids, pipeline_rows
from quail.backends.quail.executor.set_passes import SetPasses
from quail.common.tokens import DocumentPrefixes
from quail.execution.finished import report_outputs
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


def _join_run(nodes, index) -> list:
    """The consecutive joins starting at nodes[index]."""
    run = []
    for node in nodes[index:]:
        if _kind(node) != "join":
            break
        run.append(node)
    return run


def set_steps(state, nodes, ids, prefixes, ask, *, texts=None, inputs=None,
              context=None):
    """Run a pipeline's steps under set attention, in order; yields their passes.

    The operators of one pass group (PackSetPasses) are asked in one
    pass per document when the pipeline reaches the group's first
    operator. Each step then reads its answers for the rows still kept:
    a filter keeps the rows it passes, a condition the rows it accepts,
    and a score, a classification, or an extraction answers every kept
    row. Each node's rows are reported as soon as its result is built.
    A question asked of a row that a filter of the same group removes is
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
        ask: What asks the model: group(pass group, kept positions),
            joins(nodes, anchor ids), and joined(node, join), each a
            generator of Passes returning GroupAnswers, the
            JoinOperator, or the NodeResult.
        texts: The table's texts by document id, for extractions.
        inputs: Per node id, its input values from outside the pipeline.
        context: The execution context, for conditions and joins.

    Returns:
        Node id -> NodeResult.
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
                group = yield from ask.group(node.pass_group, kept)
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
                report_outputs(node, results[node.node_id])
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
            report_outputs(node, results[node.node_id])
            kept = asked
            continue
        if kind == "condition":
            results[node.node_id] = FilterRuntime().execute(
                node, _node_inputs(node, results, inputs), context)
        elif kind == "join":
            if index and _kind(nodes[index - 1]) == "join":
                continue
            run = _join_run(nodes, index)
            join = yield from ask.joins(run, ids[kept])
            for member in run:
                results[member.node_id] = join.results()[member.node_id]
                report_outputs(member, results[member.node_id])
            node = run[-1]
        else:
            results[node.node_id] = yield from ask.joined(node, join)
            report_outputs(node, results[node.node_id])
            continue
        rows = results[node.node_id].outputs.get(f"rows:{alias}")
        if rows is not None:
            kept = _positions(ids, id_positions(rows.ids[alias]))
    return results


class SetChain:
    """A chain under set attention: every step's passes in one SetPasses run.

    The chain's stages are added before the run, in step order: one per
    pass of each pass group, one per join, and one per classification
    of a join's pairs. Each step reports its nodes' rows as it finishes.

    Args:
        state: The query's execution state.
        nodes: The chain's nodes, in order.
        inputs: Per node id, its input values from outside the chain.
        context: The execution context.
    """

    def __init__(self, state, nodes, inputs, context):
        self.state = state
        self.nodes = tuple(nodes)
        self.inputs = inputs
        self.context = context
        first = nodes[0]
        alias = first.document_aliases[0]
        self.ids = pipeline_ids(first, inputs)
        head = next((node.document_head for node in nodes
                     if _kind(node) in PACKED), ())
        self.prefixes = DocumentPrefixes(head, context.state["documents"][alias],
                                         self.ids)
        self.texts = context.state.get("texts", {}).get(alias)
        self.run = SetPasses(state)
        self.groups = {}
        self.stages = {}        # join or joined classification -> its stage
        readout = None
        for node in nodes:
            kind = _kind(node)
            if kind in PACKED and node.pass_group not in self.groups:
                members = [member for member in nodes if _kind(member) in PACKED
                           and member.pass_group == node.pass_group]
                self.groups[node.pass_group] = PassGroup(
                    self.run, [node_question(member) for member in members],
                    self.prefixes, f"pass group ({len(members)} questions)")
            elif kind == "join":
                readout = readout or AsyncSlotMargins(state.torch, state.answer_rows)
                self.stages[node.node_id] = self.run.own_stage(
                    readout, f"join predicate {node.written_position}")
            elif kind == "joined":
                self.stages[node.node_id] = joined_label_stage(self.run, node)
        # a limit and stop keys apply to the pass group that ends the
        # chain, when it is one pass
        last = self.nodes[-1]
        ending = (self.groups.get(last.pass_group) if _kind(last) in PACKED
                  else None)
        self.ending = ending if ending and len(ending.passes) == 1 else None
        first_filter = next((node for node in nodes if isinstance(node, AiFilter)),
                            None)
        self.stop_groups = (None if first_filter is None or self.ending is None
                            else filter_stop_groups(
                                context.state, first_filter,
                                pipeline_rows(first, inputs)
                                if first_filter is first else self.ids))
        self.num_key_survivors = (1 if first_filter is None
                                  else first_filter.num_key_survivors)

    def group(self, pass_group, kept):
        """Ask a pass group of the kept documents; yields its Passes."""
        group = self.groups[pass_group]
        stops = (None if self.stop_groups is None or group is not self.ending
                 else [self.stop_groups[int(p)] for p in kept])
        return (yield from group.ask([self.prefixes[int(p)] for p in kept], stops))

    def joins(self, nodes, anchor_ids):
        """Ask a group of joins of the anchors; yields their Passes."""
        join = upstream_join(self.state, nodes, anchor_ids, (), self.inputs,
                             self.context)
        yield from set_joins(self.run, [self.stages[node.node_id] for node in nodes],
                             join)
        return join

    def joined(self, node, join):
        """Classify a join's kept pairs; yields their Passes."""
        return (yield from joined_labels(self.run, self.stages[node.node_id], node,
                                         join))

    def execute(self) -> dict:
        """Run the chain's steps in one GPU loop; return node id -> NodeResult."""
        steps = set_steps(self.state, self.nodes, self.ids, self.prefixes, self,
                          texts=self.texts, inputs=self.inputs,
                          context=self.context)
        return self.run.run(
            steps, label=f"set pipeline {self.nodes[0].document_aliases[0]}",
            limit=(self.context.state.get("filter_limit")
                   if self.ending is not None else None),
            stop_keys=self.stop_groups is not None,
            num_key_survivors=self.num_key_survivors)
