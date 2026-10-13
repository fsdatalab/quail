"""Prepare join requests, retain matching pairs, and collect predicate results."""

from functools import partial

import numpy as np

from quail.backends.quail.executor.parts import (
    ExecutionPart,
    execute_parts,
    execution_prefix_tree,
    gpu_seconds,
    stage_spans,
)
from quail.backends.quail.executor.stages import Stage
from quail.backends.quail.executor.state import QueryExecutionState
from quail.backends.quail.graph import stage_partner_lists
from quail.execution.runner import NodeMetrics, NodeResult
from quail.execution.tokens import DocumentPrefixes


def execute_join(state: QueryExecutionState, node, inputs) -> NodeResult:
    """Run one join predicate."""
    return execute_joins(state, (node,), inputs)[node.node_id]


def execute_joins(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run consecutive join predicates through the document scheduler."""
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    part = JoinPart(state, nodes, inputs, inputs["anchor_ids"])
    return execute_parts(
        state, [part], inputs["prefixes"], inputs["anchor_keys"],
        keep_kv=nodes[-1].keep_kv,
        tree=execution_prefix_tree(nodes[0], inputs["prefixes"],
                                   state.loaded_model.arena),
        label=f"join ({len(nodes)} predicates)")


class JoinPart(ExecutionPart):
    """Join predicates and their anchor KV within one scheduler run."""

    combine = True

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Bind partner inputs and any pairs produced by preceding operators."""
        from quail.backends.quail.executor.operators.apply import ApplyGate

        first = parts[0].node if parts else nodes[0]
        head = first.document_head
        prefixes = DocumentPrefixes(
            context.state["pre"] if head is None else head,
            context.state["docs"][nodes[0].anchor], ids)
        chain = {"documents": prefixes, "document_ids": ids, "parts": tuple(parts),
                 "pairs": {part.node.written_pos: part.rows for part in parts
                           if isinstance(part, ApplyGate) and part.node.ids == "pairs"}}
        prepared = context.model_inputs(nodes[0], inputs, context, chain, nodes=nodes)
        return cls(state, nodes, prepared, ids, context=context)

    def __init__(self, state, nodes, inputs, ids, *, context=None):
        self.nodes = nodes
        self.node = nodes[0]
        self.inputs = inputs
        self.ids = ids
        self.context = context
        self.canvas_tokens = len(state.loaded_model.pipeline.canvas_ids)
        self.attention = self.node.attention or None
        self.settled = set()
        self.final_rows = {}
        self.readers = []
        lists_for = inputs.get("anchor_partners")
        self.stages = [
            Stage(suffixes=suffixes, readout=state.async_answers, frame=frame,
                  requests=(None if lists_for is None
                            else lambda key, j=j: lists_for(key[1], j)),
                  label=f"join predicate {j}")
            for j, (suffixes, frame) in enumerate(
                zip(inputs["stage_suffixes"], inputs["stage_frames"]))]
        self._results = {}

    def add_reader(self, kept):
        """Expose surviving pairs to a following classification."""
        self.readers.append(kept)
        for index in (kept.stage, len(self.stages) - 1):
            self.stages[index].decide = partial(self._record, index)

    def _record(self, index, anchor, row):
        lists_for = self.inputs.get("anchor_partners")
        key = self.inputs["anchor_keys"][anchor]
        indices = None if lists_for is None else lists_for(key[1], index)
        for reader in self.readers:
            if reader.stage == index:
                reader.record(anchor, row, indices)
        if index == len(self.stages) - 1:
            self.final_rows[anchor] = row
        return bool(any(row))

    def on_settled(self, anchor, survived, row):
        """Settle anchor KV using join answers after all readers finish."""
        self.settled.add(anchor)
        self.inputs["anchor_done"](anchor, self.final_rows.get(anchor, row))

    def finish(self, every) -> int:
        """Map predicate answers to reached anchors and count suffix work."""
        self.own_tokens = _predicate_tokens(
            self.stages, every, self.inputs.get("anchor_partners"),
            self.inputs["anchor_keys"], self.canvas_tokens)
        chain = self.inputs.get("chain")
        if chain is not None:
            reached = sorted(set(every[0] if every else ()) | self.settled)
            local_of = {position: local for local, position in enumerate(reached)}
            self.answers = [
                {local_of[position]: row for position, row in answers.items()}
                for answers in every]
            self.anchor_ids = [self.ids[position] for position in reached]
            self.kv_round = ({"hits": len(reached), "misses": 0} if chain["parts"]
                             else {"hits": 0, "misses": len(reached)})
        else:
            self.answers = every
            self.anchor_ids = list(self.ids)
            self.kv_round = self.inputs.get("kv_round") or {}
        return sum(self.own_tokens)

    def complete(self, tokens, spans, stats, offset):
        """Build each join predicate's answers and measured costs."""
        nodes, inputs = self.nodes, self.inputs
        node = self.node
        anchor_ids, answers = self.anchor_ids, self.answers
        lists_for = inputs.get("anchor_partners")
        own_tokens = list(self.own_tokens)
        kv_round = self.kv_round
        group = inputs["group"]
        partner_lists = stage_partner_lists(group, lists_for, anchor_ids)
        results = {}
        own_tokens[0] = tokens - sum(own_tokens[1:])
        for j, (member, rows, members) in enumerate(zip(nodes, answers, partner_lists)):
            reached_ids = [anchor_ids[int(local)] for local in sorted(rows)]
            matched = [anchor_ids[int(local)] for local in sorted(rows)
                       if any(rows[local])]
            survivors = matched
            # Anchors with no candidates never launch a request.
            if member.semantics == "anti":
                candidates = anchor_ids if j == 0 else results[
                    nodes[j - 1].node_id].outputs[f"ids:{node.anchor}"]
                candidates = np.asarray(candidates, dtype=np.int64)
                survivors = candidates[~np.isin(candidates, matched)].tolist()
            answer = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": inputs["partner_indices"][member.written_pos],
                "anchor_partners": members, "anchor": member.anchor,
                "partners": list(member.partners), "semantics": member.semantics,
                "selectivity": member.selectivity, "written_pos": member.written_pos,
            }
            own_spans = stage_spans(spans, offset + j, offset + j + 1)
            results[member.node_id] = NodeResult(
                outputs={f"ids:{member.anchor}": survivors,
                         f"join_answers:{member.written_pos}": answer},
                metrics=NodeMetrics(
                    input_rows=len(anchor_ids) if j == 0 else len(reached_ids),
                    output_rows=len(survivors),
                    evaluated_document_pairs=sum(len(row) for row in rows.values()),
                    kv_hits=kv_round.get("hits", 0) if j == 0 else len(rows),
                    kv_misses=kv_round.get("misses", 0) if j == 0 else 0,
                    fresh_tokens=own_tokens[j],
                    gpu_s=gpu_seconds(own_spans),
                    chunks=len(own_spans),
                    extension={
                        "answers": [rows],
                        **({"borrowed_prefix_tokens": stats["borrowed_tokens"]}
                           if j == 0 and stats.get("borrowed_tokens") else {}),
                        **({"pack_s": round(stats["pack_s"], 3)}
                           if j == 0 and stats.get("pack_s") else {}),
                    }))
        self._results = results
        self.result_value = results[node.node_id]
        if self.context is not None:
            last = nodes[-1]
            self.context.model_result(last, results[last.node_id], self.context)

    def results(self) -> dict:
        """Return results for every join predicate in the group."""
        return self._results


def _predicate_tokens(join_stages, answers, lists_for, anchor_keys,
                      canvas_tokens) -> list[int]:
    """Count each predicate's fresh frame and partner suffix tokens."""
    totals = []
    previous = None
    for j, (stage, rows) in enumerate(zip(join_stages, answers)):
        frame = list(stage.frame)
        writes = bool(frame) and frame != previous
        previous = frame
        lengths = np.fromiter(
            (len(suffix) + canvas_tokens for suffix in stage.suffixes),
            dtype=np.int64, count=len(stage.suffixes))
        suffix_tokens = int(lengths.sum())
        total = len(rows) * (len(frame) + canvas_tokens if writes else 0)
        if lists_for is None:
            total += len(rows) * suffix_tokens
        else:
            for anchor in rows:
                indices = lists_for(anchor_keys[anchor][1], j)
                total += (suffix_tokens if indices is None
                          else int(lengths[indices].sum()))
        totals.append(total)
    return totals
