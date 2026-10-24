"""Run a pipeline's maps and build their answer tables."""

import time
from dataclasses import replace

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.extract import load_vocabulary
from quail.backends.quail.executor.map import MapStages
from quail.backends.quail.executor.parts import (
    DocumentAnswerPart,
    execute_parts,
    gpu_seconds,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.state import QueryExecutionState
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import map_rows
from quail.physical import AiMap


def execute_maps(state: QueryExecutionState, nodes, inputs) -> dict:
    """Run a pipeline's maps over its documents as one stage run.

    Each document is read once and put every node's instruction while
    its KV is resident. The first node is charged the documents' tokens
    and the wall time. Each instruction is charged its own suffix
    tokens and GPU time.

    Args:
        state: The loaded model and current query state.
        nodes: The AiMap nodes, in pipeline order.
        inputs: The documents' ids, the tokenized documents and texts
            by alias, the query's shared preamble, and the gpu_timing
            setting.

    Returns:
        Node id -> NodeResult with the node's answer table.

    Raises:
        TypeError: A node is not a map.
    """
    state.gpu_timing = bool(inputs.get("gpu_timing", False))
    started = time.perf_counter()
    part = MapPart(state, nodes, inputs["document_ids"], inputs)
    stages = part.mapping
    tree = (shared_prefix_tree(nodes[0].spec.name, stages.prefixes,
                               state.loaded_model.arena)
            if nodes[0].spec.share_prefixes else None)
    results = execute_parts(
        state, [part], stages.prefixes, stages.keys,
        keep_kv=nodes[-1].keep_kv, prefix_tree=tree, limit=inputs.get("limit"),
        label=f"map {', '.join(node.spec.name for node in nodes)}")
    first = nodes[0].node_id
    results[first] = replace(results[first], metrics=replace(
        results[first].metrics, wall_s=time.perf_counter() - started))
    return results


def map_results(nodes, stages, document_ids, *, fresh, own,
                stats, counts, kept=None) -> dict:
    """Build each map node's answer table and metrics.

    Args:
        nodes: The AiMap nodes, in instruction order.
        stages: The finished MapStages.
        document_ids: The documents' ids.
        fresh: The run's fresh tokens; the first node is charged them
            less the later instructions' suffix tokens.
        own: Instruction index -> (its GPU seconds, its timed chunks).
        stats: The run's packing time and borrowed tokens.
        counts: Per-instruction counts collected when answers finished.
        kept: The indices of the documents the answer tables hold;
            None holds every document.
    """
    later = sum(count["suffix_tokens"] for count in counts[1:])
    if kept is not None:
        document_ids = document_ids[kept]
    results = {}
    for q, node in enumerate(nodes):
        spec = node.spec
        answers = map_rows(spec.alias, spec.name, document_ids,
                           stages.answers(q, kept))
        first = q == 0
        gpu_s, timed = own(q)
        results[node.node_id] = NodeResult(
            {node.rows_port: answers},
            NodeMetrics(
                input_rows=len(document_ids), output_rows=len(document_ids),
                evaluated_documents=len(document_ids),
                fresh_tokens=(fresh - later if first
                              else counts[q]["suffix_tokens"]),
                gpu_s=gpu_s, chunks=timed,
                extension={"output": spec.name, "aliases": list(spec.aliases),
                           "input_rows": len(document_ids), **counts[q],
                           "borrowed_prefix_tokens": (
                               stats.get("borrowed_tokens", 0) if first else 0),
                           "pack_s": (round(stats.get("pack_s", 0.0), 3)
                                      if first else 0.0)}))
    return results


class MapPart(DocumentAnswerPart):
    """Map stages and per-instruction results for the shared executor."""

    combine = True
    execute_group = staticmethod(execute_maps)

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        """Prepare consecutive instructions over the preceding operator's documents.

        Raises:
            ValueError: A document uses a different prompt or token sequence.
        """
        first = parts[0].node if parts else nodes[0]
        part = cls(state, nodes, ids, {
            "documents": context.state["documents"],
            "texts": context.state.get("texts", {}),
            "pipeline_head": first.document_head})
        alias = part.node.spec.alias
        for document, document_id in zip(part.mapping.documents, part.ids):
            if document.key != (alias, document_id):
                raise ValueError(
                    f"AI.MAP {part.node.spec.name!r} shows document "
                    f"{document_id} of {alias!r} unlike the pipeline before it")
        return part

    def __init__(self, state, nodes, ids, inputs):
        self.nodes = list(nodes)
        self.node = self.nodes[0]
        self.ids = np.asarray(ids, dtype=np.int64)
        if any(not isinstance(node, AiMap) for node in nodes):
            raise TypeError("a map pipeline holds AiMap nodes only")
        alias = self.node.spec.alias
        self.mapping = MapStages(
            state, [node.spec for node in nodes], load_vocabulary(state), self.ids,
            inputs["documents"][alias], inputs.get("texts", {}).get(alias),
            pipeline_head=inputs.get("pipeline_head", self.node.document_head),
            document_positions=inputs.get("document_positions"))
        self.stages = self.mapping.stages
        self._results = {}

    @property
    def appended_tokens(self) -> int:
        return self.mapping.appended_tokens

    @property
    def tested_columns(self) -> tuple[str, ...]:
        """The columns a Filter in the pipeline may test."""
        return tuple(node.spec.name for node in self.nodes)

    def tested_values(self, column: str, positions) -> pa.Array:
        """Return the answers settled so far at the positions.

        Args:
            column: A map's answer column.
            positions: Positions among the part's documents.
        """
        q = next(q for q, node in enumerate(self.nodes)
                 if column == node.spec.name)
        return pa.array(
            [self.mapping.documents[position].answers[q].answer
             for position in positions], pa.string())

    def finish(self, every) -> int:
        """Settle every answer; return the tokens the instructions fed."""
        self.mapping.finish()
        self.counts = [self.mapping.counts(q) for q in range(len(self.nodes))]
        return sum(count["suffix_tokens"] for count in self.counts)

    def complete(self, fresh, spans, stats, offset):
        """Build each instruction's answer table and attribute its GPU work."""
        per_instruction = len(self.stages) // len(self.nodes)

        def own(q):
            start = offset + q * per_instruction
            selected = stage_spans(spans, start, start + per_instruction)
            return gpu_seconds(selected), len(selected)

        self._results = map_results(
            self.nodes, self.mapping, self.ids, fresh=fresh, counts=self.counts,
            own=own, stats=stats, kept=self.mapping.asked())
        self.result_value = self._results[self.node.node_id]

    def results(self) -> dict:
        """Return the answer table and metrics for each map."""
        return self._results
