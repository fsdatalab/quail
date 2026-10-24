"""AI.SAE_FEATURES through the shared document scheduler."""

import numpy as np
import pyarrow as pa

from quail.backends.quail.executor.parts import SingleResultPart
from quail.backends.quail.executor.sae import (
    SaeFeaturesReadout,
    SaeMaxAccumulator,
    feature_lists_of_rows,
    sae_encoder,
)
from quail.backends.quail.executor.stages import Stage
from quail.backends.quail.reranker import classify_outputs
from quail.catalog.tokens import DocumentKeys
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import Rows
from quail.logical import FEATURE_LIST
from quail.physical import ValueType


def _feature_column(lengths, indices, strengths) -> pa.ListArray:
    """Build the feature-list column from flat index and strength arrays.

    ``lengths`` gives each row's feature count; the arrays hold every
    row's features in sequence. The Arrow arrays wrap the numpy ones
    without a copy.
    """
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    values = pa.StructArray.from_arrays(
        [pa.array(np.ascontiguousarray(indices, dtype=np.int32)),
         pa.array(np.ascontiguousarray(strengths, dtype=np.float32))],
        names=FEATURE_LIST.value_type.names)
    return pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)


class SaeFeaturesPart(SingleResultPart):
    """One feature call's stage and result table within a pipeline."""

    @staticmethod
    def model_inputs(node, inputs, context, sequence=None, nodes=None):
        if nodes is not None:
            inputs = inputs[node.node_id]
        # the one input carries the documents' ids, or the rows of an
        # earlier operator over them
        (value,) = inputs.values()
        ids = (value.ids[node.spec.aliases[0]].to_numpy()
               if isinstance(value, Rows)
               else np.asarray(value, dtype=np.int64))
        return {"document_ids": ids, "documents": context.state["documents"]}

    @classmethod
    def prepare(cls, state, nodes, ids, parts, inputs, context):
        node = nodes[0]
        accumulator = next(
            (part.accumulator for part in parts
             if isinstance(part, SaeFeaturesPart)
             and part.accumulator is not None), None)
        priors = cls._priors(node, parts)
        prior_tables = [inputs[node.node_id][port.name] for port in node.inputs
                        if port.value_type is ValueType.ROWS
                        and all(port.source.node_id != member.node_id
                                for part in parts for member in part.nodes)]
        return cls(state, node, ids, context.state["documents"],
                   accumulator=accumulator, priors=priors,
                   prior_tables=prior_tables)

    @staticmethod
    def _priors(node, parts):
        by_node = {member.node_id: part for part in parts
                   for member in part.nodes}
        return [(by_node[port.source.node_id], port.source)
                for port in node.inputs
                if port.value_type is ValueType.ROWS
                and port.source.node_id in by_node]

    def __init__(self, state, node, ids, documents, *, accumulator=None,
                 priors=(), prior_tables=()):
        spec = node.spec
        (alias,) = spec.aliases
        head, tail = spec.prompt_token_parts
        if not tail:
            raise ValueError(
                "AI.SAE_FEATURES needs a question after the document")
        self.node = node
        self.ids = list(ids)
        self.priors = priors
        self.prior_tables = prior_tables
        self.values = [None] * len(self.ids)
        self.pending = {}        # anchor -> settled handle not yet read
        self.harvested = []      # (anchors, lengths, indices, strengths)
        self.reached = 0
        torch = state.torch
        encoder = sae_encoder(state)
        self.keep = encoder.keep
        keys = DocumentKeys(alias, self.ids)
        self.accumulator = None
        self._pipeline = None
        if spec.pooling == "max":
            if accumulator is None:
                table = documents[alias]
                lengths = {}

                def document_tokens(key, head_tokens=len(head)):
                    if key not in lengths:
                        lengths[key] = head_tokens + len(table[key[1]])
                    return lengths[key]

                accumulator = SaeMaxAccumulator(torch, encoder,
                                                document_tokens)
                self._pipeline = state.loaded_model.pipeline
                self._pipeline.forward_hook = accumulator.observe
            self.accumulator = accumulator
            self.accumulator.register_call()

            def decide(anchor, row):
                self.pending[anchor] = self.accumulator.settle(keys[anchor])
                self.reached += 1
                return True
        else:
            # the packed readout row (indices, then strengths) stays a
            # view; every row unpacks at once in result()
            def decide(anchor, row):
                self.values[anchor] = np.asarray(row)[-1]
                self.reached += 1
                return True

        self.stages = [Stage(
            suffixes=[[tail[-1]]], readout=SaeFeaturesReadout(torch, encoder),
            frame=list(tail[:-1]), decide=decide, single=True,
            label=f"sae features {spec.name}")]

    @property
    def tested_columns(self) -> tuple[str, ...]:
        return ()

    def finish(self, every) -> int:
        """Read the settled maxima that have landed; return the suffix tokens."""
        self._harvest(completed_only=True)
        head, tail = self.node.spec.prompt_token_parts
        return self.reached * len(tail)

    def _harvest(self, *, completed_only: bool) -> None:
        """Unpack pending handles slab by slab, never waiting mid-run.

        finish() runs between chunks, so it reads only slabs whose
        copies have landed and the GPU keeps working underneath;
        result() reads the rest once the pipeline is done.
        """
        if self.accumulator is None or not self.pending:
            return
        anchors = np.fromiter(self.pending, np.int64, len(self.pending))
        taken, lengths, indices, strengths = self.accumulator.harvest(
            [self.pending[anchor] for anchor in anchors],
            completed_only=completed_only)
        if not taken.size:
            return
        read = anchors[taken]
        self.harvested.append((read, lengths, indices, strengths))
        for anchor in read.tolist():
            del self.pending[anchor]

    def complete(self, tokens, spans, stats, offset):
        # the part that set the hook releases it once the run ends
        if self._pipeline is not None:
            self._pipeline.forward_hook = None
            self._pipeline = None
        super().complete(tokens, spans, stats, offset)

    def result(self, tokens, gpu_s, chunks, stats) -> NodeResult:
        spec = self.node.spec
        if self.accumulator is not None:
            self._harvest(completed_only=False)
            batches = self.harvested or [(np.empty(0, np.int64),) * 2 + (
                np.empty(0, np.int32), np.empty(0, np.float32))]
            answered, lengths, indices, strengths = (
                np.concatenate(column) for column in zip(*batches))
            answered = answered.tolist()
        else:
            answered = [index for index, value in enumerate(self.values)
                        if value is not None]
            lengths, indices, strengths = feature_lists_of_rows(
                [self.values[index] for index in answered],
                self.keep)
        rows = np.asarray([[self.ids[index]] for index in answered],
                          dtype=np.int32).reshape(-1, 1)
        (alias,) = spec.aliases
        listed = Rows({alias: rows[:, 0]}, pa.table({
            spec.name: _feature_column(lengths, indices, strengths)}))
        prior_results = (part.results()[port.node_id].outputs[port.port]
                         for part, port in self.priors)
        for prior in (*self.prior_tables, *prior_results):
            listed = listed.joined(prior, spec.aliases[0])
        return NodeResult(classify_outputs(self.node, listed), NodeMetrics(
            input_rows=self.reached, output_rows=len(answered),
            evaluated_documents=self.reached, fresh_tokens=tokens,
            gpu_s=gpu_s, chunks=chunks,
            extension={
                "output": spec.name, "aliases": list(spec.aliases),
                "pooling": spec.pooling,
                "input_rows": self.reached,
                "suffix_tokens": len(spec.prompt_token_parts[1]),
                "borrowed_prefix_tokens": stats.get("borrowed_tokens", 0),
                "pack_s": round(stats.get("pack_s", 0.0), 3),
            }))
