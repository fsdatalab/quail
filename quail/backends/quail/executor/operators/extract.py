"""Run a chain's extractions and build their answer tables."""


import numpy as np

from quail.backends.quail.executor.extract.stages import ExtractionInputs
from quail.backends.quail.executor.model_operator import (
    gpu_seconds,
    shared_prefix_tree,
    stage_spans,
)
from quail.backends.quail.executor.operators.map import MapOperator
from quail.execution.metrics import NodeMetrics, NodeResult
from quail.execution.rows import extraction_rows
from quail.physical import AiExtract


def extraction_results(nodes, stages, document_ids, *, fresh, own,
                       stats, counts, kept=None) -> dict:
    """Build each extraction node's answer table and metrics.

    Args:
        nodes: The AiExtract nodes, in question order.
        stages: The extraction operator with finished answers.
        document_ids: The documents' ids.
        fresh: The run's fresh tokens; the first node is charged them
            less the later questions' suffix tokens.
        own: Question index -> (its GPU seconds, its timed chunks).
        stats: The run's packing time and borrowed tokens.
        counts: Per-question counts collected when answers finished.
        kept: The indices of the documents the answer tables hold;
            None holds every document.
    """
    later = sum(count["suffix_tokens"] for count in counts[1:])
    if kept is not None:
        document_ids = document_ids[kept]
    results = {}
    for q, node in enumerate(nodes):
        spec = node.spec
        answers, answer_spans = stages.answers(q, kept)
        answers = extraction_rows(spec.alias, spec.name, spec.span_name,
                                document_ids, answers, answer_spans)
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


class ExtractOperator(MapOperator):
    """Extraction fields decoded as document branches by MapDecoder.

    Extractions alone show each document as the vocabulary encodes its
    text, under the extraction's head; after other operators, under the
    chain's head, so they read the KV those operators wrote.
    """

    word = "extract"

    @classmethod
    def build(cls, region, nodes, chain):
        texts = region.context.state.get("texts", {})
        documents = region.documents
        if region.one_family:
            return cls(region.query, nodes, region.document_ids(),
                       {"documents": documents, "texts": texts})
        operator = cls(region.query, nodes, region.document_ids(), {
            "documents": documents, "texts": texts,
            "pipeline_head": (chain[0].node if chain else nodes[0]).document_head},
            first=sum(len(operator.answer_limits) for operator in chain))
        alias = operator.node.spec.alias
        if any(document.key != (alias, document_id) for document, document_id
               in zip(operator.extraction.documents, operator.ids)):
            raise ValueError("AI.EXTRACT needs the pipeline's document tokens and head")
        return operator

    def layout(self, region):
        """Extractions alone show their documents in their own prompts."""
        if not region.one_family:
            return None
        from quail.backends.quail.executor.region import RegionDocuments

        stages = self.extraction
        tree = (shared_prefix_tree(self.node.spec.name, stages.prefixes,
                                   region.query.loaded_model.arena)
                if self.node.spec.share_prefixes else None)
        return RegionDocuments(stages.prefixes, stages.keys, tree)

    def column_values(self, index, positions) -> tuple:
        values, spans = self.answers(index, positions)
        spec = self.nodes[index].spec
        return tuple(extraction_rows(
            spec.alias, spec.name, spec.span_name, self.ids[positions], values, spans
        ).values.columns)

    def __init__(self, state, nodes, ids, inputs, first=0):
        if any(not isinstance(node, AiExtract) for node in nodes):
            raise TypeError("an extraction pipeline holds AiExtract nodes only")
        alias = nodes[0].spec.alias
        texts = inputs.get("texts", {}).get(alias)
        if texts is None:
            raise ValueError(f"AI.EXTRACT needs the text of {alias!r}")
        super().__init__(state, nodes, ids, first)
        self.extraction = ExtractionInputs(
            state, [node.spec for node in nodes], state.loaded_model.vocabulary,
            self.ids,
            inputs["documents"][alias], texts,
            pipeline_head=inputs.get("pipeline_head", self.node.document_head),
            document_positions=inputs.get("document_positions"))

    def bind_decoder(self, decoder):
        super().bind_decoder(decoder)
        decoder.configure_copy(self.first, self.slots, self.extraction,
                               [node.spec for node in self.nodes])

    def answers(self, q, kept=None):
        slots = self.slots[q] if kept is None else self.slots[q, kept]
        return self.decoder.output_answers(self.first + q, slots)

    def read_answers(self, every):
        fed = super().read_answers(every)
        self.counts = []
        for q, node in enumerate(self.nodes):
            lengths = self.decoder.lengths[self.first + q][self.slots[q]]
            self.counts.append(dict(
                none=int((lengths <= 0).sum()),
                steps=int(self.decoder.steps[self.first + q][self.slots[q]].sum()),
                answer_tokens=int(np.maximum(0, lengths).sum()),
                answers_stopped_early=int(self.decoder.stopped_early[self.first + q]),
                suffix_tokens=(int(self.asked[q].sum())
                               * len(node.spec.prompt_token_parts[1]) + self.fed[q])))
        return fed

    def close(self, fresh, spans, stats, offset) -> dict:
        selected = stage_spans(spans, offset, offset + 1)
        total = sum(count["suffix_tokens"] for count in self.counts)

        def own(q):
            return (gpu_seconds(selected) * self.counts[q]["suffix_tokens"]
                    / max(1, total), len(selected))

        self._results = extraction_results(
            self.nodes, self, self.ids, fresh=fresh, counts=self.counts,
            own=own, stats=stats, kept=np.flatnonzero(self.asked[0]))
        return self._results
