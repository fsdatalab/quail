"""Run a chain's extractions and build their answer tables."""


import numpy as np

from quail.backends.quail.executor.extract.stages import ExtractionInputs
from quail.backends.quail.executor.model_operator import shared_prefix_tree
from quail.backends.quail.executor.operators.map import MapOperator
from quail.execution.rows import extraction_rows
from quail.physical import AiExtract


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

    def answer_rows(self, q, positions):
        """The Rows of field q's answers and spans over documents, by position."""
        spec = self.nodes[q].spec
        values, spans = self.answers(q, positions)
        return extraction_rows(spec.alias, spec.name, spec.span_name,
            self.ids[positions], values, spans)

    def answers(self, q, kept=None):
        slots = self.slots[q] if kept is None else self.slots[q, kept]
        return self.decoder.output_answers(self.first + q, slots)

    def answered_positions(self, q):
        """Return the documents the first field was asked: every field's rows."""
        return np.flatnonzero(self.asked[0])

    def answer_counts(self, q) -> dict:
        """Return field q's unanswered documents, rounds, and draft counts."""
        index = self.first + q
        lengths = self.decoder.lengths[index][self.slots[q]]
        steps = self.decoder.steps[index][self.slots[q]]
        rejected = self.decoder.rejected_steps[index][self.slots[q]]
        return dict(
            none=int((lengths <= 0).sum()),
            steps=int(steps.sum()),
            max_steps=int(steps.max(initial=0)),
            max_steps_unrejected=int((steps - rejected).max(initial=0)),
            **self.decoder.draft_counts[index])
