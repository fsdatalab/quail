"""The stages that ask a pipeline's questions of its documents.

The extractions of one pipeline run as one stage list over the
documents, with the shared decode-round shape of
quail.backends.quail.executor.decoding: each question is the tail
stage, whose row chooses the answer's first token, then rounds that
feed the chosen token and a draft of the document's next tokens and
verify them.
"""

import numpy as np

from quail.backends.quail.executor.decoding import AnswerStages
from quail.backends.quail.executor.extract.candidates import DocumentTokens
from quail.backends.quail.executor.extract.decode import (
    AnswerState,
    Decoder,
    Document,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.catalog.tokens import concat_tokens


class ExtractionStages(AnswerStages):
    """The stages of one pipeline's extractions and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The ExtractSpecs, in order; they share a table and a
            prompt head.
        vocabulary: The model's vocabulary index.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readout: Optional readout in place of the model's. Its one
            extra is the stop tokens as a group.
        pipeline_head: The token ids the pipeline shows the documents
            under; an extraction under the same head reads the KV the
            operators before it wrote.
        document_positions: Positions in the supplied documents and texts.
            None uses document_ids.

    Raises:
        ValueError: The specs differ in their table, head, or decoding settings.
    """

    operator = "extraction"

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts, readout=None, pipeline_head=None,
                 document_positions=None):
        self._parse_specs(state, specs)
        self.pipeline_head = (None if pipeline_head is None
                              else [int(t) for t in pipeline_head])
        self.vocabulary = vocabulary
        positions = (document_ids if document_positions is None
                     else document_positions)
        shown = [text.as_py() if hasattr(text, "as_py") else str(text)
                 for text in (texts[position] for position in positions)]
        stored = [None if documents is None else documents[document_id]
                  for document_id in positions]
        encoded = vocabulary.document_batch(shown, stored)
        self.documents = [self._prepare(index, document_id, text, tokens, prior)
                     for index, (document_id, text, tokens, prior) in enumerate(
                         zip(document_ids, shown, encoded, stored))]
        self._order_documents(document_ids)
        self.readout = self._readout(state) if readout is None else readout
        self.decoder = Decoder(vocabulary, self.readout.k, self.readout.width,
                               len(self.specs), self.spec)
        self._build()

    def _prepare(self, index, document_id, text, ids, stored) -> Document:
        """Build a document's prefix: the head and its text.

        The key names the document under the extraction's head, so a
        later extraction of the same table reads the KV this one
        leaves resident. A document shown under the query's shared
        head, whose tokens are the table's, keeps the table's key, so
        the KV a filter wrote is read.
        """
        alias = self.specs[0].alias
        key = (("extract", alias), int(document_id))
        shared = self.head == self.pipeline_head
        tokens = ids.numpy()
        same_tokens = stored is ids
        if shared and stored is not None and not same_tokens:
            prior = stored.numpy() if hasattr(stored, "numpy") else np.asarray(stored)
            same_tokens = np.array_equal(prior, tokens)
        if shared and stored is not None and same_tokens:
            key = (alias, int(document_id))
        prefix = concat_tokens(self.head, ids)
        scan = DocumentTokens.from_tokens(
            self.vocabulary, tokens, ids.character_offsets, len(self.head))
        # the round's KV read covers the prompt through the first tail
        rho = self.spec.round_cost_per_token * (len(prefix) + len(self.tails[0]))
        return Document(index=index, text=text, prefix=prefix, scan=scan, key=key,
                        answers=[AnswerState() for _ in self.specs], rho=rho)

    def _draft(self, document, answer, q) -> list:
        """Return the chosen token, then the document's next tokens as a draft."""
        return self.decoder.draft(answer, document.rho, q)

    def answers(self, q: int, kept=None) -> tuple:
        """Return each document's answer and span for question q.

        Args:
            q: The question's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        documents = (
            self.documents if kept is None else [self.documents[i] for i in kept]
        )
        return ([document.answers[q].answer for document in documents],
                [document.answers[q].span for document in documents])

    def counts(self, q: int) -> dict:
        """Return how many documents answered none and how much was fed."""
        states = [document.answers[q] for document in self.documents]
        return {
            "none": sum(answer.answer is None for answer in states),
            "steps": sum(answer.steps for answer in states),
            "answer_tokens": sum(len(answer.written) for answer in states),
            "suffix_tokens": sum(answer.fed_tokens for answer in states),
        }
