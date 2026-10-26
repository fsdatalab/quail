"""The stages that put a pipeline's instructions to its documents.

The maps of one pipeline run as one stage list over the documents,
with the shared decode-round shape of
quail.backends.quail.executor.decoding: each instruction is the tail
stage, whose row chooses the answer's first token, then rounds that
feed the chosen token and a draft copied from the document and verify
them.
"""

import numpy as np

from quail.backends.quail.executor.decoding import AnswerStages
from quail.backends.quail.executor.map.decoder import (
    MapAnswer,
    MapDecoder,
    MapDocument,
)
from quail.backends.quail.executor.vocabulary import Vocabulary
from quail.catalog.tokens import concat_tokens


class MapStages(AnswerStages):
    """The stages of one pipeline's maps and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The MapSpecs, in order; they share a table and a prompt
            head.
        vocabulary: The model's vocabulary index.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts, read only for a document
            without stored tokens.
        readout: Optional readout in place of the model's. Its one
            extra is the stop tokens as a group.
        pipeline_head: The token ids the pipeline shows the documents
            under; a map under the same head reads the KV the
            operators before it wrote.
        document_positions: Positions in the supplied documents and texts.
            None uses document_ids.

    Raises:
        ValueError: The specs differ in their table, head, or decoding settings.
    """

    operator = "map"

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts=None, readout=None, pipeline_head=None,
                 document_positions=None):
        self._parse_specs(state, specs)
        self.pipeline_head = (None if pipeline_head is None
                              else [int(t) for t in pipeline_head])
        self.vocabulary = vocabulary
        positions = (document_ids if document_positions is None
                     else document_positions)
        self.documents = [
            self._prepare(document_id,
                          None if documents is None else documents[position],
                          None if texts is None else texts[position])
            for document_id, position in zip(document_ids, positions)]
        self._order_documents(document_ids)
        self.readout = self._readout(state) if readout is None else readout
        self.decoder = MapDecoder(vocabulary, self.readout.k, self.spec)
        self._build()

    def _prepare(self, document_id, stored, text) -> MapDocument:
        """Build a document's prefix: the head and its tokens.

        The key names the document under the map's head, so a later
        map of the same table reads the KV this one leaves resident. A
        document shown under the query's shared head, whose tokens are
        the table's, keeps the table's key, so the KV an earlier
        operator wrote is read.
        """
        alias = self.specs[0].alias
        if stored is None:
            if text is None:
                raise ValueError(
                    f"AI.MAP needs the tokens or the text of {alias!r}")
            shown = text.as_py() if hasattr(text, "as_py") else str(text)
            stored = self.vocabulary.encode_ids(shown)
        tokens = (stored.numpy() if hasattr(stored, "numpy")
                  else np.asarray(stored, dtype=np.int64))
        key = (("map", alias), int(document_id))
        if self.head == self.pipeline_head:
            key = (alias, int(document_id))
        prefix = concat_tokens(self.head, stored)
        return MapDocument(prefix=prefix, tokens=tokens, key=key,
                           answers=[MapAnswer() for _ in self.specs])

    def _draft(self, document, answer, q) -> list:
        """Return the chosen token, then the document's next tokens as a draft."""
        return self.decoder.draft(answer, document)

    def finish(self) -> None:
        """Keep only documents whose answers finished before the run stopped."""
        self.completed = [index for index, document in enumerate(self.documents)
                          if all(answer.done for answer in document.answers)]

    def asked(self) -> list:
        """Return the indices of documents whose answers finished."""
        return self.completed

    def answers(self, q: int, kept=None) -> list:
        """Return each document's answer for instruction q.

        Args:
            q: The instruction's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        documents = (
            self.documents if kept is None else [self.documents[i] for i in kept]
        )
        return [document.answers[q].answer for document in documents]

    def counts(self, q: int) -> dict:
        """Return how many documents answered empty and how much was fed."""
        states = [document.answers[q] for document in self.documents]
        return {
            "empty": sum(not answer.answer for answer in states),
            "steps": sum(answer.steps for answer in states),
            "answer_tokens": sum(len(answer.written) for answer in states),
            "suffix_tokens": sum(answer.fed_tokens for answer in states),
        }
