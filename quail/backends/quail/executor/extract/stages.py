"""Documents and prompts for copy-constrained answer decoding."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quail.backends.quail.executor.extract.candidates import DocumentTokens
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.catalog.tokens import concat_tokens


@dataclass
class Document:
    """One document's prompt and copy candidates."""

    index: int
    text: str
    prefix: Sequence
    scan: DocumentTokens
    key: tuple
    rho: float = 0.0    # the document's KV read per round, in fed tokens


class ExtractionInputs:
    """The documents and prompts of one pipeline's extractions.

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

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts, readout=None, pipeline_head=None,
                 document_positions=None):
        self.specs = list(specs)
        first = self.specs[0]
        self.spec = first
        if any(not first.same_decoding(spec) for spec in self.specs):
            raise ValueError("a pipeline's extractions share decoding settings")
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError("a pipeline's extractions share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError("a pipeline's extractions share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        self.pipeline_head = (None if pipeline_head is None
                              else [int(t) for t in pipeline_head])
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError("an extraction's tail ends in its cue after the question")
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
        document_ids = np.asarray(document_ids, dtype=np.int64)
        self.document_order = np.argsort(document_ids, kind="stable")
        self.document_ids = document_ids[self.document_order]
        self.prefixes = [document.prefix for document in self.documents]
        self.keys = [document.key for document in self.documents]

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
                        rho=rho)


