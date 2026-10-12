"""The stages that ask a pipeline's questions of its documents.

The extractions of one pipeline run as one stage list over the
documents: a question's tail is the frame of its stages, written right
after the document in place of the tokens the question before it
appended, so a document is read once however many questions it takes.
Each question is the tail stage, whose row chooses the answer's first
token, then up to ANSWER_MAX_TOKENS rounds, each feeding the chosen
token and a draft of the document's next tokens and verifying them.
"""

from functools import partial

import numpy as np

from quail.backends.quail.executor.extract.candidates import DocumentTokens
from quail.backends.quail.executor.extract.decode import (
    CUE_TOP,
    AnswerState,
    Decoder,
    Document,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncTopLogprobs
from quail.backends.quail.executor.stages import Stage
from quail.cost.extract import ANSWER_MAX_TOKENS, DRAFT_TOKENS


class ExtractionStages:
    """The stages of one pipeline's extractions and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The ExtractSpecs, in order; they share a table and a
            prompt head.
        vocabulary: The model's vocabulary index.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readout: Optional readout in place of the model's. Its extras
            are the none token, then the stop tokens as one group.
        shared_head: The query's shared preamble, under which a filter
            writes the documents' KV; None when the query has none.

    Raises:
        ValueError: The specs differ in their table or head.
    """

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts, readout=None, shared_head=None):
        self.specs = list(specs)
        first = self.specs[0]
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError("a pipeline's extractions share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError("a pipeline's extractions share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        self.shared_head = (None if shared_head is None
                            else [int(t) for t in shared_head])
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError("an extraction's tail ends in its cue after the question")
        self.vocabulary = vocabulary
        shown = [text.as_py() if hasattr(text, "as_py") else str(text)
                 for text in (texts[document] for document in document_ids)]
        encoded = vocabulary.encode_batch(shown)
        self.docs = [self._prepare(index, document, text, *ids,
                                   None if documents is None
                                   else documents[document])
                     for index, (document, text, ids)
                     in enumerate(zip(document_ids, shown, encoded))]
        self.by_key = {doc.key: doc for doc in self.docs}
        self.prefixes = [doc.prefix for doc in self.docs]
        self.keys = [doc.key for doc in self.docs]
        self.readout = self._readout(state) if readout is None else readout
        self.decoder = Decoder(vocabulary, self.readout.k, self.readout.width,
                               [spec.trim for spec in self.specs])
        self.stages = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.stages.extend(self._question_stages(q, spec.name, tail))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, then the answer's rounds.

        The tail but its last token is the frame of every stage, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail, the cue, is the stage's suffix, whose
        row chooses the answer's first token.
        """
        frame, last = tail[:-1], tail[-1]
        stages = [Stage(
            suffixes=[[last]], frame=frame, rewind=True,
            readout=self.readout, decide=partial(self._tail_decide, q),
            single=True, append=True, label=f"{name} tail")]
        stages.extend(
            Stage(suffixes=partial(self._step_suffix, q),
                  suffix_tokens=1 + DRAFT_TOKENS, frame=frame,
                  readout=self.readout, decide=partial(self._step_decide, q),
                  discard=partial(self._step_discard, q), read_all_rows=True,
                  single=True, append=True, label=f"{name} round {t}")
            for t in range(1, ANSWER_MAX_TOKENS + 1))
        return stages

    @property
    def appended_tokens(self) -> int:
        """The most tokens a question leaves in KV after its frame.

        The cue, and the answer's tokens with the last round's draft
        past them.
        """
        return 1 + ANSWER_MAX_TOKENS + DRAFT_TOKENS

    def _readout(self, state):
        """The readout of every row: none, the stop, then the top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        vocabulary = self.vocabulary
        return AsyncTopLogprobs(torch, torch.nn.functional, head, CUE_TOP,
                                [vocabulary.none_id, vocabulary.stop_ids],
                                ragged=True)

    def _prepare(self, index, document, text, ids, offsets, stored) -> Document:
        """Build a document's prefix: the head and its text.

        The key names the document under the extraction's head, so a
        later extraction of the same table reads the KV this one
        leaves resident. A document shown under the query's shared
        head, whose tokens are the table's, keeps the table's key, so
        the KV a filter wrote is read.
        """
        alias = self.specs[0].alias
        key = (("extract", alias), int(document))
        shared = (self.head == self.shared_head if self.shared_head is not None
                  else not self.head)
        if shared and stored is not None and [int(t) for t in stored] == ids:
            key = (alias, int(document))
        prefix = self.head + ids
        bounds = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        tokens = np.asarray(ids, dtype=np.int64)
        scan = DocumentTokens(
            np.arange(len(ids), dtype=np.int64) + len(self.head),
            bounds[:, 0], bounds[:, 1], tokens, self.vocabulary.key_of[tokens])
        return Document(index=index, text=text, prefix=prefix, scan=scan, key=key,
                        answers=[AnswerState() for _ in self.specs])

    # ---- the tail, then the answer, a round of tokens at a time ------------

    def _tail_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        doc = self.docs[anchor]
        answer = doc.answers[q]
        answer.asked = True
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = q > 0 and self.tails[q] == self.tails[q - 1]
        answer.fed_tokens += 1 if same else len(self.tails[q])
        self.decoder.verify(doc, q, self.decoder.rows(row), [])
        return True

    def _step_suffix(self, q, key):
        """Feed the chosen token, then the document's next tokens as a draft."""
        answer = self.by_key[key].answers[q]
        if answer.done:
            return Stage.SKIP
        return [self.decoder.draft(answer)]

    def _step_discard(self, q, key):
        return self.by_key[key].answers[q].discard

    def _step_decide(self, q, anchor, row):
        """Verify the round's draft by the row after each of its tokens."""
        doc = self.docs[anchor]
        answer = doc.answers[q]
        answer.fed_tokens += len(answer.fed)
        self.decoder.verify(doc, q, self.decoder.rows(row), answer.fed)
        return True

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for doc in self.docs:
            for q, answer in enumerate(doc.answers):
                if not answer.done:
                    self.decoder.settle(doc, q)

    def answers(self, q: int, kept=None) -> tuple:
        """Return each document's answer and span for question q.

        Args:
            q: The question's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        docs = self.docs if kept is None else [self.docs[i] for i in kept]
        return ([doc.answers[q].answer for doc in docs],
                [doc.answers[q].span for doc in docs])

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return [index for index, doc in enumerate(self.docs)
                if doc.answers[0].asked]

    def counts(self, q: int) -> dict:
        """Return how many documents answered none and how much was fed."""
        states = [doc.answers[q] for doc in self.docs]
        return {
            "none": sum(answer.answer is None for answer in states),
            "steps": sum(answer.steps for answer in states),
            "answer_tokens": sum(len(answer.written) for answer in states),
            "suffix_tokens": sum(answer.fed_tokens for answer in states),
        }
