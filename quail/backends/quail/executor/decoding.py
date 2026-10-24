"""The shared stage shape of operators that decode an answer in rounds.

An extraction and a map put each of their questions to a document the
same way: the question's tail is the frame of its stages, written
right after the document in place of the tokens the question before it
appended, so a document is read once however many questions it takes.
Each question is the tail stage, whose row chooses the answer's first
token, then up to the configured answer limit in rounds, each feeding
the chosen token and a draft and verifying the draft against every
row. The operators differ in how a token is chosen and drafted, which
their decoders define.
"""

from functools import partial

import numpy as np

from quail.backends.quail.executor.model_loading import full_output_head
from quail.backends.quail.executor.readout import AsyncTopLogprobs, readout_rows
from quail.backends.quail.executor.stages import Stage, StageRequests


class AnswerStages:
    """The stages of one pipeline's decoded answers.

    A subclass prepares its documents and decoder, then calls
    _build(). It supplies _draft(document, answer, q), the tokens a
    round feeds, and a decoder with verify(document, q, rows, fed) and
    settle(document, q).

    Attributes set by the subclass before _build(): specs, documents,
    document_ids (sorted), document_order, decoder, readout, tails,
    recurrent.
    """

    operator = ""   # the operator's name in error messages

    def _parse_specs(self, state, specs) -> None:
        """Check the specs share a table, head, and decoding settings."""
        self.specs = list(specs)
        first = self.specs[0]
        self.spec = first
        self.recurrent = bool(getattr(state.loaded_model.arena, "num_state_slots", 0))
        if any(not first.same_decoding(spec) for spec in self.specs):
            raise ValueError(
                f"a pipeline's {self.operator}s share decoding settings")
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError(f"a pipeline's {self.operator}s share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError(f"a pipeline's {self.operator}s share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError(
                f"a {self.operator}'s tail ends in its cue after its question")

    def _order_documents(self, document_ids) -> None:
        """Index the documents by id for the suffix callbacks."""
        document_ids = np.asarray(document_ids, dtype=np.int64)
        self.document_order = np.argsort(document_ids, kind="stable")
        self.document_ids = document_ids[self.document_order]
        self.prefixes = [document.prefix for document in self.documents]
        self.keys = [document.key for document in self.documents]

    def _build(self) -> None:
        """Build each question's stages, in spec order."""
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
                  suffix_tokens=(1 + self.spec.draft_tokens)
                  * (2 if self.recurrent else 1), frame=frame,
                  readout=self.readout, decide=partial(self._step_decide, q),
                  read_all_rows=True,
                  single=True, append=True, label=f"{name} round {t}")
            for t in range(1, self.spec.max_answer_tokens + 1))
        return stages

    @property
    def appended_tokens(self) -> int:
        """The most tokens a question leaves in KV after its frame.

        The cue, and the answer's tokens with the last round's draft
        past them.
        """
        return 1 + self.spec.max_answer_tokens + self.spec.draft_tokens

    def _readout(self, state):
        """The readout of every row: the stop, then the top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        return AsyncTopLogprobs(torch, torch.nn.functional, head,
                                self.spec.top_tokens, [self.vocabulary.stop_ids],
                                ragged=True)

    # ---- the tail, then the answer, a round of tokens at a time ------------

    def _tail_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.asked = True
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = (not self.recurrent and q > 0
                and self.tails[q] == self.tails[q - 1])
        answer.fed_tokens += 1 if same else len(self.tails[q])
        self.decoder.verify(document, q, self._rows(row), [])
        return True

    def _rows(self, row):
        """Return a readout's rows for one answer."""
        # the admission hands over the document's one answer in a list
        # or an object array
        if isinstance(row, list) and len(row) == 1:
            row = row[0]
        elif isinstance(row, np.ndarray) and row.dtype == object and row.size == 1:
            row = row.reshape(-1)[0]
        return readout_rows(row, self.readout.width)

    def _step_suffix(self, q, key):
        """Feed the chosen token, then a draft of the answer's next tokens.

        A recurrent model replays the accepted tokens to advance its
        checkpoint, so they are fed again and written as the suffix.
        """
        document = self._document(key)
        answer = document.answers[q]
        if answer.done:
            return Stage.SKIP
        accepted = answer.fed[:len(answer.fed) - answer.discard]
        tokens = self._draft(document, answer, q)
        if self.recurrent:
            answer.fed_tokens += len(accepted)
            return StageRequests([accepted + tokens],
                                 write_suffix_tokens=len(accepted),
                                 read_rows=[len(tokens)])
        return StageRequests([tokens], discard=answer.discard)

    def _draft(self, document, answer, q) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft."""
        raise NotImplementedError

    def _document(self, key):
        position = np.searchsorted(self.document_ids, key[1], side="right") - 1
        return self.documents[self.document_order[position]]

    def _step_decide(self, q, anchor, row):
        """Verify the round's draft by the row after each of its tokens."""
        document = self.documents[anchor]
        answer = document.answers[q]
        answer.fed_tokens += len(answer.fed)
        self.decoder.verify(document, q, self._rows(row), answer.fed)
        return True

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for document in self.documents:
            for q, answer in enumerate(document.answers):
                if not answer.done:
                    self.decoder.settle(document, q)

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return [index for index, document in enumerate(self.documents)
                if document.answers[0].asked]
