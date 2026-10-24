"""The stages that put a pipeline's instructions to its documents.

The maps of one pipeline run as one stage list over the documents,
with the shared decode-round shape of
quail.backends.quail.executor.decoding: each instruction is the tail
stage, whose row chooses the answer's first token, then rounds that
feed the chosen token and a draft and verify them. A chunk's rounds
are decided together, in arrays over its documents, before the
admission asks about each document.
"""

import numpy as np

from quail.backends.quail.executor.chunk import Suffixes
from quail.backends.quail.executor.decoding import AnswerStages
from quail.backends.quail.executor.map.decoder import MapDecoder, MapDocument
from quail.backends.quail.executor.stages import Stage, StageRequests
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
            self._prepare_document(
                index, document_id,
                None if documents is None else documents[position],
                None if texts is None else texts[position])
            for index, (document_id, position) in enumerate(
                zip(document_ids, positions))]
        self._order_documents(document_ids)
        self.readout = self._readout(state) if readout is None else readout
        self.decoder = MapDecoder(vocabulary, self.readout.k, self.readout.width,
                                  len(self.specs), self.spec, self.documents)
        # documents not asked yet, and documents between their first ask
        # and their last settled answer, for the draft policy's slack
        self.num_queued_documents = len(self.documents)
        self.num_decoding_documents = 0
        self._build()

    def _prepare_document(self, index, document_id, stored, text) -> MapDocument:
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
        # the round's KV read covers the prompt through the first tail
        rho = self.spec.round_cost_per_token * (len(prefix) + len(self.tails[0]))
        return MapDocument(index=index, prefix=prefix, tokens=tokens, key=key,
                           rho=rho)

    # ---- a chunk's rounds, decided together ---------------------------------

    def _stage_prepare(self, q: int, tail: bool):
        def prepare(anchors, rows):
            self._prepare_rounds(q, tail, anchors, rows)
        return prepare

    def _prepare_rounds(self, q: int, tail: bool, anchors, rows) -> None:
        """Verify a chunk's rounds for instruction q and draft the next ones.

        Args:
            q: The instruction's index.
            tail: Whether the rows follow the cue, choosing each
                answer's first token, rather than a round's fed tokens.
            anchors: The documents' indices.
            rows: Each document's readout rows, as the admission hands
                them over.
        """
        state = self.decoder.states[q]
        index = np.asarray(anchors, dtype=np.int64)
        width = self.readout.width
        arrays = [self._rows(row) for row in rows]
        stacked = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        stacked = np.ascontiguousarray(stacked, dtype=np.float32).reshape(-1, width)
        num_rows = np.fromiter((len(array) for array in arrays), dtype=np.int64,
                               count=len(arrays))
        if tail:
            state.asked[index] = True
            # a tail equal to the question's before it is in KV already:
            # only its cue token is fed
            same = (not self.recurrent and q > 0
                    and self.tails[q] == self.tails[q - 1])
            state.fed_tokens[index] += 1 if same else len(self.tails[q])
            if q == 0:
                self.num_queued_documents -= len(index)
                self.num_decoding_documents += len(index)
        else:
            state.fed_tokens[index] += num_rows
        ended = self.decoder.round(q, index, stacked, num_rows, self._slack())
        if q == len(self.specs) - 1:
            self.num_decoding_documents -= ended

    def _slack(self) -> float:
        """Fed tokens one document's round can add below the ridge, for free.

        While documents wait for their first ask, their prefills fill
        the passes, so no token is free. Afterwards the live documents'
        rounds are the passes' work: what is left under the ridge is
        split across them. This reads the pipeline's own queue, not the
        whole batch, so it understates the pass when other operators
        share it.
        """
        ridge = self.spec.ridge_tokens
        decoding = self.num_decoding_documents
        if not ridge or self.num_queued_documents > 0 or decoding <= 0:
            return 0.0
        return max(0.0, ridge - decoding) / decoding

    def _tail_decide(self, q, anchor, row):
        # decided by the chunk's prepare
        return True

    def _step_decide(self, q, anchor, row):
        return True

    def _step_suffix(self, q, key):
        """Feed the chosen token, then the draft the chunk's prepare chose.

        A recurrent model replays the accepted tokens to advance its
        checkpoint, so they are fed again and written as the suffix.
        """
        document = self._document(key)
        state = self.decoder.states[q]
        index = document.index
        if state.done[index]:
            return Stage.SkipTo(self.question_starts[q + 1])
        fed = state.fed[index, :state.fed_len[index]]
        discard = int(state.discard[index])
        if self.recurrent:
            previous = self.previous_fed[q].get(index, [])
            accepted = previous[:len(previous) - discard]
            tokens = fed.tolist()
            self.previous_fed[q][index] = tokens
            state.fed_tokens[index] += len(accepted)
            return StageRequests([accepted + tokens],
                                 write_suffix_tokens=len(accepted),
                                 read_rows=[len(tokens)])
        return StageRequests(Suffixes(fed, [len(fed)]), discard=discard)

    def _build(self) -> None:
        # a recurrent model's replay needs each document's previous
        # round, which the arrays overwrite when the next is drafted
        self.previous_fed = [{} for _ in self.specs]
        super()._build()

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for q, state in enumerate(self.decoder.states):
            for index in np.flatnonzero(~state.done):
                self.decoder.settle(self.documents[index], q)

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return np.flatnonzero(self.decoder.states[0].asked).tolist()

    def answers(self, q: int, kept=None) -> list:
        """Return each document's answer for instruction q.

        Args:
            q: The instruction's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        answers = self.decoder.states[q].answers
        return list(answers) if kept is None else [answers[i] for i in kept]

    def counts(self, q: int) -> dict:
        """Return how many documents answered empty and how much was fed."""
        state = self.decoder.states[q]
        return {
            "empty": sum(not answer for answer in state.answers),
            "steps": int(state.steps.sum()),
            "answer_tokens": int(state.num_written.sum()),
            "suffix_tokens": int(state.fed_tokens.sum()),
            **self._draft_counts(q),
        }

    def _draft_counts(self, q: int) -> dict:
        """Return the instruction's chained-draft trials and accepts."""
        if self.decoder.policies is None:
            return {"draft_trials": 0, "draft_accepts": 0}
        policy = self.decoder.policies[q]
        return {"draft_trials": int(policy.trials.sum()),
                "draft_accepts": int(policy.accepts.sum())}
