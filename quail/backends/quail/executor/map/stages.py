"""The stages that put a pipeline's instructions to its documents.

The maps of one pipeline run as one stage list, with the shared
decode-round shape of quail.backends.quail.executor.decoding: each
instruction is a tail stage, whose row chooses the answer's first
token, then rounds that feed the chosen token and a draft and verify
them. A chunk's rounds are decided together, in arrays, before the
admission asks about each document. On the tree and unified paths the
stages run ahead of their answers: a document's next round is
launched before the one before it is decided, and the GPU puts the
token that round chose first.

A pipeline of several maps run on its own puts its instructions to a
document at once. Each (document, instruction) pair is a branch: its
own anchor, whose KV holds the instruction's tail and answer, and
which borrows the document's pages from the document's first branch,
so a document is computed once. Otherwise, as when the maps follow
another operator over the same documents, the instructions run one
after another over each document, each rewinding past the one before.

The stages also size each document's KV as it writes (see
quail.backends.quail.executor.growth): they report the rows each
branch's next round needs and the rows it is expected to still write.
"""

from functools import partial

import numpy as np

from quail.backends.quail.executor.chunk import Suffixes
from quail.backends.quail.executor.decoding import AnswerStages
from quail.backends.quail.executor.map.decoder import MapDecoder, MapDocument
from quail.backends.quail.executor.stages import RoundRequests, Stage, StageRequests
from quail.backends.quail.executor.vocabulary import Vocabulary
from quail.catalog.tokens import PrefixTree, concat_tokens

# the first token of a device-fed round, which the GPU replaces
_CHOSEN_ON_GPU = np.zeros(1, dtype=np.int64)
_CHOSEN_ON_GPU.setflags(write=False)
# the fraction of answers the KV projection expects to end within
# its count: a higher value admits fewer documents and stalls fewer
OCCUPANCY_QUANTILE = 0.9


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
        branches: Whether several instructions are put to a document
            at once, as branches that borrow its pages; a recurrent
            model, whose state cannot branch, puts them one after
            another.

    Raises:
        ValueError: The specs differ in their table, head, or decoding settings.
    """

    operator = "map"

    def __init__(self, state, specs, vocabulary: Vocabulary, document_ids,
                 documents, texts=None, readout=None, pipeline_head=None,
                 document_positions=None, branches=False):
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
        self.branches = bool(branches) and len(self.specs) > 1 and not self.recurrent
        self._anchors(state.loaded_model.arena.page_tokens)
        # per instruction, each document's cue position: the first
        # round's token follows it
        self.prefix_lengths = np.array(
            [len(document.prefix) for document in self.documents], dtype=np.int64)
        self.cue_positions = [self.prefix_lengths + len(tail) - 1
                              for tail in self.tails]
        self.tail_lengths = np.array([len(tail) for tail in self.tails],
                                     dtype=np.int64)
        # a recurrent model replays each round's accepted tokens, so
        # its rounds wait for their answers
        self.ahead = not self.recurrent
        # answers not asked yet, and answers between their first ask
        # and their settling, for the draft policy's slack
        per_document = len(self.specs) if self.branches else 1
        self.num_queued_documents = len(self.documents) * per_document
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

    def _anchors(self, page_tokens: int) -> None:
        """Lay out the anchors: one per document, or one per branch.

        Branch a is document a // K's instruction a % K; a document's
        first branch keeps the document's key, and the others borrow
        its whole pages before its last token.
        """
        num_documents = len(self.documents)
        per_document = len(self.specs) if self.branches else 1
        anchors = np.arange(num_documents * per_document, dtype=np.int64)
        self.anchor_document = anchors // per_document
        self.anchor_question = anchors % per_document
        self.prefix_tree = None
        if not self.branches:
            return
        self.prefixes = [self.documents[d].prefix for d in self.anchor_document]
        self.keys = [document.key if q == 0 else ("branch", document.key, int(q))
                     for document, q in zip(
                         (self.documents[d] for d in self.anchor_document),
                         self.anchor_question)]
        self._anchor_of_key = {key: a for a, key in enumerate(self.keys)}
        page = page_tokens
        parent, shared = [], []
        for a, (d, q) in enumerate(zip(self.anchor_document, self.anchor_question)):
            # a branch borrows the whole pages of the document before
            # its last token from the document's first branch
            length = len(self.documents[d].prefix)
            borrowed = (length - 1) // page * page
            if q == 0 or borrowed == 0:
                parent.append(None)
                shared.append(0)
            else:
                parent.append(a - int(q))
                shared.append(borrowed)
        self.prefix_tree = PrefixTree(list(anchors), parent, shared)

    # ---- anchors to documents and questions ---------------------------------

    def _split(self, anchors, q):
        """Yield (question, positions in anchors, documents) for each question.

        q is the question a serial stage is bound to, or None for a
        branch stage, whose anchors carry their own.
        """
        anchors = np.asarray(anchors, dtype=np.int64)
        if q is not None:
            yield q, slice(None), anchors
            return
        questions = self.anchor_question[anchors]
        documents = self.anchor_document[anchors]
        for question in np.unique(questions).tolist():
            positions = np.flatnonzero(questions == question)
            yield question, positions, documents[positions]

    def _locate(self, key, q):
        """Return the question and document index of a stage's key."""
        if q is not None:
            return q, self._document(key).index
        a = self._anchor_of_key[key]
        return int(self.anchor_question[a]), int(self.anchor_document[a])

    # ---- the stages ---------------------------------------------------------

    def _build(self) -> None:
        # a recurrent model's replay needs each document's previous
        # round, which the arrays overwrite when the next is drafted
        self.previous_fed = [{} for _ in self.specs]
        if not self.branches:
            super()._build()
            return
        names = ", ".join(spec.name for spec in self.specs)
        self.question_starts = [0, 2]
        self.stages = [
            Stage(suffixes=self._tail_suffix,
                  suffix_tokens=int(self.tail_lengths.max()),
                  readout=self.readout, decide=partial(self._tail_decide, None),
                  prepare=self._stage_prepare(None, tail=True),
                  launched=self._stage_launched(None, tail=True), ahead=self.ahead,
                  single=True, append=True, label=f"{names} tails"),
            Stage(suffixes=partial(self._step_suffix, None),
                  suffix_tokens=1 + self.spec.draft_tokens,
                  readout=self.readout, decide=partial(self._step_decide, None),
                  prepare=self._stage_prepare(None, tail=False),
                  launched=self._stage_launched(None, tail=False), ahead=self.ahead,
                  costs=self._stage_costs(None), rounds=self._stage_rounds(None),
                  read_all_rows=True, single=True, append=True, repeats=True,
                  label=f"{names} rounds"),
        ]

    def _tail_suffix(self, key):
        """A branch's first request: its instruction's whole tail."""
        q, _ = self._locate(key, None)
        tail = np.asarray(self.tails[q], dtype=np.int64)
        return StageRequests(Suffixes(tail, [len(tail)]))

    @property
    def appended_tokens(self) -> int:
        """The most tokens an instruction leaves in KV after its frame.

        The cue (a branch's whole tail, which has no frame), the
        answer, a round's draft past it, and the round launched before
        the answer's end was decided.
        """
        tail = int(self.tail_lengths.max()) if self.branches else 1
        return tail + self.spec.max_answer_tokens + 2 * self.spec.draft_tokens

    # ---- the growth model ---------------------------------------------------

    @property
    def grows(self) -> bool:
        """Whether each anchor answers one question, so its KV can grow as it writes.

        A recurrent model's checkpoints are sized up front, and
        questions run one after another share a document's KV.
        """
        return not self.recurrent and (self.branches or len(self.specs) == 1)

    @property
    def initial_tokens(self) -> int:
        """Rows past the prefix a document's pages cover when admitted.

        Its tail, and the two rounds that can be in flight before its
        pages first grow, each at most a chosen token and a draft.
        """
        return int(self.tail_lengths.max()) + 2 * (1 + self.spec.draft_tokens)

    def remaining(self, anchors) -> np.ndarray:
        """Return the rows each anchor is expected to still write.

        Its answer's remaining tokens at OCCUPANCY_QUANTILE, its tail
        when not yet asked, and a round in flight.
        """
        anchors = np.asarray(anchors, dtype=np.int64)
        out = np.zeros(len(anchors), dtype=np.int64)
        for q, positions, documents in self._split(anchors, None):
            state = self.decoder.states[q]
            left = self.decoder.remaining(q, documents, OCCUPANCY_QUANTILE)
            left = left + np.where(state.started[documents], 0, self.tail_lengths[q])
            left = left + 1 + self.spec.draft_tokens
            out[positions] = np.where(state.done[documents], 0, left)
        return out

    def capacities(self, anchors) -> np.ndarray:
        """Return the rows each anchor's pages must cover for its next request.

        A device-fed round lands past the tokens the round before it
        accepts, at most a chosen token and a draft.
        """
        anchors = np.asarray(anchors, dtype=np.int64)
        out = np.zeros(len(anchors), dtype=np.int64)
        for q, positions, documents in self._split(anchors, None):
            state = self.decoder.states[q]
            device_fed, lengths = self._round_lengths(q, documents)
            rounds = (state.base[documents] + lengths
                      + np.where(device_fed, 1 + self.spec.draft_tokens, 0))
            tails = self.prefix_lengths[documents] + self.tail_lengths[q]
            need = np.where(state.started[documents], rounds, tails)
            out[positions] = np.where(state.done[documents], 0, need)
        return out

    # ---- a chunk's rounds, decided together ---------------------------------

    def _stage_prepare(self, q, tail: bool):
        def prepare(anchors, rows):
            for question, positions, documents in self._split(anchors, q):
                chosen = (rows if isinstance(positions, slice)
                          else [rows[i] for i in positions.tolist()])
                self._prepare_rounds(question, tail, documents, chosen)
        return prepare

    def _stage_launched(self, q, tail: bool):
        def launched(anchors):
            for question, _, documents in self._split(anchors, q):
                self.decoder.launch(question, documents, tail,
                                    self.cue_positions[question][documents])
        return launched

    def _round_lengths(self, q: int, index: np.ndarray) -> tuple:
        """Return (device_fed, lengths) of the documents' next rounds."""
        state = self.decoder.states[q]
        device_fed = state.unreported[index] > 0
        return device_fed, np.where(device_fed, 1 + state.draft_len[index],
                                    state.fed_len[index])

    def _stage_costs(self, q):
        if self.recurrent:
            return None

        def costs(anchors):
            out = np.zeros(len(anchors), dtype=np.int64)
            for question, positions, documents in self._split(anchors, q):
                state = self.decoder.states[question]
                _, lengths = self._round_lengths(question, documents)
                out[positions] = np.where(state.done[documents], -1, lengths)
            return out
        return costs

    def _stage_rounds(self, q):
        """The rounds of many anchors at once, as _step_suffix builds one."""
        if self.recurrent:
            return None
        width = 1 + self.spec.draft_tokens

        def rounds(anchors):
            count = len(anchors)
            tokens = np.zeros((count, width), dtype=np.int64)
            lengths = np.zeros(count, dtype=np.int64)
            positions = np.zeros(count, dtype=np.int64)
            device_fed = np.zeros(count, dtype=bool)
            for question, at, documents in self._split(anchors, q):
                state = self.decoder.states[question]
                fed, length = self._round_lengths(question, documents)
                # a device-fed round: the placeholder, then the pending
                # draft; otherwise the round's known tokens
                ahead = np.concatenate(
                    (np.zeros((len(documents), 1), dtype=np.int64),
                     state.draft[documents]), axis=1)
                tokens[at] = np.where(fed[:, None], ahead, state.fed[documents])
                lengths[at] = length
                positions[at] = state.base[documents]
                device_fed[at] = fed
            hop = np.arange(width)
            return RoundRequests(ids=tokens[hop[None, :] < lengths[:, None]],
                                 lengths=lengths, positions=positions,
                                 device_fed=device_fed)
        return rounds

    def _prepare_rounds(self, q: int, tail: bool, documents, rows) -> None:
        """Verify a chunk's rounds for instruction q and draft the next ones.

        Args:
            q: The instruction's index.
            tail: Whether the rows follow the cue, choosing each
                answer's first token, rather than a round's fed tokens.
            documents: The documents' indices.
            rows: Each document's readout rows, as the admission hands
                them over.
        """
        state = self.decoder.states[q]
        index = np.asarray(documents, dtype=np.int64)
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
            same = (not self.recurrent and not self.branches and q > 0
                    and self.tails[q] == self.tails[q - 1])
            state.fed_tokens[index] += 1 if same else len(self.tails[q])
            if q == 0 or self.branches:
                self.num_queued_documents -= len(index)
                self.num_decoding_documents += len(index)
        else:
            state.fed_tokens[index] += num_rows
        keep = ~state.done[index]
        if not keep.all():
            # a round launched for an answer that ended with the round
            # before it: its rows decide nothing
            index = index[keep]
            if not len(index):
                return
            stacked = stacked[np.repeat(keep, num_rows)]
            num_rows = num_rows[keep]
        ended = self.decoder.round(q, index, stacked, num_rows, self._slack())
        if q == len(self.specs) - 1 or self.branches:
            self.num_decoding_documents -= ended

    def _slack(self) -> float:
        """Fed tokens one answer's round can add below the ridge, for free.

        While answers wait for their first ask, their prefills fill
        the passes, so no token is free. Afterwards the decoding
        answers' rounds are the passes' work: what is left under the
        ridge is split across them. This reads the pipeline's own
        queue, not the whole batch, so it understates the pass when
        other operators share it.
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

        While the round before is on the stream, the GPU puts the
        token it chooses first and the request carries the draft of
        the round after it. A recurrent model replays the accepted
        tokens to advance its checkpoint, so they are fed again and
        written as the suffix.
        """
        serial = q is not None
        q, index = self._locate(key, q)
        state = self.decoder.states[q]
        if state.done[index]:
            return Stage.SkipTo(self.question_starts[q + 1] if serial
                                else len(self.stages))
        position = int(state.base[index])
        if state.unreported[index]:
            draft = state.draft[index, :state.draft_len[index]]
            return StageRequests(
                Suffixes(np.concatenate((_CHOSEN_ON_GPU, draft)), [1 + len(draft)]),
                position=position, device_fed=True)
        fed = state.fed[index, :state.fed_len[index]].copy()
        if self.recurrent:
            previous = self.previous_fed[q].get(index, [])
            accepted = previous[:len(previous) - int(state.discard[index])]
            tokens = fed.tolist()
            self.previous_fed[q][index] = tokens
            state.fed_tokens[index] += len(accepted)
            return StageRequests([accepted + tokens],
                                 write_suffix_tokens=len(accepted),
                                 read_rows=[len(tokens)],
                                 position=position - len(accepted))
        return StageRequests(Suffixes(fed, [len(fed)]), position=position)

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
