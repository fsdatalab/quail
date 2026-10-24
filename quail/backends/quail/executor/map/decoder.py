"""The decode of a map answer: the greedy choice at each readout row.

The answer is decoded after the answer cue, one round at a time. Each
round feeds the chosen token and a draft, and the rows after them
verify it: a draft token is written exactly when the model would have
chosen it one token at a time. A draft is copied from the document
where the last written tokens occur, or chained from the n-gram cache
of the answers written so far; the adaptive policy sizes it from the
acceptance odds and the forward pass's slack. The answer ends where
the model ends its turn, or at the configured answer limit.

The state of every document's answer lives in arrays over the
documents, and a chunk's rounds are verified, written, and drafted
together in a few array passes; only the n-gram chain, which walks a
dictionary, runs per document.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.vocabulary import Vocabulary
from quail.cost.draft_policy import MapDraftPolicy
from quail.cost.ngram_cache import MATCH_ORDERS, NgramCache
from quail.physical import MapSpec

# a draft token's policy cell: copied from the document, or none
COPY_CELL = -1
NO_CELL = -2
# the chain stops once the hops' combined odds fall below this
CHAIN_CUTOFF = 0.02


class AnswerState:
    """Every document's answer to one instruction, in arrays over the documents.

    Args:
        num_documents: How many documents the pipeline holds.
        max_answer_tokens: The answer limit, the width of written.
        draft_tokens: The most draft tokens a round feeds.
    """

    def __init__(self, num_documents: int, max_answer_tokens: int,
                 draft_tokens: int):
        shape = (num_documents,)
        self.written = np.zeros((num_documents, max_answer_tokens), dtype=np.int64)
        self.num_written = np.zeros(shape, dtype=np.int64)
        self.done = np.zeros(shape, dtype=bool)
        self.asked = np.zeros(shape, dtype=bool)
        self.steps = np.zeros(shape, dtype=np.int64)
        self.fed_tokens = np.zeros(shape, dtype=np.int64)
        # the last round's fed tokens, its unverified tail, and each
        # draft token's policy cell
        self.fed = np.zeros((num_documents, 1 + draft_tokens), dtype=np.int64)
        self.fed_len = np.zeros(shape, dtype=np.int64)
        self.discard = np.zeros(shape, dtype=np.int64)
        self.cells = np.full((num_documents, max(1, draft_tokens)), NO_CELL,
                             dtype=np.int64)
        self.cached = np.zeros(shape, dtype=np.int64)
        self.answers = [None] * num_documents


class MapAnswer:
    """One document's rounds and result for one instruction, read from the arrays."""

    __slots__ = ("state", "index")

    def __init__(self, state: AnswerState, index: int):
        self.state = state
        self.index = index

    @property
    def written(self) -> list:
        return self.state.written[self.index, :self.state.num_written[self.index]
                                  ].tolist()

    @property
    def fed(self) -> list:
        return self.state.fed[self.index, :self.state.fed_len[self.index]].tolist()

    @property
    def done(self) -> bool:
        return bool(self.state.done[self.index])

    @property
    def asked(self) -> bool:
        return bool(self.state.asked[self.index])

    @property
    def steps(self) -> int:
        return int(self.state.steps[self.index])

    @property
    def fed_tokens(self) -> int:
        return int(self.state.fed_tokens[self.index])

    @property
    def discard(self) -> int:
        return int(self.state.discard[self.index])

    @property
    def answer(self) -> str | None:
        return self.state.answers[self.index]


@dataclass
class MapDocument:
    """One document's prompt, its tokens for drafting, and its answers."""

    index: int
    prefix: Sequence
    tokens: np.ndarray  # the document's token ids
    key: tuple
    answers: list = field(default_factory=list)   # one MapAnswer per instruction
    rho: float = 0.0    # the document's KV read per round, in fed tokens
    counted: bool = False   # taken off the live count after its last answer


class MapDecoder:
    """The greedy token rule for a pipeline's maps, over all its documents.

    Args:
        vocabulary: The model's vocabulary index, for the stop tokens.
        k: How many likeliest tokens each readout row carries.
        width: The values in each readout row: the stop, then k ids
            and k log probabilities.
        questions: How many instructions the pipeline puts.
        spec: Map plan carrying the execution settings.
        documents: The MapDocuments, in index order.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, questions: int,
                 spec: MapSpec, documents: list):
        self.spec = spec
        self.vocabulary = vocabulary
        self.k = k
        self.width = width
        self.cache = NgramCache()
        # one acceptance table and answer-length curve per instruction
        self.policies = ([MapDraftPolicy(spec.draft_tokens, spec.max_answer_tokens,
                                         spec.expected_answer_tokens)
                          for _ in range(questions)]
                         if spec.draft_policy == "adaptive" else None)
        self.documents = documents
        num_documents = len(documents)
        self.states = [AnswerState(num_documents, spec.max_answer_tokens,
                                   spec.draft_tokens) for _ in range(questions)]
        for document in documents:
            document.answers = [MapAnswer(state, document.index)
                                for state in self.states]
        self.rho = np.array([document.rho for document in documents],
                            dtype=np.float64)
        self._index_documents()

    # ---- the documents, indexed for copying -------------------------------

    def _index_documents(self) -> None:
        """Sort every (document, token) and (document, bigram) with its position.

        A draft copies the document from where the answer's last
        written tokens occur: among the positions of the last token,
        the first whose preceding token also matches the answer, else
        the first position. The sorted keys answer that for a whole
        chunk's documents with two searches.
        """
        lengths = np.array([len(document.tokens) for document in self.documents],
                           dtype=np.int64)
        self.document_lengths = lengths
        self.document_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        tokens = (np.concatenate([np.asarray(document.tokens, dtype=np.int64)
                                  for document in self.documents])
                  if len(self.documents) else np.zeros(0, dtype=np.int64))
        self.document_tokens = tokens
        self.vocab_size = int(max(len(self.vocabulary.is_stop),
                                  int(tokens.max()) + 1 if len(tokens) else 0))
        documents = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        positions = np.arange(len(tokens), dtype=np.int64) - np.repeat(
            self.document_starts, lengths)
        scale = self.vocab_size
        single = documents * scale + tokens
        order = np.argsort(single, kind="stable")
        self.single_keys = single[order]
        self.single_positions = positions[order]
        inner = positions >= 1
        pair = ((documents[inner] * scale + np.concatenate(([0], tokens[:-1]))[inner])
                * scale + tokens[inner])
        order = np.argsort(pair, kind="stable")
        self.pair_keys = pair[order]
        self.pair_positions = positions[inner][order]

    def _source_positions(self, index: np.ndarray, state: AnswerState) -> np.ndarray:
        """Return where each answer's last written tokens occur in its document.

        Returns:
            Per document, the position of the last written token's
            occurrence, or -1 when it is not in the document or its
            occurrence ends it.
        """
        count = state.num_written[index]
        last = state.written[index, np.maximum(count - 1, 0)]
        before = state.written[index, np.maximum(count - 2, 0)]
        scale = self.vocab_size
        single = index * scale + last
        at = self._first_position(self.single_keys, self.single_positions, single)
        pair = (index * scale + before) * scale + last
        paired = self._first_position(self.pair_keys, self.pair_positions, pair)
        paired[count < 2] = -1
        at = np.where(paired >= 0, paired, at)
        at[at + 1 >= self.document_lengths[index]] = -1
        return at

    @staticmethod
    def _first_position(keys, positions, wanted) -> np.ndarray:
        """Return each wanted key's first position in the sorted keys, or -1."""
        if not len(keys):
            return np.full(len(wanted), -1, dtype=np.int64)
        at = np.searchsorted(keys, wanted)
        clipped = np.minimum(at, len(keys) - 1)
        found = keys[clipped] == wanted
        return np.where(found, positions[clipped], -1)

    # ---- a chunk's rounds, verified and drafted together --------------------

    def round(self, q: int, index: np.ndarray, rows: np.ndarray,
              num_rows: np.ndarray, slack: float) -> int:
        """Verify a chunk's rounds for instruction q and draft the next ones.

        Row i of a document follows token i of the round's fed tokens,
        so the token chosen from it is written and, when it is not the
        token fed next, the rest of the round is discarded: the next
        round feeds the chosen token in its place. The tail stage's one
        row, after the cue, chooses the answer's first token.

        Args:
            q: The instruction's index.
            index: The documents of the chunk's rounds.
            rows: Their readout rows, stacked in index order.
            num_rows: Each document's rows; a round has one per fed
                token, the tail stage one.
            slack: Fed tokens the current passes carry for free.

        Returns:
            How many of the documents' answers ended in this round.
        """
        state = self.states[q]
        cap = self.spec.max_answer_tokens
        state.steps[index] += 1
        num_documents = len(index)
        starts = np.concatenate(([0], np.cumsum(num_rows)[:-1]))
        total = int(starts[-1] + num_rows[-1]) if num_documents else 0
        segment = np.repeat(np.arange(num_documents), num_rows)
        within = np.arange(total) - np.repeat(starts, num_rows)
        top, lp_top = self._top(rows)
        ok = (lp_top > rows[:, 0]) & ~self.vocabulary.is_stop[top]
        # the draft token fed after each row; none after the last row
        fed_len = state.fed_len[index]
        has_next = within + 1 < fed_len[segment]
        following = np.where(has_next, state.fed[index[segment],
                                                 np.minimum(within + 1,
                                                            state.fed.shape[1] - 1)],
                             -1)
        match = ok & (top == following)
        # leading matches write the draft; the row after them chooses
        accepted = num_rows.copy()
        missed = np.flatnonzero(~match)
        np.minimum.at(accepted, segment[missed], within[missed])
        written = state.num_written[index]
        accepted = np.minimum(accepted, cap - written)
        writes = within < accepted[segment]
        if writes.any():
            state.written[index[segment[writes]],
                          written[segment[writes]] + within[writes]] = top[writes]
        chooses = within == accepted[segment]
        chosen = np.flatnonzero(chooses)
        fresh = np.zeros(num_documents, dtype=np.int64)
        room = written + accepted < cap
        fresh_rows = chosen[ok[chosen] & room[segment[chosen]]]
        fresh[segment[fresh_rows]] = 1
        state.written[index[segment[fresh_rows]],
                      written[segment[fresh_rows]] + accepted[segment[fresh_rows]]
                      ] = top[fresh_rows]
        ended_rows = chosen[~ok[chosen]]
        ended = np.zeros(num_documents, dtype=bool)
        ended[segment[ended_rows]] = True
        written = written + accepted + fresh
        state.num_written[index] = written
        ended |= written >= cap
        # a draft token past the chosen row was fed off the answer's path
        missed_round = fresh.astype(bool) & (accepted + 1 < num_rows)
        state.discard[index] = np.where(missed_round, num_rows - accepted - 1, 0)
        if self.policies is not None:
            self._observe(q, index, accepted, missed_round)
        newly = np.flatnonzero(ended & ~state.done[index])
        for position in newly:
            self.settle(self.documents[index[position]], q)
        if self.policies is not None:
            self._cache(state, index)
        self._draft(q, index[np.flatnonzero(~ended)], slack)
        return len(newly)

    def _top(self, rows: np.ndarray) -> tuple:
        """Return each row's likeliest token and its log probability."""
        k = self.k
        if k == 1:
            return rows[:, 1].astype(np.int64), rows[:, 2]
        lps = rows[:, -k:]
        best = np.argmax(lps, axis=1)
        which = np.arange(len(rows))
        return (rows[:, -2 * k:-k][which, best].astype(np.int64),
                lps[which, best])

    def _observe(self, q: int, index: np.ndarray, accepted: np.ndarray,
                 missed: np.ndarray) -> None:
        """Record the round's draft outcomes per policy cell.

        A draft token is a trial for its cell: written while every
        token before it was, or the round's first miss. Tokens past a
        miss were fed off the answer's path, so they teach the policy
        nothing.
        """
        state = self.states[q]
        policy = self.policies[q]
        cells = state.cells[index]
        hop = np.arange(cells.shape[1])
        tried = (hop[None, :] < accepted[:, None]) | (
            missed[:, None] & (hop[None, :] == accepted[:, None]))
        tried &= cells != NO_CELL
        tried_cells = cells[tried]
        wrote = (hop[None, :] < accepted[:, None])[tried]
        copied = tried_cells == COPY_CELL
        policy.copy_trials += int(copied.sum())
        policy.copy_accepts += int(wrote[copied].sum())
        chained = tried_cells[~copied]
        if len(chained):
            policy.trials += np.bincount(chained, minlength=len(policy.trials))
            policy.accepts += np.bincount(chained[wrote[~copied]],
                                          minlength=len(policy.accepts))

    def _cache(self, state: AnswerState, index: np.ndarray) -> None:
        """Insert the documents' newly written tokens into the n-gram cache."""
        for document in index[state.num_written[index] > state.cached[index]]:
            count = int(state.num_written[document])
            self.cache.add(state.written[document, :count].tolist(), int(document),
                           start=int(state.cached[document]))
            state.cached[document] = count

    def _draft(self, q: int, index: np.ndarray, slack: float) -> None:
        """Set the next round's fed tokens: the chosen token, then a draft.

        The draft copies the document's next tokens from the position
        where the answer's last written tokens occur, or, when the
        document does not continue the answer, chains continuations
        from the n-gram cache. The adaptive policy sizes each draft
        from its acceptance odds, the pass's slack, and the round's
        price: the document's KV read, which every round pays however
        many tokens it feeds, so a draft that saves rounds saves reads.
        The fixed policy copies up to draft_tokens from the document.
        """
        state = self.states[q]
        cap = self.spec.max_answer_tokens
        state.cells[index] = NO_CELL
        written = state.num_written[index]
        state.fed[index, 0] = state.written[index, np.maximum(written - 1, 0)]
        room = np.minimum(self.spec.draft_tokens, cap - written - 1)
        policy = None if self.policies is None else self.policies[q]
        length = np.zeros(len(index), dtype=np.int64)
        at = self._source_positions(index, state)
        at[room <= 0] = -1
        rho = self.rho[index]
        copying = at >= 0
        if copying.any():
            length[copying] = (room[copying] if policy is None
                               else policy.copy_lengths(rho[copying],
                                                        written[copying], slack,
                                                        room[copying]))
            length = np.minimum(length, self.document_lengths[index] - at - 1)
            hop = np.arange(self.spec.draft_tokens)
            source = (self.document_starts[index][:, None] + at[:, None] + 1
                      + hop[None, :])
            taking = hop[None, :] < length[:, None]
            source = np.minimum(source, max(len(self.document_tokens) - 1, 0))
            state.fed[index[:, None], 1 + hop[None, :]] = np.where(
                taking, self.document_tokens[source], 0)
            state.cells[index[:, None], hop[None, :]] = np.where(taking, COPY_CELL,
                                                                 NO_CELL)
        if policy is not None:
            for position in np.flatnonzero(~copying & (room > 0)):
                length[position] = self._chain(
                    state, int(index[position]), policy, int(room[position]),
                    float(rho[position]), slack)
        state.fed_len[index] = 1 + length

    def _chain(self, state: AnswerState, document: int, policy: MapDraftPolicy,
               room: int, rho: float, slack: float) -> int:
        """Fill one document's draft from the cache; return its length.

        The chain stops once the hops' combined odds fall below the
        cutoff: tokens past that point add under 0.02 expected writes,
        so the policy would not keep them, and each hop costs a cache
        lookup.
        """
        count = int(state.num_written[document])
        context = state.written[document, max(0, count - max(MATCH_ORDERS)):count
                                ].tolist()
        tokens, cells, rates = [], [], []
        combined = 1.0
        for _ in range(room):
            match = self.cache.lookup(context, document)
            if match is None:
                break
            cell, rate = policy.odds(match)
            combined *= rate
            if combined < CHAIN_CUTOFF:
                break
            tokens.append(match.token)
            cells.append(cell)
            rates.append(rate)
            context.append(match.token)
        length = policy.chain_length(rho, count, rates, slack)
        if length:
            state.fed[document, 1:1 + length] = tokens[:length]
            state.cells[document, :length] = cells[:length]
        return length

    def settle(self, document: MapDocument, q: int) -> None:
        """Set the instruction's answer for the document: its written text."""
        state = self.states[q]
        index = document.index
        if state.done[index]:
            return
        state.done[index] = True
        count = int(state.num_written[index])
        if self.policies is not None:
            self.policies[q].curve.settle(count)
        state.answers[index] = self.vocabulary.tokenizer.decode(
            state.written[index, :count].tolist(), skip_special_tokens=True).strip()
