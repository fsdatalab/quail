"""The decode of a map answer: the greedy choice at each readout row.

The answer is decoded after the answer cue, one round at a time. Each
round feeds the chosen token and a draft, and the rows after them
verify it: a draft token is written exactly when the model would have
chosen it one token at a time. A draft is copied from the document
where the last written tokens occur, or chained from the n-gram cache
of the answers written so far; the adaptive policy sizes it from the
acceptance odds and the forward pass's slack. The answer ends where
the model ends its turn, or at the configured answer limit.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.vocabulary import Vocabulary
from quail.cost.draft_policy import BUSY_ROUND_TOKENS, MapDraftPolicy
from quail.cost.ngram_cache import MATCH_ORDERS, NgramCache
from quail.physical import MapSpec


@dataclass
class MapAnswer:
    """One document's rounds and result for one instruction."""

    asked: bool = False     # the document reached the instruction's stages
    written: list = field(default_factory=list)   # the answer's tokens so far
    fed: list = field(default_factory=list)       # the last round's tokens
    discard: int = 0    # tokens of the last round past what it verified
    done: bool = False
    steps: int = 0
    fed_tokens: int = 0
    answer: str | None = None
    cells: list = field(default_factory=list)  # the last draft's policy cells
    cached: int = 0     # written tokens inserted into the cache


@dataclass
class MapDocument:
    """One document's prompt, its tokens for drafting, and its answers."""

    index: int
    prefix: Sequence
    tokens: np.ndarray  # the document's token ids
    key: tuple
    answers: list       # one MapAnswer per instruction
    positions: dict | None = None   # token id -> its positions, built lazily
    rho: float = 0.0    # the document's KV read per round, in fed tokens
    counted: bool = False   # taken off the live count after its last answer


class MapDecoder:
    """The greedy token rule for a pipeline's maps.

    Args:
        vocabulary: The model's vocabulary index, for the stop tokens.
        k: How many likeliest tokens each readout row carries.
        width: The values in each readout row: the stop, then k ids
            and k log probabilities.
        questions: How many instructions the pipeline puts.
        spec: Map plan carrying the execution settings.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, questions: int,
                 spec: MapSpec):
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

    def draft(self, answer: MapAnswer, document: MapDocument, q: int = 0,
              slack: float = 0.0) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft.

        The draft copies the document's next tokens from the position
        where the answer's last written tokens occur, or, when the
        document does not continue the answer, chains continuations
        from the n-gram cache. The rows after them verify it, so a
        round writes as many tokens as the model would have chosen one
        at a time. The adaptive policy sizes each draft from its
        acceptance odds, the document's KV read, and the pass's slack;
        the fixed policy copies up to draft_tokens from the document.

        Args:
            answer: The answer's state.
            document: The document.
            q: The instruction's index.
            slack: Fed tokens the current passes carry for free.
        """
        draft, cells = [], []
        policy = None if self.policies is None else self.policies[q]
        room = min(self.spec.draft_tokens,
                   self.spec.max_answer_tokens - len(answer.written) - 1)
        # in a full pass the KV read rides other rows' compute, so a
        # round's own cost is its dispatch, not the read
        rho = document.rho if slack > 0 else BUSY_ROUND_TOKENS
        if room > 0:
            at = self._source_position(answer, document)
            if at is not None:
                length = room if policy is None else policy.copy_length(
                    rho, len(answer.written), slack)
                draft = [int(token)
                         for token in document.tokens[at + 1:at + 1 + length]]
                cells = [None] * len(draft)
            elif policy is not None:
                if slack <= 0:
                    # a long chain cannot pay for itself in a full pass
                    room = min(room, 4)
                draft, cells = self._chain(answer, document, policy, room, rho,
                                           slack)
        answer.fed = [answer.written[-1], *draft]
        answer.cells = cells
        return answer.fed

    def _chain(self, answer: MapAnswer, document: MapDocument,
               policy: MapDraftPolicy, room: int, rho: float,
               slack: float) -> tuple:
        """Return a draft chained from the cache, and each hop's cell.

        The chain stops once the hops' combined odds fall below 2%:
        tokens past that point add under 0.02 expected writes, so the
        policy would not keep them, and each hop costs a cache lookup.
        """
        context = list(answer.written[-max(MATCH_ORDERS):])
        tokens, cells, rates = [], [], []
        combined = 1.0
        for _ in range(room):
            match = self.cache.lookup(context, document.index)
            if match is None:
                break
            cell, rate = policy.odds(match)
            combined *= rate
            if combined < 0.02:
                break
            tokens.append(match.token)
            cells.append(cell)
            rates.append(rate)
            context.append(match.token)
        length = policy.chain_length(rho, len(answer.written), rates, slack)
        return tokens[:length], cells[:length]

    def _source_position(self, answer: MapAnswer,
                         document: MapDocument) -> int | None:
        """Return where the answer's last written tokens occur in the document.

        Among the positions of the last written token, the first whose
        preceding token also matches the written text wins; without
        one, the first position. None when the token is not in the
        document or its occurrence ends it.
        """
        if document.positions is None:
            positions = {}
            for at, token in enumerate(document.tokens):
                positions.setdefault(int(token), []).append(at)
            document.positions = positions
        found = document.positions.get(int(answer.written[-1]))
        if not found:
            return None
        if len(answer.written) > 1:
            previous = int(answer.written[-2])
            for at in found:
                if at > 0 and int(document.tokens[at - 1]) == previous:
                    return at if at + 1 < len(document.tokens) else None
        at = found[0]
        return at if at + 1 < len(document.tokens) else None

    def verify(self, document: MapDocument, q: int, rows, fed) -> None:
        """Choose the answer's tokens from a round's rows, in order.

        Row i follows token i of the round's fed tokens, so the token
        chosen from it is written and, when it is not the token fed
        next, the rest of the round is discarded: the next round feeds
        the chosen token in its place.

        Args:
            document: The document.
            q: The instruction's index.
            rows: The readout rows of the round.
            fed: The round's fed tokens.
        """
        answer = document.answers[q]
        answer.steps += 1
        answer.discard = 0
        rows = np.asarray(rows)
        accepted = 0
        if answer.written and len(fed) > 1:
            accepted = self.accepted(answer, rows, fed)
            rows = rows[accepted:]
        hits, missed = accepted, False
        if accepted:
            written = min(accepted, self.spec.max_answer_tokens - len(answer.written))
            answer.written.extend(fed[1:written + 1])
            if len(answer.written) >= self.spec.max_answer_tokens:
                self._close_round(document, q, hits, missed)
                self.settle(document, q)
                return
        for i, values in enumerate(rows):
            token = self.choose(document, q, values)
            if answer.done:
                break
            after = accepted + i + 1
            if after < len(fed):
                if token == fed[after]:
                    hits += 1
                else:
                    answer.discard = len(fed) - after
                    missed = True
                    break
        self._close_round(document, q, hits, missed)

    def _close_round(self, document: MapDocument, q: int, hits: int,
                     missed: bool) -> None:
        """Record the round's draft outcomes and cache its written tokens.

        A chained draft token is a trial for its cell: written while
        every token before it was, or the round's first miss. Tokens
        past a miss were fed off the answer's path, so they teach the
        policy nothing.
        """
        answer = document.answers[q]
        if self.policies is not None:
            policy = self.policies[q]
            for j, cell in enumerate(answer.cells):
                if j > hits or (j == hits and not missed):
                    break
                if cell is None:
                    policy.observe_copy(j < hits)
                else:
                    policy.observe(cell, j < hits)
        answer.cells = []
        self.cache.add(answer.written, document.index, start=answer.cached)
        answer.cached = len(answer.written)

    def accepted(self, answer: MapAnswer, rows: np.ndarray, fed) -> int:
        """Return how many leading rows write the draft token fed after them.

        A row writes the draft token exactly when choose() would: the
        draft token is the model's likeliest, it beats ending the
        turn, and it is not a stop token. Those tests read a few
        values per row, so every row of the round is decided at once;
        the first row that fails, and the rows after it, go through
        choose().
        """
        k = self.k
        n = min(len(rows), len(fed) - 1,
                self.spec.max_answer_tokens - len(answer.written))
        if n <= 0:
            return 0
        if n == 1 or k == 1:
            # a one-token draft is the common round, and the top-1
            # readout decides a row with three scalar reads
            written = 0
            for values in rows[:n]:
                top, lp_top = self._top(values)
                if (top != fed[1 + written] or float(values[0]) >= lp_top
                        or self.vocabulary.is_stop[top]):
                    break
                written += 1
            return written
        values = rows[:n]
        ids = values[:, -2 * k:-k].astype(np.int64)
        lps = values[:, -k:]
        draft = np.asarray(fed[1:n + 1], dtype=np.int64)
        index = np.arange(n)
        top_col = lps.argmax(axis=1)
        top, lp_top = ids[index, top_col], lps[index, top_col]
        stop = values[:, 0]
        writes = ((top == draft) & (lp_top > stop)
                  & ~self.vocabulary.is_stop[top])
        return int(np.argmin(writes)) if not writes.all() else n

    def _top(self, values) -> tuple:
        """Return a row's likeliest token and its log probability."""
        k = self.k
        if k == 1:
            return int(values[1]), float(values[2])
        lps = values[-k:]
        best = int(np.argmax(lps))
        return int(values[-2 * k + best]), float(lps[best])

    def choose(self, document: MapDocument, q: int, values):
        """Choose the answer's next token: the model's likeliest.

        The answer ends when ending the turn is at least as likely as
        every token, when the likeliest token is a stop token, or at
        the configured answer limit.

        Returns:
            The token written, or None when the answer ended.
        """
        answer = document.answers[q]
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        token, lp_top = self._top(values)
        stop_lp = float(values[0])
        if stop_lp >= lp_top or self.vocabulary.is_stop[token]:
            self.settle(document, q)
            return None
        answer.written.append(token)
        if len(answer.written) >= self.spec.max_answer_tokens:
            self.settle(document, q)
        return token

    def settle(self, document: MapDocument, q: int) -> None:
        """Set the instruction's answer for the document: its written text."""
        answer = document.answers[q]
        if answer.done:
            return
        answer.done = True
        if self.policies is not None:
            self.policies[q].curve.settle(len(answer.written))
        answer.answer = self.vocabulary.tokenizer.decode(
            answer.written, skip_special_tokens=True).strip()
