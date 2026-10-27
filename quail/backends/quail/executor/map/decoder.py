"""The decode of a map answer: the greedy choice at each readout row.

The answer is decoded after the answer cue, one round at a time. Each
round feeds the token the round before it chose, and the readout row
after that token chooses the next one: the likeliest token, written
unless it ends the model's turn. The answer ends there, or at the
configured answer limit.

A round is launched before the round before it is decided, so a
document runs in every pass: the GPU puts the token the earlier round
chose into the round's one row, and the host decides that round one
pass later.

The state of every document's answer lives in arrays over the
documents, and a chunk's rounds are decided together in a few array
passes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.vocabulary import Vocabulary
from quail.cost.draft_policy import AnswerLengthCurve


class AnswerState:
    """Every document's answer to one instruction, in arrays over the documents.

    Args:
        num_documents: How many documents the pipeline holds.
        max_answer_tokens: The answer limit, the width of written.
    """

    def __init__(self, num_documents: int, max_answer_tokens: int):
        shape = (num_documents,)
        self.written = np.zeros((num_documents, max_answer_tokens), dtype=np.int64)
        self.num_written = np.zeros(shape, dtype=np.int64)
        self.done = np.zeros(shape, dtype=bool)
        self.asked = np.zeros(shape, dtype=bool)
        self.steps = np.zeros(shape, dtype=np.int64)
        self.fed_tokens = np.zeros(shape, dtype=np.int64)
        # the next round's token, known once the round before it is decided
        self.fed = np.zeros(shape, dtype=np.int64)
        # whether the tail is launched; the launched rounds, the tail
        # among them, whose rows are not decided; and the position of
        # the oldest such round's token, or the next round's
        self.started = np.zeros(shape, dtype=bool)
        self.unreported = np.zeros(shape, dtype=np.int64)
        self.base = np.zeros(shape, dtype=np.int64)
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
    def done(self) -> bool:
        return bool(self.state.done[self.index])

    @property
    def steps(self) -> int:
        return int(self.state.steps[self.index])

    @property
    def fed_tokens(self) -> int:
        return int(self.state.fed_tokens[self.index])

    @property
    def answer(self) -> str | None:
        return self.state.answers[self.index]


@dataclass
class MapDocument:
    """One document's prompt and its answers."""

    index: int
    prefix: Sequence
    key: tuple
    answers: list = field(default_factory=list)   # one MapAnswer per instruction
    counted: bool = False   # taken off the live count after its last answer


class MapDecoder:
    """The greedy token rule for a pipeline's maps, over all its documents.

    Args:
        vocabulary: The model's vocabulary index, for the stop tokens.
        k: How many likeliest tokens each readout row carries.
        width: The values in each readout row: the stop, then k ids
            and k log probabilities.
        specs: The pipeline's MapSpecs, one per instruction, each with
            its own answer limit and expected answer length.
        documents: The MapDocuments, in index order.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, specs: list,
                 documents: list):
        self.caps = [spec.max_answer_tokens for spec in specs]
        self.vocabulary = vocabulary
        self.k = k
        self.width = width
        # one answer-length curve per instruction, for sizing each
        # document's KV
        self.curves = [AnswerLengthCurve(spec.max_answer_tokens,
                                         spec.expected_answer_tokens)
                       for spec in specs]
        self.documents = documents
        num_documents = len(documents)
        self.states = [AnswerState(num_documents, spec.max_answer_tokens)
                       for spec in specs]
        # per instruction, survival() and the round count it was built at
        self._survival = [(None, -1) for _ in specs]
        self._versions = [0] * len(specs)
        self._fresh = {}
        for document in documents:
            document.answers = [MapAnswer(state, document.index)
                                for state in self.states]

    def launch(self, q: int, index: np.ndarray, tail: bool,
               positions: np.ndarray) -> None:
        """Record that the documents' tail or next round is on the stream.

        Args:
            q: The instruction's index.
            index: The documents launched.
            tail: Whether the tail was launched, rather than a round.
            positions: Per document in index, its cue's position; a
                tail launched first starts the answer there.
        """
        state = self.states[q]
        if tail:
            new = ~state.started[index]
            state.started[index[new]] = True
            state.base[index[new]] = np.asarray(positions)[new]
        state.unreported[index] += 1

    def round(self, q: int, index: np.ndarray, rows: np.ndarray) -> int:
        """Decide a chunk's oldest launched rounds for instruction q.

        Each document's one readout row chooses its next token: the
        likeliest, written unless it beats no stop token or the answer
        is at its limit. The tail stage's row, after the cue, chooses
        the answer's first token. The next round feeds the token
        written last.

        Args:
            q: The instruction's index.
            index: The documents of the chunk's rounds.
            rows: Their readout rows, one per document, in index order.

        Returns:
            How many of the documents' answers ended in this round.
        """
        if len(rows) != len(index):
            raise ValueError("a round has one readout row per document")
        state = self.states[q]
        cap = self.caps[q]
        self._versions[q] += 1
        state.steps[index] += 1
        top, lp_top = self._top(rows)
        ok = (lp_top > rows[:, 0]) & ~self.vocabulary.is_stop[top]
        written = state.num_written[index]
        writes = ok & (written < cap)
        state.written[index[writes], written[writes]] = top[writes]
        written = written + writes
        state.num_written[index] = written
        ended = ~ok | (written >= cap)
        state.unreported[index] -= 1
        state.base[index] += 1
        newly = np.flatnonzero(ended & ~state.done[index])
        for position in newly:
            self.settle(self.documents[index[position]], q)
        going = ~ended
        state.fed[index[going]] = state.written[index[going], written[going] - 1]
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

    def survival(self, q: int) -> np.ndarray:
        """Return P(an answer has more than j tokens) for j = 0 to the limit.

        A Kaplan-Meier estimate: answers still being written count as
        lasting at least as long as they have written, so the long
        answers that settle last are not missing from it while the
        short ones settle first. Where no answer has been seen the
        estimate stays flat, so an unseen length is taken to go on.
        It is built once per round of the instruction.
        """
        cached, version = self._survival[q]
        if version == self._versions[q]:
            return cached
        state = self.states[q]
        cap = self.caps[q]
        settled = self.curves[q].lengths.astype(np.float64)
        writing = state.asked & ~state.done
        running = np.bincount(np.minimum(state.num_written[writing], cap),
                              minlength=cap + 1).astype(np.float64)
        # at risk at length j: answers that reached j tokens
        at_risk = (np.cumsum(settled[::-1])[::-1] + np.cumsum(running[::-1])[::-1])
        hazard = np.divide(settled, at_risk, out=np.zeros_like(settled),
                           where=at_risk > 0)
        survival = np.cumprod(1.0 - hazard)
        self._survival[q] = survival, self._versions[q]
        return survival

    def remaining(self, q: int, index: np.ndarray, quantile: float) -> np.ndarray:
        """Return the answer tokens each document is expected to still write.

        The count is the quantile of the instruction's answer lengths,
        given the tokens each answer has written, from survival(); a
        settled answer writes none.

        Args:
            q: The instruction's index.
            index: The documents.
            quantile: The fraction of answers expected to end within
                the count.
        """
        state = self.states[q]
        survival = self.survival(q)
        written = state.num_written[index]
        # P(longer than written - 1) conditions on what is written;
        # survival never rises, so the first length at or below the
        # threshold ends the quantile
        before = np.concatenate(([1.0], survival))[written]
        end = np.searchsorted(-survival, -(1.0 - quantile) * before, side="left")
        left = np.clip(end - written, 0, self.caps[q] - written)
        left[state.done[index]] = 0
        return left

    def fresh_remaining(self, q: int, quantile: float) -> int:
        """Return remaining() for an answer not yet written, once per round."""
        key = (q, quantile, self._versions[q])
        cached = self._fresh.get(q)
        if cached is None or cached[0] != key:
            survival = self.survival(q)
            end = int(np.searchsorted(-survival, -(1.0 - quantile), side="left"))
            cached = key, min(end, self.caps[q])
            self._fresh[q] = cached
        return cached[1]

    def restart(self, q: int, index: np.ndarray) -> None:
        """Forget the documents' unfinished answers to instruction q.

        Their steps and fed tokens are kept, as work done.
        """
        state = self.states[q]
        if state.done[index].any():
            raise ValueError("a settled answer cannot restart")
        self._versions[q] += 1
        state.num_written[index] = 0
        state.asked[index] = False
        state.fed[index] = 0
        state.started[index] = False
        state.unreported[index] = 0
        state.base[index] = 0

    def settle(self, document: MapDocument, q: int) -> None:
        """Set the instruction's answer for the document: its written text."""
        state = self.states[q]
        index = document.index
        if state.done[index]:
            return
        state.done[index] = True
        self._versions[q] += 1
        count = int(state.num_written[index])
        self.curves[q].settle(count)
        state.answers[index] = self.vocabulary.tokenizer.decode(
            state.written[index, :count].tolist(), skip_special_tokens=True).strip()
