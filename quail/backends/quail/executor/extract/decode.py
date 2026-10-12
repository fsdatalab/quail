"""The decode of an answer: the token rule applied to each readout row.

The answer is decoded after the phrase cue as a decoder writes it, but
each token is chosen among the tokens that continue some position of
the document: the span is the document's own text wherever the
model's writing leads. Each round feeds the chosen token and the
document's next tokens as a draft, and the rows after them verify it.
The answer ends where the model would close the quote, or where its
writing would leave the document.
"""

from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    NEG,
    DocumentTokens,
    first_tokens,
    make_streams,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.cost.extract import ANSWER_MAX_TOKENS, DRAFT_TOKENS
from quail.execution.spans import line_span

CUE_TOP = 1024      # vocabulary entries read at the cue and each round


@dataclass
class AnswerState:
    """One document's rounds and result for one question."""

    asked: bool = False     # the document reached the question's stages
    none_lp: float = NEG
    best_lp: float = NEG
    written: list = field(default_factory=list)   # the answer's tokens so far
    live: list = field(default_factory=list)      # streams the writing fits
    fed: list = field(default_factory=list)       # the last round's tokens
    discard: int = 0    # tokens of the last round past what it verified
    done: bool = False
    steps: int = 0
    fed_tokens: int = 0
    answer: str | None = None
    span: tuple | None = None
    none_wins: bool = False


@dataclass
class Document:
    """One document's prompt and its answer to each question."""

    index: int
    text: str
    prefix: list
    scan: DocumentTokens
    key: tuple
    answers: list       # one AnswerState per question


class Decoder:
    """The token rule over one model's vocabulary, for a pipeline's questions.

    Args:
        vocabulary: The model's vocabulary index.
        k: How many likeliest tokens each readout row carries.
        width: The values in each readout row: none, the stop, then
            k ids and k log probabilities.
        trim: Per question, whether the answer is the copied words or
            the whole lines that hold them.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, trim):
        self.vocabulary = vocabulary
        self.k = k
        self.width = width
        self.trim = list(trim)

    def rows(self, row) -> np.ndarray:
        """Return a stage's answer as its rows of readout values."""
        values = np.asarray(row)
        if values.dtype == object:
            values = np.asarray(values.reshape(-1)[0], dtype=np.float32)
        return values.reshape(-1, self.width)

    def draft(self, answer: AnswerState) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft.

        The draft is the document's next tokens along the first
        position the writing fits; the rows after them verify it, so a
        round writes as many tokens as the model would have chosen one
        at a time.
        """
        at = len(answer.written)
        stream = min(answer.live, key=lambda s: s.char)
        draft = []
        for i in range(DRAFT_TOKENS):
            token = stream.next(at + i, self.vocabulary)
            if token is None:
                break
            draft.append(token)
        answer.fed = [answer.written[-1], *draft]
        return answer.fed

    def lp(self, values) -> tuple:
        """Return a row's log probabilities by token id, and none's and the stop's."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        k = self.k
        lp = dict(zip(values[-2 * k:-k].astype(np.int64).tolist(),
                      values[-k:].tolist()))
        return lp, float(values[0]), float(values[1])

    def verify(self, doc: Document, q: int, rows, fed) -> None:
        """Choose the answer's tokens from a round's rows, in order.

        Row i follows token i of the round's fed tokens, so the token
        chosen from it is written and, when it is not the token fed
        next, the rest of the round is discarded: the next round feeds
        the chosen token in its place.
        """
        answer = doc.answers[q]
        answer.steps += 1
        answer.discard = 0
        for i, values in enumerate(rows):
            token = self.choose(doc, q, values)
            if answer.done:
                return
            if i + 1 < len(fed) and token != fed[i + 1]:
                answer.discard = len(fed) - (i + 1)
                return

    def choose(self, doc: Document, q: int, values):
        """Choose the answer's next token among the document's continuations.

        The answer is the document text the model writes before its
        writing would leave the document: a token it likes better than
        every continuation and the stop, and that continues no
        position, ends the answer, and at the cue it answers none, as
        does the model writing none or closing the quote at once. At
        the cue, the token is the likeliest first token of any
        position in the document; afterwards, the likeliest continuation
        of the streams the writing so far fits. A token the model
        would write with the closing quote, such as `."`, ends the
        answer after that token. The answer also ends when closing the
        quote is at least as likely as every continuation, when no
        stream continues, or at ANSWER_MAX_TOKENS.

        Returns:
            The token written, or None when the answer ended.
        """
        answer = doc.answers[q]
        vocabulary = self.vocabulary
        lp, none_lp, stop_lp = self.lp(values)
        if not answer.written:
            starts = first_tokens(vocabulary, lp, doc.prefix, None, scan=doc.scan)
            answer.none_lp = none_lp
            answer.best_lp = starts.best_lp
            answer.none_wins = (max(none_lp, stop_lp) > answer.best_lp
                                or self.leaves(lp, starts.allowed, answer.best_lp,
                                               stop_lp))
            if answer.none_wins:
                self.settle(doc, q)
                return None
            token = starts.best_token
            answer.live = make_streams(vocabulary, doc.text, starts, token)
        else:
            at = len(answer.written)
            following = {token for token in (s.next(at, vocabulary)
                                             for s in answer.live)
                         if token is not None}
            best = max(following, key=lambda t: lp.get(t, NEG), default=None)
            best_lp = NEG if best is None else lp.get(best, NEG)
            merged = {t: vocabulary.merged[t] for t in following
                      if t in vocabulary.merged and vocabulary.merged[t] in lp}
            joined = max(merged, key=lambda t: lp[merged[t]], default=None)
            if joined is not None and lp[merged[joined]] > max(best_lp, stop_lp):
                answer.written.append(joined)
                answer.live = [s for s in answer.live
                               if s.next(at, vocabulary) == joined]
                self.settle(doc, q)
                return None
            if (best is None or stop_lp >= best_lp
                    or self.leaves(lp, following | set(merged.values()),
                                   best_lp, stop_lp)):
                self.settle(doc, q)
                return None
            token = best
            answer.live = [s for s in answer.live if s.next(at, vocabulary) == token]
        answer.written.append(token)
        if len(answer.written) >= ANSWER_MAX_TOKENS:
            self.settle(doc, q)
        return token

    def leaves(self, lp, allowed, best_lp, stop_lp) -> bool:
        """Return whether the model's likeliest token leaves the document."""
        vocabulary = self.vocabulary
        top = max(lp, key=lp.get)
        return (lp[top] > max(best_lp, stop_lp) and top not in allowed
                and top not in vocabulary._stops and top != vocabulary.none_id)

    def settle(self, doc: Document, q: int) -> None:
        """Set the question's answer for the document: its span, or none.

        Among the streams the writing fits, the first position of the
        document wins.
        """
        answer = doc.answers[q]
        answer.done = True
        if answer.none_wins or not answer.written or not answer.live:
            answer.answer, answer.span = None, None
            return
        stream = min(answer.live, key=lambda s: s.char)
        start, end = stream.char, stream.ends[len(answer.written) - 1]
        text = doc.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if not self.trim[q]:
            start, end = line_span(text, (start, end))
        answer.answer, answer.span = text[start:end], (start, end)
