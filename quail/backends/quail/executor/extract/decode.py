"""The decode of an answer: the token rule applied to each readout row.

The answer is decoded after the phrase cue as a decoder writes it, but
each token is chosen among the tokens that continue some position of
the document: the span is the document's own text wherever the
model's writing leads. Each round feeds the chosen token and the
document's next tokens as a draft, and the rows after them verify it.
The answer ends where the model would close the quote, or where its
writing would leave the document.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.extract.candidates import (
    NEG,
    DocumentTokens,
    ensure_tokens,
    first_tokens,
    make_streams,
)
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.execution.spans import line_span
from quail.extraction import ExtractionConfig


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
    prefix: Sequence
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
        config: Execution limits carried by the extraction plan.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, width: int, trim,
                 config: ExtractionConfig):
        self.config = config
        self.vocabulary = vocabulary
        self.k = k
        self.width = width
        self.trim = list(trim)
        self.scores = np.full(len(vocabulary.vocab), NEG, dtype=np.float32)
        self.previous_ids = np.empty(0, dtype=np.int64)

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
        for i in range(self.config.draft_tokens):
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
        self.scores[self.previous_ids] = NEG
        ids = values[-2 * k:-k].astype(np.int64)
        self.scores[ids] = values[-k:]
        self.previous_ids = ids
        self.top = int(ids[np.argmax(self.scores[ids])])
        return self.scores, float(values[0]), float(values[1])

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
        stream continues, or at the configured answer limit.

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
                                or self.leaves(lp, starts.allows(self.top),
                                               answer.best_lp, stop_lp))
            if answer.none_wins:
                self.settle(doc, q)
                return None
            token = starts.best_token
            answer.live = make_streams(vocabulary, doc.text, starts, token, self.config)
        else:
            at = len(answer.written)
            ensure_tokens(answer.live, at, vocabulary)
            best, joined, joined_lp, allowed, next_tokens = self.continuations(
                answer, at, lp)
            best_lp = NEG if best is None else lp[best]
            if joined is not None and joined_lp > max(best_lp, stop_lp):
                answer.written.append(joined)
                if next_tokens is not None:
                    answer.live = [s for s, keep in zip(
                        answer.live, next_tokens == joined) if keep]
                self.settle(doc, q)
                return None
            if (best is None or stop_lp >= best_lp
                    or self.leaves(lp, allowed, best_lp, stop_lp)):
                self.settle(doc, q)
                return None
            token = best
            if next_tokens is not None:
                answer.live = [s for s, keep in zip(answer.live, next_tokens == token)
                               if keep]
        answer.written.append(token)
        if len(answer.written) >= self.config.max_answer_tokens:
            self.settle(doc, q)
        return token

    def continuations(self, answer: AnswerState, at: int, lp) -> tuple:
        """Rate the next tokens and quote endings of the answer's live streams."""
        merged_ids = self.vocabulary.merged
        if len(answer.live) == 1:
            stream = answer.live[0].stream
            best = stream[at] if at < len(stream) else None
            merged = int(merged_ids[best]) if best is not None else -1
            joined_lp = lp[merged] if merged >= 0 else NEG
            joined = best if joined_lp > NEG else None
            allowed = self.top == best or self.top == merged
            return best, joined, joined_lp, allowed, None
        next_tokens = np.fromiter(
            (s.stream[at] if at < len(s.stream) else -1 for s in answer.live),
            dtype=np.int64, count=len(answer.live))
        following = np.unique(next_tokens[next_tokens >= 0])
        best = int(following[np.argmax(lp[following])]) if len(following) else None
        merged = merged_ids[following]
        valid = merged >= 0
        originals, merged = following[valid], merged[valid]
        rated = lp[merged] > NEG
        originals, merged = originals[rated], merged[rated]
        index = int(np.argmax(lp[merged])) if len(merged) else None
        joined = int(originals[index]) if index is not None else None
        joined_lp = lp[merged[index]] if index is not None else NEG
        allowed = bool(np.any(following == self.top) or np.any(merged == self.top))
        return best, joined, joined_lp, allowed, next_tokens

    def leaves(self, lp, top_allowed: bool, best_lp, stop_lp) -> bool:
        """Return whether the model's likeliest token leaves the document."""
        vocabulary = self.vocabulary
        top = self.top
        return (lp[top] > max(best_lp, stop_lp) and not top_allowed
                and not vocabulary.is_stop[top] and top != vocabulary.none_id)

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
