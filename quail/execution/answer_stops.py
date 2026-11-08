"""Stop a map's or extraction's answer once its tests are decided.

While the model writes an answer, the tokens decoded so far limit the
complete answers still possible. A test is decided TRUE when every one
of them passes it, and decided not TRUE when none does; otherwise it is
undecided (see ``Expression.decide``). The executor stops an answer,
and frees its KV, once a stop test or a hint test is decided not TRUE:
the region drops the row. When nothing else reads the answer, it also
stops once every stop test is decided TRUE: the region keeps the row
whatever the rest of the answer says.

The answer then holds its text so far, which is itself one of the
complete answers still possible, so the region's tests give it the
value they were decided to have.

An extraction's answer is a span of its document with the whitespace
around it stripped. While the copy decoder still has several document
positions in play, the answer may become the span of any of them, so a
test is decided only when it is decided the same way for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stop(Enum):
    """What to do with an answer after its tokens so far."""

    CONTINUE = "continue"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


@dataclass(frozen=True)
class AnswerStop:
    """A map's or extraction's tests of its answer, as the module explains.

    Attributes:
        answer: The name of the column that holds the answer.
        stop_tests: The region's exact tests of the answer alone.
        hint_tests: Tests of the answer the node may drop rows by.
        when_true: Whether the answer stops once every stop test is
            decided TRUE.
    """

    answer: str
    stop_tests: tuple
    hint_tests: tuple
    when_true: bool

    def decide(self, prefix: str, *, may_be_null: bool = False,
               tails: list[str] | None = None,
               max_chars: int | None = None) -> Stop:
        """Return whether to stop an answer whose text so far is prefix.

        Args:
            prefix: The answer's text so far.
            may_be_null: Whether the answer may still be NULL.
            tails: The texts the answer may go on with, each cut
                anywhere, or None for any text.
            max_chars: The most characters the answer may add, or None.
        """
        return self.decide_any([dict(prefix=prefix, may_be_null=may_be_null,
                                     tails=tails, max_chars=max_chars)])

    def decide_any(self, completions: list[dict]) -> Stop:
        """Return whether to stop an answer that may go on as any of completions.

        A test is decided only when it is decided the same way for every
        completion, as when an extraction may still copy any of several
        document positions.

        Args:
            completions: Each a set of complete answers still possible,
                as keyword arguments of ``Expression.decide``: prefix,
                and optionally may_be_null, tails, and max_chars.
        """
        if not completions:
            return Stop.CONTINUE
        decided = []
        for test in (*self.stop_tests, *self.hint_tests):
            found = {test.decide(self.answer, **completion)
                     if hasattr(test, "decide") else "undecided"
                     for completion in completions}
            decided.append(found.pop() if len(found) == 1 else "undecided")
        if "not true" in decided:
            return Stop.REJECTED
        exact = decided[:len(self.stop_tests)]
        if self.when_true and exact and all(found == "true" for found in exact):
            return Stop.ACCEPTED
        return Stop.CONTINUE


def answer_stop(node) -> AnswerStop | None:
    """Return a map's or extraction's answer stop, or None without tests."""
    stop_tests = tuple(getattr(node, "stop_tests", ()))
    hint_tests = tuple(getattr(node, "hint_tests", ()))
    if not stop_tests and not hint_tests:
        return None
    return AnswerStop(node.spec.name, stop_tests, hint_tests,
                      bool(getattr(node, "stop_when_true", False)))


def copied_completions(text: str, spans) -> list[dict]:
    """Return the complete answers an extraction may still give, by span.

    An extraction copies a span of its document text and strips the
    whitespace around it. Each (start, end) is the span one document
    position still in play covers so far; its answer so far is that
    span stripped, and goes on only with the document text after it.

    Args:
        text: The document text.
        spans: Each live position's start and end characters so far.

    Returns:
        One set of completions per span, as ``AnswerStop.decide_any``
        reads them.
    """
    completions = []
    for start, end in spans:
        span = text[start:end]
        prefix = span.strip()
        if prefix:
            tail = text[start + len(span.rstrip()):]
        else:
            tail = text[start:].lstrip()
        completions.append(dict(prefix=prefix, tails=[tail]))
    return completions
