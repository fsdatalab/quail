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
        decided = [
            test.decide(self.answer, prefix, may_be_null=may_be_null,
                        tails=tails, max_chars=max_chars)
            if hasattr(test, "decide") else "undecided"
            for test in (*self.stop_tests, *self.hint_tests)]
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


def copied_tails(text: str, prefix: str) -> list[str] | None:
    """Return the text after each place the document text holds the prefix.

    An extraction copies a span of its document, so an answer that
    starts with the prefix goes on with one of these texts. An empty
    prefix may go on with any text, which None stands for.
    """
    if not prefix:
        return None
    tails = []
    start = text.find(prefix)
    while start >= 0:
        tails.append(text[start + len(prefix):])
        start = text.find(prefix, start + 1)
    return tails
