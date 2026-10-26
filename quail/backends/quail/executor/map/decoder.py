"""The decode of a map answer: the greedy choice at each readout row.

The answer is decoded after the answer cue, one round at a time. Each
round feeds the chosen token and a draft copied from the document
where the last written tokens occur, and the rows after them verify
it: a draft token is written exactly when the model would have chosen
it one token at a time. The answer ends where the model ends its
turn, or at the configured answer limit.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from quail.backends.quail.executor.vocabulary import Vocabulary
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


@dataclass
class MapDocument:
    """One document's prompt, its tokens for drafting, and its answers."""

    prefix: Sequence
    tokens: np.ndarray  # the document's token ids
    key: tuple
    answers: list       # one MapAnswer per instruction
    positions: dict | None = None   # token id -> its positions, built lazily


class MapDecoder:
    """The greedy token rule for a pipeline's maps.

    Args:
        vocabulary: The model's vocabulary index, for the stop tokens.
        k: How many likeliest tokens each readout row carries.
        spec: Map plan carrying the execution settings.
    """

    def __init__(self, vocabulary: Vocabulary, k: int, spec: MapSpec):
        self.spec = spec
        self.vocabulary = vocabulary
        self.k = k

    def draft(self, answer: MapAnswer, document: MapDocument) -> list:
        """Return the tokens the next round feeds: the chosen token, then a draft.

        The draft copies the document's next tokens from the position
        where the answer's last written tokens occur; the rows after
        them verify it, so a round writes as many tokens as the model
        would have chosen one at a time. A token the document does not
        hold gets no draft.
        """
        draft = []
        length = min(self.spec.draft_tokens,
                     self.spec.max_answer_tokens - len(answer.written) - 1)
        if length > 0:
            at = self._source_position(answer, document)
            if at is not None:
                draft = [int(token)
                         for token in document.tokens[at + 1:at + 1 + length]]
        answer.fed = [answer.written[-1], *draft]
        return answer.fed

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
        for i, values in enumerate(rows):
            token = self.choose(document, q, values)
            if answer.done:
                return
            after = i + 1
            if after < len(fed) and token != fed[after]:
                answer.discard = len(fed) - after
                return

    def choose(self, document: MapDocument, q: int, values):
        """Choose the answer's next token: the model's likeliest.

        The answer ends when the likeliest token is a stop token or at
        the configured answer limit.

        Returns:
            The token written, or None when the answer ended.
        """
        answer = document.answers[q]
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        k = self.k
        ids = values[-2 * k:-k].astype(np.int64)
        lps = values[-k:]
        best = int(np.argmax(lps))
        token = int(ids[best])
        if self.vocabulary.is_stop[token]:
            self.settle(document, q)
            return None
        answer.written.append(token)
        if len(answer.written) >= self.spec.max_answer_tokens:
            self.settle(document, q)
        return token

    def settle(self, document: MapDocument, q: int) -> None:
        """Set the instruction's answer for the document: its written text."""
        answer = document.answers[q]
        answer.done = True
        answer.answer = self.vocabulary.tokenizer.decode(
            answer.written, skip_special_tokens=True).strip()
