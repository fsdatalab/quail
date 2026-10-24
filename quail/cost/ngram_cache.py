"""The continuations seen after each run of recent tokens, with counts.

A map's draft needs a guess at the model's next tokens even where the
document does not continue the answer. The cache indexes every run of
the last 2 to 4 committed tokens to the tokens that followed it in
this query run: the answers written so far, across documents. Counts
seed the acceptance estimate, and the document that last inserted a
continuation says whether a match comes from the drafting document's
own output or a sibling's.
"""

from dataclasses import dataclass

# longest match first: a guess from more context is more reliable
MATCH_ORDERS = (4, 3, 2)
# distinct continuations kept per key; a fuller key stops growing
MAX_CONTINUATIONS = 4


@dataclass(frozen=True)
class NgramMatch:
    """One lookup's best continuation and the features that price it.

    Attributes:
        token: The continuation seen most often after the key.
        order: The key's tokens.
        fan_out: Distinct continuations seen after the key.
        share: The continuation's count over the key's count, the
            seed of the acceptance estimate.
        own: Whether the drafting document's own output last inserted
            the continuation.
    """

    token: int
    order: int
    fan_out: int
    share: float
    own: bool


class _Entry:
    """One key's continuations, with the best one kept up to date.

    The best continuation is the one seen most often; on a tie the
    one that reached that count first keeps the place.
    """

    __slots__ = ("continuations", "total", "best", "best_seen")

    def __init__(self):
        self.continuations = {}    # token -> [count, last inserting document]
        self.total = 0
        self.best = None
        self.best_seen = 0

    def __len__(self):
        return len(self.continuations)


class NgramCache:
    """Continuation counts for one query run, shared across documents."""

    def __init__(self):
        # key -> its entry
        self.entries = {}

    def add(self, tokens, document: int, start: int = 0) -> None:
        """Insert the continuations the tokens show from start on.

        Args:
            tokens: One document's committed answer tokens so far, as a
                list of Python ints; the decoder writes them that way.
            document: The document that wrote them.
            start: The first position not inserted yet.
        """
        entries = self.entries
        for at in range(max(start, MATCH_ORDERS[-1]), len(tokens)):
            token = tokens[at]
            for order in MATCH_ORDERS:
                if at < order:
                    continue
                key = tuple(tokens[at - order:at])
                entry = entries.get(key)
                if entry is None:
                    entry = entries[key] = _Entry()
                found = entry.continuations.get(token)
                if found is None:
                    if len(entry.continuations) >= MAX_CONTINUATIONS:
                        continue
                    found = entry.continuations[token] = [1, document]
                    count = 1
                else:
                    count = found[0] = found[0] + 1
                    found[1] = document
                entry.total += 1
                if count > entry.best_seen:
                    entry.best, entry.best_seen = token, count

    def lookup(self, context, document: int) -> NgramMatch | None:
        """Return the longest match's best continuation, or None.

        Args:
            context: The answer's last tokens, most recent last, as a
                list of Python ints.
            document: The drafting document.
        """
        entries = self.entries
        for order in MATCH_ORDERS:
            if len(context) < order:
                continue
            entry = entries.get(tuple(context[-order:]))
            if entry is None:
                continue
            token = entry.best
            return NgramMatch(token=token, order=order,
                              fan_out=len(entry.continuations),
                              share=entry.best_seen / entry.total,
                              own=entry.continuations[token][1] == document)
        return None
