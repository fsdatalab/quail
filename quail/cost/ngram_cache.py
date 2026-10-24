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


class NgramCache:
    """Continuation counts for one query run, shared across documents."""

    def __init__(self):
        # key -> {continuation token: [count, last inserting document]}
        self.entries = {}

    def add(self, tokens, document: int, start: int = 0) -> None:
        """Insert the continuations the tokens show from start on.

        Args:
            tokens: One document's committed answer tokens so far.
            document: The document that wrote them.
            start: The first position not inserted yet.
        """
        for at in range(max(start, MATCH_ORDERS[-1]), len(tokens)):
            token = int(tokens[at])
            for order in MATCH_ORDERS:
                if at < order:
                    continue
                key = tuple(int(t) for t in tokens[at - order:at])
                found = self.entries.setdefault(key, {})
                entry = found.get(token)
                if entry is None:
                    if len(found) >= MAX_CONTINUATIONS:
                        continue
                    found[token] = [1, document]
                else:
                    entry[0] += 1
                    entry[1] = document

    def lookup(self, context, document: int) -> NgramMatch | None:
        """Return the longest match's best continuation, or None.

        Args:
            context: The answer's last tokens, most recent last.
            document: The drafting document.
        """
        for order in MATCH_ORDERS:
            if len(context) < order:
                continue
            key = tuple(int(t) for t in context[-order:])
            found = self.entries.get(key)
            if not found:
                continue
            token, (count, last) = max(found.items(), key=lambda item: item[1][0])
            total = sum(entry[0] for entry in found.values())
            return NgramMatch(token=token, order=order, fan_out=len(found),
                              share=count / total, own=last == document)
        return None
