"""The positions of a document an answer can start at, and their continuations.

The scan of a document's tokens rates every word's first token
against one readout row, so the answer's first token is chosen over
every position at once. From the positions the chosen token begins,
streams give the document's own text as the model would write it,
with each run of whitespace as one space, tokenized a window at a
time as the writing reaches it.
"""

import re
from dataclasses import dataclass

import numpy as np

from quail.backends.quail.executor.extract.vocabulary import Vocabulary

NEG = float("-inf")


def document_tokens(offsets, shift: int) -> list:
    """Return (token index, (start, end) in the document) of each document token.

    Args:
        offsets: Each token's character range in the document.
        shift: The token index of the document's first token.
    """
    return [(shift + index, (int(a), int(b))) for index, (a, b) in enumerate(offsets)]


@dataclass
class DocumentTokens:
    """A document's tokens as the first-token scan reads them.

    Attributes:
        positions: Each document token's index in the prefix.
        starts: The document character each one's text starts at.
        ends: The document character each one's text ends at.
        tokens: Each one's token.
        kids: Each one's word key id in the vocabulary.
    """

    positions: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    tokens: np.ndarray
    kids: np.ndarray

    @classmethod
    def of(cls, vocabulary, prefix, tokens) -> "DocumentTokens":
        """Build a document's tokens from its prefix and document_tokens list."""
        positions = np.array([i for i, _ in tokens], dtype=np.int64)
        spans = np.array([span for _, span in tokens], dtype=np.int64).reshape(-1, 2)
        ids = np.asarray(prefix, dtype=np.int64)[positions]
        return cls(positions, spans[:, 0], spans[:, 1], ids, vocabulary.key_of[ids])


# characters a stream tokenizes ahead of its start at a time: room
# for ANSWER_MAX_TOKENS tokens of several characters each; the first
# window is short, since most streams end at their second token
STREAM_WINDOW = 512
STREAM_FIRST_WINDOW = 64


@dataclass
class Stream:
    """One start position's text, as the model would write it from there."""

    char: int
    stream: list        # the tokens from the start, as far as tokenized
    ends: list          # the document character each stream token ends at
    text: str           # the document
    exhausted: bool = False
    windows: int = 0    # windows tokenized so far

    def extend(self, vocabulary: Vocabulary) -> None:
        """Tokenize the next window of the document past the stream's end.

        A window ends before a whitespace character, so no word is cut
        and the tokens are the whole text's.
        """
        start = self.ends[-1]
        cut = start + (STREAM_FIRST_WINDOW if not self.windows else STREAM_WINDOW)
        self.windows += 1
        if cut < len(self.text):
            boundary = re.compile(r"\s").search(self.text, cut)
            cut = len(self.text) if boundary is None else boundary.start()
        window = self.text[start:cut]
        rest, ends = squeezed(window)
        if not rest:
            self.exhausted = True
            return
        ids, offsets = vocabulary.encode(rest)
        self.stream.extend(ids)
        self.ends.extend(start + ends[b - 1] for _, b in offsets)
        self.exhausted = start + len(window) >= len(self.text)

    def next(self, at: int, vocabulary: Vocabulary):
        """Return the stream's token at index at, or None past its end."""
        while at >= len(self.stream) and not self.exhausted:
            self.extend(vocabulary)
        return self.stream[at] if at < len(self.stream) else None


@dataclass
class FirstTokens:
    """Each word's first token as the model writes it, over the document.

    Attributes:
        scan: The document's tokens.
        kids: The distinct key ids among its positions, sorted.
        inverse: Each position's index into kids.
        token: Each key's likeliest first token.
        lp: Its log probability, NEG for a key the row does not rate.
        length: The word's characters that token spells.
        anchor: The scan index of each key's first position, whose
            token spells the word for the filler.
    """

    scan: DocumentTokens
    kids: np.ndarray
    inverse: np.ndarray
    token: np.ndarray
    lp: np.ndarray
    length: np.ndarray
    anchor: np.ndarray

    @property
    def rated(self) -> np.ndarray:
        return self.lp > NEG

    @property
    def best_lp(self) -> float:
        rated = self.rated
        return float(self.lp[rated].max()) if rated.any() else NEG

    @property
    def best_token(self) -> int:
        return int(self.token[int(np.argmax(self.lp))])

    @property
    def allowed(self) -> set:
        return set(self.token[self.rated].tolist())

    def keys_of(self, token: int) -> np.ndarray:
        """Return the indices into kids of the keys the token begins."""
        return np.flatnonzero(self.rated & (self.token == token))

    def positions_of(self, keys) -> np.ndarray:
        """Return the scan indices of the positions of those keys."""
        return np.flatnonzero(np.isin(self.inverse, keys))

    def __iter__(self):
        """Yield the prefix index of every rated position."""
        yield from self.scan.positions[self.positions_of(
            np.flatnonzero(self.rated))].tolist()

    def items(self, vocabulary: Vocabulary):
        """Yield (prefix index, (first token, filler)) of every rated position."""
        for k in np.flatnonzero(self.rated):
            filler = vocabulary.filler(int(self.scan.tokens[self.anchor[k]]),
                                   int(self.length[k]))
            for at in self.positions_of([k]):
                yield int(self.scan.positions[at]), (int(self.token[k]), filler)


def first_tokens(vocabulary: Vocabulary, lp, prefix, tokens,
                 scan: DocumentTokens | None = None) -> FirstTokens:
    """Return each start position's first token, as the model writes it.

    The model may write a word without its leading space, in another
    case, or split differently from the document: any vocabulary token
    whose text is a prefix of the word, up to case, can start it; the
    likeliest wins, and a shorter one is followed by the rest of the
    word as the document writes it. Positions of one word share the
    choice.

    Args:
        vocabulary: The model's vocabulary index.
        lp: The row's log probabilities by token id.
        prefix: The document's prefix tokens.
        tokens: The document tokens, as document_tokens returns them,
            or None with scan.
        scan: The document's tokens, built from prefix and tokens when
            not given.
    """
    if scan is None:
        scan = DocumentTokens.of(vocabulary, prefix, tokens)
    kids, first, inverse = np.unique(scan.kids, return_index=True,
                                     return_inverse=True)
    options = [vocabulary.options(int(kid)) for kid in kids]
    counts = np.array([len(t) for t, _ in options], dtype=np.int64)
    option_token = np.concatenate([t for t, _ in options]) if options else \
        np.zeros(0, dtype=np.int64)
    option_length = np.concatenate([j for _, j in options]) if options else \
        np.zeros(0, dtype=np.int64)
    segment = np.repeat(np.arange(len(kids), dtype=np.int64), counts)
    rated = np.full(len(vocabulary.vocab), NEG, dtype=np.float32)
    rated[np.fromiter(lp.keys(), dtype=np.int64, count=len(lp))] = \
        np.fromiter(lp.values(), dtype=np.float32, count=len(lp))
    values = rated[option_token]
    # per key, the likeliest option; the whole word wins a tie with a
    # prefix of itself
    order = np.lexsort((-option_length, -values, segment))
    _, top = np.unique(segment[order], return_index=True)
    best = order[top]
    return FirstTokens(scan=scan, kids=kids, inverse=inverse,
                  token=option_token[best], lp=values[best],
                  length=option_length[best], anchor=first)


def squeezed(text: str) -> tuple:
    """Return the text with each run of whitespace as one space, and a map back.

    The model writes a run of spaces or line breaks as one space, so
    the stream follows the squeezed text; the map gives, for each
    squeezed character, the document character it ends at.
    """
    out, ends = [], []
    run = False
    for at, char in enumerate(text):
        if char.isspace():
            if not run:
                out.append(" ")
                ends.append(at + 1)
            else:
                ends[-1] = at + 1
            run = True
        else:
            out.append(char)
            ends.append(at + 1)
            run = False
    return "".join(out), ends


def make_streams(vocabulary: Vocabulary, text: str, starts: FirstTokens,
                 first) -> list:
    """Return a stream for each start the chosen first token begins.

    A stream's tokens are the text from the start as the model writes
    it, with every run of whitespace as one space, tokenized a window
    at a time as the writing reaches it; its ends map each token back
    to the document.
    """
    scan = starts.scan
    streams = []
    for k in starts.keys_of(first):
        filler = vocabulary.filler(int(scan.tokens[starts.anchor[k]]),
                                   int(starts.length[k]))
        for at in starts.positions_of([k]):
            end = int(scan.ends[at])
            streams.append(Stream(int(scan.starts[at]), [first, *filler],
                                  [end] * (1 + len(filler)), text))
    return streams
