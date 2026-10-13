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
from quail.physical import ExtractSpec

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
    candidates: "CandidateIndex | None" = None
    normalized: "SqueezedText | None" = None

    @classmethod
    def of(cls, vocabulary, prefix, tokens) -> "DocumentTokens":
        """Build a document's tokens from its prefix and document_tokens list."""
        positions = np.array([i for i, _ in tokens], dtype=np.int64)
        spans = np.array([span for _, span in tokens], dtype=np.int64).reshape(-1, 2)
        ids = np.asarray(prefix, dtype=np.int64)[positions]
        return cls(positions, spans[:, 0], spans[:, 1], ids, vocabulary.key_of[ids])


class SqueezedText:
    """A document with single spaces and a map to original character ends."""

    def __init__(self, text: str):
        self.text = re.sub(r"\s+", " ", text)
        self.ends = np.arange(1, len(self.text) + 1, dtype=np.int64)
        runs = list(re.finditer(r"\s{2,}", text))
        if runs:
            removed = np.array([run.end() - run.start() - 1 for run in runs])
            starts = np.array([run.start() for run in runs])
            positions = starts - (np.cumsum(removed) - removed)
            shifts = np.zeros(len(self.text), dtype=np.int64)
            shifts[positions] = removed
            self.ends += np.cumsum(shifts)

    def window(self, start: int, end: int) -> tuple:
        """Return a normalized original-text interval and its character ends."""
        if start >= end:
            return "", self.ends[:0]
        left = int(np.searchsorted(self.ends, start, side="right"))
        right = int(np.searchsorted(self.ends, end - 1, side="right")) + 1
        ends = self.ends[left:right]
        if len(ends) and ends[-1] > end:
            # A window may end inside a whitespace run.
            ends = ends.copy()
            ends[-1] = end
        return self.text[left:right], ends


@dataclass
class CandidateIndex:
    """The document's candidate starts, shared across extraction questions."""

    kids: np.ndarray
    anchor: np.ndarray
    inverse: np.ndarray
    token: np.ndarray
    length: np.ndarray
    segment: np.ndarray
    offsets: np.ndarray
    position_order: np.ndarray
    position_offsets: np.ndarray

    @classmethod
    def of(cls, scan: DocumentTokens, vocabulary: Vocabulary) -> "CandidateIndex":
        kids, first, inverse, occurrences = np.unique(
            scan.kids, return_index=True, return_inverse=True, return_counts=True)
        options = [vocabulary.options(int(kid)) for kid in kids]
        counts = np.array([len(t) for t, _ in options], dtype=np.int64)
        token = (np.concatenate([t for t, _ in options]) if options
                 else np.empty(0, dtype=np.int64))
        length = (np.concatenate([j for _, j in options]) if options
                  else np.empty(0, dtype=np.int64))
        return cls(kids, first, inverse, token, length,
                   np.repeat(np.arange(len(kids)), counts),
                   np.cumsum(counts) - counts,
                   np.argsort(inverse, kind="stable"),
                   np.concatenate(([0], np.cumsum(occurrences))))


@dataclass
class Stream:
    """One start position's text, as the model would write it from there."""

    char: int
    stream: list        # the tokens from the start, as far as tokenized
    ends: list          # the document character each stream token ends at
    text: str           # the document
    exhausted: bool = False
    windows: int = 0    # windows tokenized so far
    normalized: SqueezedText | None = None
    first_window_chars: int = ExtractSpec.first_window_chars
    window_chars: int = ExtractSpec.window_chars

    def extend(self, vocabulary: Vocabulary) -> None:
        """Tokenize the next window of the document past the stream's end.

        A window ends before a whitespace character, so no word is cut
        and the tokens are the whole text's.
        """
        extend_streams([self], vocabulary)

    def window(self) -> tuple:
        """Return the next squeezed window and its document character ends."""
        start = self.ends[-1]
        cut = start + (self.first_window_chars if not self.windows
                       else self.window_chars)
        self.windows += 1
        if cut < len(self.text):
            boundary = re.compile(r"\s").search(self.text, cut)
            cut = len(self.text) if boundary is None else boundary.start()
        cut = min(cut, len(self.text))
        if self.normalized is None:
            self.normalized = SqueezedText(self.text)
        rest, ends = self.normalized.window(start, cut)
        self.exhausted = cut == len(self.text)
        if not rest:
            self.exhausted = True
        return rest, ends

    def append_window(self, window, encoded) -> None:
        """Append a tokenized window and its document character ends."""
        _, ends = window
        ids, offsets = encoded
        if isinstance(ids, np.ndarray):
            self.stream.extend(ids.tolist())
            self.ends.extend(ends[offsets[:, 1] - 1].tolist())
        else:
            self.stream.extend(ids)
            self.ends.extend(int(ends[b - 1]) for _, b in offsets)

    def next(self, at: int, vocabulary: Vocabulary):
        """Return the stream's token at index at, or None past its end."""
        while at >= len(self.stream) and not self.exhausted:
            self.extend(vocabulary)
        return self.stream[at] if at < len(self.stream) else None


def extend_streams(streams: list, vocabulary: Vocabulary) -> None:
    """Tokenize one window for each stream together, preserving character offsets."""
    pending = []
    for stream in streams:
        window = stream.window()
        if window[0]:
            pending.append((stream, window))
    encoded = vocabulary.encode_batch([window[0] for _, window in pending])
    for (stream, window), tokens in zip(pending, encoded):
        stream.append_window(window, tokens)


def ensure_tokens(streams: list, at: int, vocabulary: Vocabulary) -> None:
    """Tokenize windows together until each live stream reaches the requested token."""
    while pending := [stream for stream in streams
                      if at >= len(stream.stream) and not stream.exhausted]:
        extend_streams(pending, vocabulary)


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

    def allows(self, token: int) -> bool:
        """Return whether the token can begin a rated document word."""
        return bool(np.any(self.rated & (self.token == token)))

    def keys_of(self, token: int) -> np.ndarray:
        """Return the indices into kids of the keys the token begins."""
        return np.flatnonzero(self.rated & (self.token == token))

    def positions_of(self, keys) -> np.ndarray:
        """Return the scan indices of the positions of those keys."""
        if len(keys) == 1:
            index = self.scan.candidates
            start, end = index.position_offsets[keys[0]:keys[0] + 2]
            return index.position_order[start:end]
        selected = np.zeros(len(self.kids), dtype=bool)
        selected[keys] = True
        return np.flatnonzero(selected[self.inverse])

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
        lp: The row's log probabilities indexed by token id.
        prefix: The document's prefix tokens.
        tokens: The document tokens, as document_tokens returns them,
            or None with scan.
        scan: The document's tokens, built from prefix and tokens when
            not given.
    """
    if scan is None:
        scan = DocumentTokens.of(vocabulary, prefix, tokens)
    if scan.candidates is None:
        scan.candidates = CandidateIndex.of(scan, vocabulary)
    index = scan.candidates
    values = lp[index.token]
    maxima = np.maximum.reduceat(values, index.offsets)
    # Vocabulary.options orders equal-score spellings longest first.
    eligible = np.where(values == maxima[index.segment],
                        np.arange(len(values)), len(values))
    best = np.minimum.reduceat(eligible, index.offsets)
    return FirstTokens(scan=scan, kids=index.kids, inverse=index.inverse,
                       token=index.token[best], lp=values[best],
                       length=index.length[best], anchor=index.anchor)


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
                 first, *, first_window_chars: int = ExtractSpec.first_window_chars,
                 window_chars: int = ExtractSpec.window_chars) -> list:
    """Return a stream for each start the chosen first token begins.

    A stream's tokens are the text from the start as the model writes
    it, with every run of whitespace as one space, tokenized a window
    at a time as the writing reaches it; its ends map each token back
    to the document.
    """
    scan = starts.scan
    if scan.normalized is None:
        scan.normalized = SqueezedText(text)
    streams = []
    for k in starts.keys_of(first):
        filler = vocabulary.filler(int(scan.tokens[starts.anchor[k]]),
                                   int(starts.length[k]))
        for at in starts.positions_of([k]):
            end = int(scan.ends[at])
            streams.append(Stream(int(scan.starts[at]), [first, *filler],
                                  [end] * (1 + len(filler)), text,
                                  normalized=scan.normalized,
                                  first_window_chars=first_window_chars,
                                  window_chars=window_chars))
    return streams
