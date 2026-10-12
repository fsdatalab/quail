"""Copy each document's answers to its questions with the stage scheduler.

The answer is decoded after the phrase cue as a decoder writes it, but
each token is chosen among the tokens that continue some position of
the document: the span is the document's own text wherever the
model's writing leads. Each round feeds the chosen token and the
document's next tokens as a draft, and verifies them by the rows after
them. The answer ends where the model would close the quote, or where
its writing would leave the document. The document stays in KV across
the rounds; the verified tokens are appended after it.

The extractions of one pipeline run as one stage list over the
documents: a question's tail is the frame of its stages, written right
after the document in place of the tokens the question before it
appended, so a document is read once however many questions it takes.
"""

import re
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncTopLogprobs
from quail.backends.quail.executor.stages import Stage
from quail.cost.extract import ANSWER_MAX_TOKENS, DRAFT_TOKENS
from quail.execution.spans import line_span
from quail.progress import say

CUE_TOP = 1024      # vocabulary entries read at the cue and each step
NEG = float("-inf")


def same_word(token) -> str:
    """The key under which tokens differing only in a leading space or case meet."""
    if not isinstance(token, str):
        return ""
    text = token[1:] if token.startswith("Ġ") else token
    return text.lower() if text and "Ġ" not in text and "Ċ" not in text else ""


class Tables:
    """The tokenizer and the token tables one model's extractions read.

    Args:
        tokenizer: A Hugging Face fast tokenizer, which returns the
            character offsets of the tokens it produces.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        # a document token may be written without its leading space
        # or in another case after the opening quote: every
        # vocabulary token with the same text up to those can start it
        self.variants = {}
        # every token's key: its word, or the token itself without one
        self.keys = []
        key_ids = {}
        key_of = np.empty(len(self.vocab), dtype=np.int64)
        for index, token in enumerate(self.vocab):
            key = same_word(token)
            if key:
                self.variants.setdefault(key, []).append(index)
            else:
                key = index
            kid = key_ids.get(key)
            if kid is None:
                kid = len(self.keys)
                self.keys.append(key)
                key_ids[key] = kid
            key_of[index] = kid
        self.key_of = key_of
        self._options = {}      # key id -> (its first tokens, their lengths)
        self._fillers = {}      # (token, length) -> the rest of its word
        self.quote_id = tokenizer.convert_tokens_to_ids('"')
        # a decoder closes the answer with any token that opens with the
        # quote, such as `".` or `",`: together they are the stop
        self.stop_ids = tuple(index for index, token in enumerate(self.vocab)
                              if token.startswith('"'))
        self._stops = set(self.stop_ids)
        # a decoder may copy a token and close the quote in one token,
        # such as `."` or `)"`: the merged token of each token that has one
        index = {token: i for i, token in enumerate(self.vocab)}
        self.merged = {index[token[:-1]]: i for token, i in index.items()
                       if len(token) > 1 and token.endswith('"')
                       and token[:-1] in index}
        self.none_id = self.encode_ids("none")[0]

    def options(self, kid: int) -> tuple:
        """Return the tokens that can begin a key's word, and their lengths.

        A word's token, in any spelling, begins it whole; so does any
        token spelling a prefix of an ASCII word, followed by the rest
        of the word. A token without a word begins itself alone.
        """
        found = self._options.get(kid)
        if found is None:
            key = self.keys[kid]
            if isinstance(key, str) and key.isascii():
                pairs = [(t, j) for j in range(len(key), 0, -1)
                         for t in self.variants.get(key[:j], [])]
            elif isinstance(key, str):
                pairs = [(t, len(key)) for t in self.variants[key]]
            else:
                pairs = [(key, len(self.vocab[key]))]
            found = (np.array([t for t, _ in pairs], dtype=np.int64),
                     np.array([j for _, j in pairs], dtype=np.int64))
            self._options[kid] = found
        return found

    def filler(self, token: int, length: int) -> list:
        """Return the rest of a token's word after its first length characters."""
        found = self._fillers.get((token, length))
        if found is None:
            text = self.vocab[token]
            text = text[1:] if text.startswith("Ġ") else text
            found = self.encode_ids(text[length:]) if length < len(text) else []
            self._fillers[(token, length)] = found
        return found

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        enc = self.tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=False)
        return list(enc["input_ids"]), list(enc["offset_mapping"])

    def encode_batch(self, texts: list) -> list:
        """Return (token ids, character ranges) of each text, encoded together."""
        if not texts:
            return []
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return [(enc.ids, enc.offsets) for enc in backend.encode_batch(
                list(texts), add_special_tokens=False)]
        enc = self.tokenizer(list(texts), return_offsets_mapping=True,
                             add_special_tokens=False)
        return [(list(ids), list(offsets))
                for ids, offsets in zip(enc["input_ids"], enc["offset_mapping"])]

    def encode_ids(self, text: str) -> list:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def load_tables(state) -> Tables:
    """Return the loaded model's extract tables, loading its tokenizer once."""
    loaded = state.loaded_model
    if loaded.extract_tables is None:
        from transformers import AutoTokenizer

        from quail.backends.quail.executor.model import resolve_model_path

        spec = loaded.model_spec
        path = resolve_model_path(spec.hf_name, spec.revision)
        say(f"loading the tokenizer of {spec.name} for AI.EXTRACT")
        loaded.extract_tables = Tables(AutoTokenizer.from_pretrained(path))
    return loaded.extract_tables


def document_tokens(offsets, shift: int) -> list:
    """Return (token index, (start, end) in the document) of each document token.

    Args:
        offsets: Each token's character range in the document.
        shift: The token index of the document's first token.
    """
    return [(shift + index, (int(a), int(b))) for index, (a, b) in enumerate(offsets)]


@dataclass
class Scan:
    """A document's tokens as the first-token scan reads them.

    Attributes:
        positions: Each document token's index in the prefix.
        starts: The document character each one's text starts at.
        ends: The document character each one's text ends at.
        tokens: Each one's token.
        kids: Each one's key id in the tables.
    """

    positions: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    tokens: np.ndarray
    kids: np.ndarray

    @classmethod
    def of(cls, tables, prefix, tokens) -> "Scan":
        """Build the scan of a document from its prefix and document tokens."""
        positions = np.array([i for i, _ in tokens], dtype=np.int64)
        spans = np.array([span for _, span in tokens], dtype=np.int64).reshape(-1, 2)
        ids = np.asarray(prefix, dtype=np.int64)[positions]
        return cls(positions, spans[:, 0], spans[:, 1], ids, tables.key_of[ids])


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

    def extend(self, tables: Tables) -> None:
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
        ids, offsets = tables.encode(rest)
        self.stream.extend(ids)
        self.ends.extend(start + ends[b - 1] for _, b in offsets)
        self.exhausted = start + len(window) >= len(self.text)

    def next(self, at: int, tables: Tables):
        """Return the stream's token at index at, or None past its end."""
        while at >= len(self.stream) and not self.exhausted:
            self.extend(tables)
        return self.stream[at] if at < len(self.stream) else None


@dataclass
class Question:
    """One document's steps and result for one question."""

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
class Extraction:
    """One document's prompt and its answer to each question."""

    index: int
    text: str
    prefix: list
    scan: Scan          # its document tokens
    key: tuple
    questions: list     # one Question per spec


@dataclass
class Starts:
    """Each word's first token as the model writes it, over the document.

    Attributes:
        scan: The document's scan.
        kids: The distinct key ids among its positions, sorted.
        inverse: Each position's index into kids.
        token: Each key's likeliest first token.
        lp: Its log probability, NEG for a key the row does not rate.
        length: The word's characters that token spells.
        anchor: The scan index of each key's first position, whose
            token spells the word for the filler.
    """

    scan: Scan
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

    def items(self, tables: Tables):
        """Yield (prefix index, (first token, filler)) of every rated position."""
        for k in np.flatnonzero(self.rated):
            filler = tables.filler(int(self.scan.tokens[self.anchor[k]]),
                                   int(self.length[k]))
            for at in self.positions_of([k]):
                yield int(self.scan.positions[at]), (int(self.token[k]), filler)


def first_tokens(tables: Tables, lp, prefix, tokens,
                 scan: Scan | None = None) -> Starts:
    """Return each start position's first token, as the model writes it.

    The model may write a word without its leading space, in another
    case, or split differently from the document: any vocabulary token
    whose text is a prefix of the word, up to case, can start it; the
    likeliest wins, and a shorter one is followed by the rest of the
    word as the document writes it. Positions of one word share the
    choice.

    Args:
        tables: The model's token tables.
        lp: The row's log probabilities by token id.
        prefix: The document's prefix tokens.
        tokens: The document tokens, as document_tokens returns them,
            or None with a scan.
        scan: The document's scan, built from prefix and tokens when
            not given.
    """
    if scan is None:
        scan = Scan.of(tables, prefix, tokens)
    kids, first, inverse = np.unique(scan.kids, return_index=True,
                                     return_inverse=True)
    options = [tables.options(int(kid)) for kid in kids]
    counts = np.array([len(t) for t, _ in options], dtype=np.int64)
    option_token = np.concatenate([t for t, _ in options]) if options else \
        np.zeros(0, dtype=np.int64)
    option_length = np.concatenate([j for _, j in options]) if options else \
        np.zeros(0, dtype=np.int64)
    segment = np.repeat(np.arange(len(kids), dtype=np.int64), counts)
    rated = np.full(len(tables.vocab), NEG, dtype=np.float32)
    rated[np.fromiter(lp.keys(), dtype=np.int64, count=len(lp))] = \
        np.fromiter(lp.values(), dtype=np.float32, count=len(lp))
    values = rated[option_token]
    # per key, the likeliest option; the whole word wins a tie with a
    # prefix of itself
    order = np.lexsort((-option_length, -values, segment))
    _, top = np.unique(segment[order], return_index=True)
    best = order[top]
    return Starts(scan=scan, kids=kids, inverse=inverse,
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


def make_streams(tables: Tables, text: str, starts: Starts, first) -> list:
    """Return a stream for each start the chosen first token begins.

    A stream's tokens are the text from the start as the model writes
    it, with every run of whitespace as one space, tokenized a window
    at a time as the writing reaches it; its ends map each token back
    to the document.
    """
    scan = starts.scan
    streams = []
    for k in starts.keys_of(first):
        filler = tables.filler(int(scan.tokens[starts.anchor[k]]),
                               int(starts.length[k]))
        for at in starts.positions_of([k]):
            end = int(scan.ends[at])
            streams.append(Stream(int(scan.starts[at]), [first, *filler],
                                  [end] * (1 + len(filler)), text))
    return streams


class ExtractStages:
    """The stages of one pipeline's extractions and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The ExtractSpecs, in order; they share a table and a
            prompt head.
        tables: The model's token tables.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readout: Optional readout in place of the model's. Its extras
            are the none token, then the stop tokens as one group.
        shared_head: The query's shared preamble, under which a filter
            writes the documents' KV; None when the query has none.

    Raises:
        ValueError: The specs differ in their table or head.
    """

    def __init__(self, state, specs, tables: Tables, document_ids, documents,
                 texts, readout=None, shared_head=None):
        self.specs = list(specs)
        first = self.specs[0]
        if any(spec.alias != first.alias for spec in self.specs):
            raise ValueError("a pipeline's extractions share a table")
        parts = [spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError("a pipeline's extractions share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        self.shared_head = (None if shared_head is None
                            else [int(t) for t in shared_head])
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError("an extraction's tail ends in its cue after the question")
        self.tables = tables
        shown = [text.as_py() if hasattr(text, "as_py") else str(text)
                 for text in (texts[document] for document in document_ids)]
        encoded = tables.encode_batch(shown)
        self.docs = [self._prepare(index, document, text, *ids,
                                   None if documents is None
                                   else documents[document])
                     for index, (document, text, ids)
                     in enumerate(zip(document_ids, shown, encoded))]
        self.by_key = {doc.key: doc for doc in self.docs}
        self.prefixes = [doc.prefix for doc in self.docs]
        self.keys = [doc.key for doc in self.docs]
        self.readout = self._readout(state) if readout is None else readout
        self.stages = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.stages.extend(self._question_stages(q, spec.name, tail))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, then the answer's rounds.

        The tail but its last token is the frame of every stage, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail, the cue, is the stage's suffix, whose
        row chooses the answer's first token.
        """
        frame, last = tail[:-1], tail[-1]
        stages = [Stage(
            suffixes=[[last]], frame=frame, rewind=True,
            readout=self.readout, decide=partial(self._tail_decide, q),
            single=True, append=True, label=f"{name} tail")]
        stages.extend(
            Stage(suffixes=partial(self._step_suffix, q),
                  suffix_tokens=1 + DRAFT_TOKENS, frame=frame,
                  readout=self.readout, decide=partial(self._step_decide, q),
                  discard=partial(self._step_discard, q), read_all_rows=True,
                  single=True, append=True, label=f"{name} round {t}")
            for t in range(1, ANSWER_MAX_TOKENS + 1))
        return stages

    @property
    def appended_tokens(self) -> int:
        """The most tokens a question leaves in KV after its frame.

        The cue, and the answer's tokens with the last round's draft
        past them.
        """
        return 1 + ANSWER_MAX_TOKENS + DRAFT_TOKENS

    def _readout(self, state):
        """The readout of every row: none, the stop, then the top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        tables = self.tables
        return AsyncTopLogprobs(torch, torch.nn.functional, head, CUE_TOP,
                                [tables.none_id, tables.stop_ids], ragged=True)

    def _prepare(self, index, document, text, ids, offsets, stored) -> Extraction:
        """Build a document's prefix: the head and its text.

        The key names the document under the extraction's head, so a
        later extraction of the same table reads the KV this one
        leaves resident. A document shown under the query's shared
        head, whose tokens are the table's, keeps the table's key, so
        the KV a filter wrote is read.
        """
        tables = self.tables
        alias = self.specs[0].alias
        key = (("extract", alias), int(document))
        shared = (self.head == self.shared_head if self.shared_head is not None
                  else not self.head)
        if shared and stored is not None and [int(t) for t in stored] == ids:
            key = (alias, int(document))
        prefix = self.head + ids
        bounds = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        tokens = np.asarray(ids, dtype=np.int64)
        scan = Scan(np.arange(len(ids), dtype=np.int64) + len(self.head),
                    bounds[:, 0], bounds[:, 1], tokens, tables.key_of[tokens])
        return Extraction(index=index, text=text, prefix=prefix, scan=scan,
                          key=key, questions=[Question() for _ in self.specs])

    def _question(self, q: int, anchor: int) -> tuple:
        doc = self.docs[anchor]
        return doc, doc.questions[q]

    # ---- the tail, then the answer, a round of tokens at a time ------------

    def _tail_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        doc, qs = self._question(q, anchor)
        qs.asked = True
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = q > 0 and self.tails[q] == self.tails[q - 1]
        qs.fed_tokens += 1 if same else len(self.tails[q])
        return self._verify(doc, q, self._rows(row), [])

    def _rows(self, row) -> np.ndarray:
        """Return a stage's answer as its rows of readout values."""
        values = np.asarray(row)
        if values.dtype == object:
            values = np.asarray(values.reshape(-1)[0], dtype=np.float32)
        return values.reshape(-1, self.readout.width)

    def _step_suffix(self, q, key):
        """Feed the chosen token, then the document's next tokens as a draft.

        The draft follows the first position the writing fits; the
        rows after its tokens verify it, so a round writes as many
        tokens as the model would have chosen one at a time.
        """
        qs = self.by_key[key].questions[q]
        if qs.done:
            return Stage.SKIP
        at = len(qs.written)
        stream = min(qs.live, key=lambda s: s.char)
        draft = []
        for i in range(DRAFT_TOKENS):
            token = stream.next(at + i, self.tables)
            if token is None:
                break
            draft.append(token)
        qs.fed = [qs.written[-1], *draft]
        return [qs.fed]

    def _step_discard(self, q, key):
        return self.by_key[key].questions[q].discard

    def _step_decide(self, q, anchor, row):
        """Verify the round's draft by the row after each of its tokens."""
        doc, qs = self._question(q, anchor)
        qs.fed_tokens += len(qs.fed)
        return self._verify(doc, q, self._rows(row), qs.fed)

    def _lp(self, values) -> tuple:
        """Return a row's log probabilities by token id, and none's and the stop's."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        k = self.readout.k
        lp = dict(zip(values[-2 * k:-k].astype(np.int64).tolist(),
                      values[-k:].tolist()))
        return lp, float(values[0]), float(values[1])

    def _verify(self, doc, q, rows, fed) -> bool:
        """Choose the answer's tokens from a round's rows, in order.

        Row i follows token i of the round's fed tokens, so the token
        chosen from it is written and, when it is not the token fed
        next, the rest of the round is discarded: the next round feeds
        the chosen token in its place.
        """
        qs = doc.questions[q]
        qs.steps += 1
        qs.discard = 0
        for i, values in enumerate(rows):
            token = self._choose(doc, q, values)
            if qs.done:
                return True
            if i + 1 < len(fed) and token != fed[i + 1]:
                qs.discard = len(fed) - (i + 1)
                return True
        return True

    def _choose(self, doc, q, values):
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
        qs = doc.questions[q]
        tables = self.tables
        lp, none_lp, stop_lp = self._lp(values)
        if not qs.written:
            starts = first_tokens(tables, lp, doc.prefix, None, scan=doc.scan)
            qs.none_lp = none_lp
            qs.best_lp = starts.best_lp
            qs.none_wins = (max(none_lp, stop_lp) > qs.best_lp
                            or self._leaves(tables, lp, starts.allowed, qs.best_lp,
                                            stop_lp))
            if qs.none_wins:
                self._answer(doc, q)
                return None
            token = starts.best_token
            qs.live = make_streams(tables, doc.text, starts, token)
        else:
            at = len(qs.written)
            following = {token for token in (s.next(at, tables) for s in qs.live)
                         if token is not None}
            best = max(following, key=lambda t: lp.get(t, NEG), default=None)
            best_lp = NEG if best is None else lp.get(best, NEG)
            merged = {t: tables.merged[t] for t in following
                      if t in tables.merged and tables.merged[t] in lp}
            joined = max(merged, key=lambda t: lp[merged[t]], default=None)
            if joined is not None and lp[merged[joined]] > max(best_lp, stop_lp):
                qs.written.append(joined)
                qs.live = [s for s in qs.live if s.next(at, tables) == joined]
                self._answer(doc, q)
                return None
            if (best is None or stop_lp >= best_lp
                    or self._leaves(tables, lp, following | set(merged.values()),
                                    best_lp, stop_lp)):
                self._answer(doc, q)
                return None
            token = best
            qs.live = [s for s in qs.live if s.next(at, tables) == token]
        qs.written.append(token)
        if len(qs.written) >= ANSWER_MAX_TOKENS:
            self._answer(doc, q)
        return token

    @staticmethod
    def _leaves(tables, lp, allowed, best_lp, stop_lp) -> bool:
        """Return whether the model's likeliest token leaves the document."""
        top = max(lp, key=lp.get)
        return (lp[top] > max(best_lp, stop_lp) and top not in allowed
                and top not in tables._stops and top != tables.none_id)

    def _answer(self, doc, q):
        """Set the question's answer for the document: its span, or none.

        Among the streams the writing fits, the first position of the
        document wins.
        """
        qs = doc.questions[q]
        qs.done = True
        if qs.none_wins or not qs.written or not qs.live:
            qs.answer, qs.span = None, None
            return
        stream = min(qs.live, key=lambda s: s.char)
        start, end = stream.char, stream.ends[len(qs.written) - 1]
        text = doc.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if not self.specs[q].trim:
            start, end = line_span(text, (start, end))
        qs.answer, qs.span = text[start:end], (start, end)

    def finish(self) -> None:
        """Answer every question the steps left open."""
        for doc in self.docs:
            for q, qs in enumerate(doc.questions):
                if not qs.done:
                    self._answer(doc, q)

    def answers(self, q: int, kept=None) -> tuple:
        """Return each document's answer and span for question q.

        Args:
            q: The question's index.
            kept: The indices of the documents to return, in order;
                None returns every document.
        """
        docs = self.docs if kept is None else [self.docs[i] for i in kept]
        return ([doc.questions[q].answer for doc in docs],
                [doc.questions[q].span for doc in docs])

    def asked(self) -> list:
        """Return the indices of the documents the questions were asked of."""
        return [index for index, doc in enumerate(self.docs)
                if doc.questions[0].asked]

    def counts(self, q: int) -> dict:
        """Return how many documents answered none and how much was fed."""
        states = [doc.questions[q] for doc in self.docs]
        return {
            "none": sum(qs.answer is None for qs in states),
            "steps": sum(qs.steps for qs in states),
            "answer_tokens": sum(len(qs.written) for qs in states),
            "suffix_tokens": sum(qs.fed_tokens for qs in states),
        }
