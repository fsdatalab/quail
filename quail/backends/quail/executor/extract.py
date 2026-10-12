"""Copy each document's answers to its questions with the stage scheduler.

With locating on, a document is shown as numbered lines and the model
writes the lines that hold the answer. The answer is then decoded one
token at a time after the phrase cue, as a decoder writes it, but
each token is chosen among the tokens that continue some position of
the document inside those lines: the span is the document's own text
wherever the model's writing leads. With locating off, the document is
shown plain and the positions are the whole of it. The answer ends
where the model would close the quote. The document stays in KV
across the steps; each chosen token is appended after it.

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
from quail.logical.prompts import EXTRACT_PHRASE_CUE
from quail.progress import say

CUE_TOP = 1024      # vocabulary entries read at the cue and each step
LINE_MAX_TOKENS = 8     # tokens the lines answer may take
NEG = float("-inf")


def document_lines(text: str) -> list:
    """Return the character ranges of the text's non-empty lines.

    Blank lines get no number, so the model does not have to count
    them; the text of each line is unchanged.
    """
    out, start = [], 0
    for line in text.split("\n"):
        end = start + len(line)
        if line.strip():
            out.append((start, end))
        start = end + 1
    return out


def numbered_text(text: str, lines) -> tuple:
    """Return the lines numbered as ``cat -n`` would, and where each line's text is.

    Returns:
        The numbered text, and (offset in it, offset in text, length)
        of each line.
    """
    numbered, pieces, at = [], [], 0
    for index, (a, b) in enumerate(lines):
        head = f"{index + 1}: "
        pieces.append((at + len(head), a, b - a))
        numbered.append(head + text[a:b])
        at += len(head) + (b - a) + 1
    return "\n".join(numbered), pieces


def parse_lines(answer: str, count: int):
    """Return the 1-based line range an answer names, within the document."""
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", answer)
    if not match:
        return None
    a, b = int(match.group(1)), int(match.group(2) or match.group(1))
    if not 1 <= a <= b <= count:
        return None
    return a, b


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
        for index, token in enumerate(self.vocab):
            key = same_word(token)
            if key:
                self.variants.setdefault(key, []).append(index)
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
        digits = [self.encode_ids(digit)[0] for digit in "0123456789"]
        dash, newline = self.encode_ids("-")[0], self.encode_ids("\n")[0]
        ends = {tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>")}
        ends = sorted(token for token in ends
                      if token is not None and token >= 0
                      and token != tokenizer.unk_token_id)
        # the lines answer is decoded over these alone: the digits,
        # the dash, and the tokens that end it
        self.range_targets = digits + [dash, newline] + ends
        self.range_text = {**{token: digit for token, digit
                              in zip(digits, "0123456789")}, dash: "-"}
        self.begins_cue = self.encode_ids("\n" + EXTRACT_PHRASE_CUE)
        self._options = {}      # word key -> its (first token, prefix length)s

    def first_token(self, key, lp, token) -> tuple | None:
        """Return the likeliest token that begins a word, and the word's rest.

        Args:
            key: The word's key from same_word, or its token when it
                has none.
            lp: The row's log probabilities by token id.
            token: The document's token for the word.
        """
        options = self._options.get(key)
        if options is None:
            if isinstance(key, str) and key.isascii():
                options = [(t, j) for j in range(len(key), 0, -1)
                           for t in self.variants.get(key[:j], [])]
            elif isinstance(key, str):
                options = [(t, len(key)) for t in self.variants.get(key, [token])]
            else:
                options = [(token, 1)]
            self._options[key] = options
        rated = [(t, j) for t, j in options if t in lp]
        if not rated:
            return None
        # the whole word first, so it wins a tie with a prefix of itself
        first, j = max(rated, key=lambda o: (lp[o[0]], o[1]))
        filler = []
        if isinstance(key, str) and j < len(key):
            text = self.vocab[token]
            text = text[1:] if text.startswith("Ġ") else text
            filler = self.encode_ids(text[j:])
        return first, filler

    def encode(self, text: str) -> tuple:
        """Return the token ids of a text and each token's character range."""
        enc = self.tokenizer(text, return_offsets_mapping=True,
                             add_special_tokens=False)
        return list(enc["input_ids"]), list(enc["offset_mapping"])

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


def document_tokens(ids, offsets, pieces, shift: int) -> list:
    """Return (token index, (start, end) in the document) of each document token.

    A token overlapping one of the pieces is clipped to it: a numbered
    line's first token carries the space after its number, and its
    last may carry the line break.

    Args:
        ids: The tokens of a text showing the document.
        offsets: Each token's character range in that text.
        pieces: (offset in the text, offset in the document, length)
            of each stretch of document text the text shows verbatim.
        shift: The token index of the text's first token.
    """
    doc = []
    for index, (a, b) in enumerate(offsets):
        for off, ctx, length in pieces:
            lo, hi = max(a, off), min(b, off + length)
            if hi > lo:
                doc.append((shift + index, (ctx + lo - off, ctx + hi - off)))
                break
    return doc


# characters a stream tokenizes ahead of its start at a time: room
# for ANSWER_MAX_TOKENS tokens of several characters each
STREAM_WINDOW = 512


@dataclass
class Stream:
    """One start position's text, as the model would write it from there."""

    char: int
    stream: list        # the tokens from the start, as far as tokenized
    ends: list          # the document character each stream token ends at
    text: str           # the document
    exhausted: bool = False

    def extend(self, tables: Tables) -> None:
        """Tokenize the next window of the document past the stream's end.

        A window ends before a whitespace character, so no word is cut
        and the tokens are the whole text's.
        """
        start = self.ends[-1]
        cut = start + STREAM_WINDOW
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

    range_tokens: list = field(default_factory=list)
    range_done: bool = False
    range: tuple | None = None
    region: tuple | None = None
    path: str = "scored"
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
    lines: list
    numbered: bool
    prefix: list
    tokens: list        # (prefix index, (start, end)) per document token
    key: tuple
    questions: list     # one Question per spec


def first_tokens(tables: Tables, lp, prefix, tokens, region) -> dict:
    """Return each start position's first token, as the model writes it.

    The model may write a word without its leading space, in another
    case, or split differently from the document: any vocabulary token
    whose text is a prefix of the word, up to case, can start it; the
    likeliest wins, and a shorter one is followed by the rest of the
    word as the document writes it. Positions of one word share the
    choice.

    Returns:
        Position index -> (first token, filler tokens) for every
        document token in the region whose first token the row rates.
    """
    by_key = {}
    for i, span in tokens:
        if region is not None and not region[0] <= span[0] < region[1]:
            continue
        token = prefix[i]
        key = same_word(tables.vocab[token]) or token
        by_key.setdefault(key, []).append(i)
    starts = {}
    for key, positions in by_key.items():
        choice = tables.first_token(key, lp, prefix[positions[0]])
        if choice is not None:
            for i in positions:
                starts[i] = choice
    return starts


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


def make_streams(tables: Tables, text: str, tokens, starts, first) -> list:
    """Return a stream for each start the chosen first token begins.

    A stream's tokens are the text from the start as the model writes
    it, with every run of whitespace as one space, tokenized a window
    at a time; its ends map each token back to the document.
    """
    spans = dict(tokens)
    streams = []
    for i, (token, filler) in starts.items():
        if token != first:
            continue
        char, end = spans[i]
        stream = Stream(char, [token, *filler], [end] * (1 + len(filler)), text)
        stream.extend(tables)
        streams.append(stream)
    return streams


class ExtractStages:
    """The stages of one pipeline's extractions and the answers they produce.

    Args:
        state: The loaded model and current query state.
        specs: The ExtractSpecs, in order; they share a table, a
            locating choice, and a prompt head.
        tables: The model's token tables.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readout: Optional readout in place of the model's. Its extras
            are the none token, the stop tokens as one group, then the
            lines answer's targets.

    Raises:
        ValueError: The specs differ in their table, locating, or head.
    """

    def __init__(self, state, specs, tables: Tables, document_ids, documents,
                 texts, readout=None):
        self.specs = list(specs)
        first = self.specs[0]
        if any(spec.alias != first.alias or spec.locate != first.locate
               for spec in self.specs):
            raise ValueError("a pipeline's extractions share a table and locating")
        self.numbered = first.locate
        parts = [spec.numbered_token_parts if self.numbered
                 else spec.prompt_token_parts for spec in self.specs]
        self.head = list(parts[0][0])
        if any(list(head) != self.head for head, _ in parts):
            raise ValueError("a pipeline's extractions share a prompt head")
        self.tails = [list(tail) for _, tail in parts]
        if any(len(tail) < 2 for tail in self.tails):
            raise ValueError("an extraction's tail ends in its cue after the question")
        self.tables = tables
        self.docs = [self._prepare(index, document, texts[document],
                                   None if documents is None
                                   else documents[document])
                     for index, document in enumerate(document_ids)]
        self.by_key = {doc.key: doc for doc in self.docs}
        self.prefixes = [doc.prefix for doc in self.docs]
        self.keys = [doc.key for doc in self.docs]
        self.readout = self._readout(state) if readout is None else readout
        self.stages = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.stages.extend(self._question_stages(q, spec.name, tail))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, lines, cue, and answer tokens.

        The tail but its last token is the frame of every stage, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail is the stage's suffix, whose row is
        read: the lines answer's first token for a numbered document,
        the answer's first token for a plain one.
        """
        frame, last = tail[:-1], tail[-1]
        stages = [Stage(
            suffixes=[[last]], frame=frame, rewind=True,
            readout=self.readout, decide=partial(self._tail_decide, q),
            single=True, append=True, label=f"{name} tail")]
        stages.extend(
            Stage(suffixes=partial(self._range_suffix, q), suffix_tokens=1,
                  frame=frame, readout=self.readout,
                  decide=partial(self._range_decide, q),
                  single=True, append=True, label=f"{name} lines {r}")
            for r in range(1, LINE_MAX_TOKENS))
        stages.append(Stage(
            suffixes=partial(self._cue_suffix, q),
            suffix_tokens=len(self.tables.begins_cue), frame=frame,
            readout=self.readout, decide=partial(self._cue_decide, q),
            single=True, append=True, label=f"{name} start"))
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

        The tail's last token, the lines answer, the cue, and the
        answer's tokens with the last round's draft past them.
        """
        return (1 + LINE_MAX_TOKENS + len(self.tables.begins_cue)
                + ANSWER_MAX_TOKENS + DRAFT_TOKENS)

    def _readout(self, state):
        """The readout of every row: none, the stop, the lines targets, top tokens."""
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        tables = self.tables
        return AsyncTopLogprobs(torch, torch.nn.functional, head, CUE_TOP,
                                [tables.none_id, tables.stop_ids,
                                 *tables.range_targets], ragged=True)

    def _prepare(self, index, document, text, stored) -> Extraction:
        """Build a document's prefix: the head and its plain or numbered text.

        The key names the document and its rendering, so a later
        extraction of the same table in the same rendering reads the
        KV this one leaves resident. A plain document shown without a
        head, whose tokens are the table's, keeps the table's key, so
        KV a filter left resident is read.
        """
        tables = self.tables
        text = text.as_py() if hasattr(text, "as_py") else str(text)
        lines = document_lines(text)
        if self.numbered:
            body, pieces = numbered_text(text, lines)
        else:
            body, pieces = text, [(0, 0, len(text))]
        ids, offsets = tables.encode(body)
        alias = self.specs[0].alias
        rendering = "numbered" if self.numbered else "plain"
        key = (("extract", alias, rendering), int(document))
        if (not self.numbered and not self.head and stored is not None
                and [int(t) for t in stored] == ids):
            key = (alias, int(document))
        prefix = self.head + ids
        tokens = document_tokens(ids, offsets, pieces, len(self.head))
        path = "located" if self.numbered else "scored"
        return Extraction(index=index, text=text, lines=lines,
                          numbered=self.numbered, prefix=prefix, tokens=tokens,
                          key=key,
                          questions=[Question(path=path) for _ in self.specs])

    def _question(self, q: int, anchor: int) -> tuple:
        doc = self.docs[anchor]
        return doc, doc.questions[q]

    # ---- the tail, then the lines range -----------------------------------

    def _tail_decide(self, q, anchor, row):
        doc, qs = self._question(q, anchor)
        # a tail equal to the question's before it is in KV already:
        # only its cue token is fed
        same = q > 0 and self.tails[q] == self.tails[q - 1]
        qs.fed_tokens += 1 if same else len(self.tails[q])
        if doc.numbered:
            return self._range_decide(q, anchor, row, first=True)
        return self._cue_decide(q, anchor, row)

    def _range_suffix(self, q, key):
        doc = self.by_key[key]
        qs = doc.questions[q]
        if not doc.numbered or qs.range_done:
            return Stage.SKIP
        return [[qs.range_tokens[-1]]]

    def _rows(self, row) -> np.ndarray:
        """Return a stage's answer as its rows of readout values."""
        values = np.asarray(row)
        if values.dtype == object:
            values = np.asarray(values.reshape(-1)[0], dtype=np.float32)
        return values.reshape(-1, self.readout.width)

    def _extras(self, values) -> np.ndarray:
        """Return a row's extra values: none, the stop, then the lines targets."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return values[:len(values) - 2 * self.readout.k]

    def _range_decide(self, q, anchor, row, first=False):
        doc, qs = self._question(q, anchor)
        tables = self.tables
        if not first:
            qs.fed_tokens += 1
        values = self._rows(row)[-1]
        token = tables.range_targets[int(np.argmax(self._extras(values)[2:]))]
        if token in tables.range_text:
            qs.range_tokens.append(token)
        else:
            qs.range_done = True
        if len(qs.range_tokens) == LINE_MAX_TOKENS:
            qs.range_done = True
        if qs.range_done:
            answer = "".join(tables.range_text[t] for t in qs.range_tokens)
            qs.range = parse_lines(answer, len(doc.lines))
            if qs.range is None:
                qs.path = "fallback"
            else:
                a, b = qs.range
                qs.region = (doc.lines[a - 1][0], doc.lines[b - 1][1])
        return True

    # ---- the answer, a round of tokens at a time ---------------------------

    def _cue_suffix(self, q, key):
        doc = self.by_key[key]
        return [self.tables.begins_cue] if doc.numbered else Stage.SKIP

    def _cue_decide(self, q, anchor, row):
        """Choose the answer's first token from the row after the cue."""
        doc, qs = self._question(q, anchor)
        if doc.numbered:
            qs.fed_tokens += len(self.tables.begins_cue)
        return self._verify(doc, q, self._rows(row), [])

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
        extras = self._extras(values)
        lp = dict(zip(values[-2 * k:-k].astype(np.int64).tolist(),
                      values[-k:].tolist()))
        return lp, float(extras[0]), float(extras[1])

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
        position in the region; afterwards, the likeliest continuation
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
            if qs.region is not None and not any(
                    qs.region[0] <= span[0] < qs.region[1] for _, span in doc.tokens):
                qs.region = None      # the range maps to no tokens
                qs.path = "fallback"
            starts = first_tokens(tables, lp, doc.prefix, doc.tokens, qs.region)
            qs.none_lp = none_lp
            qs.best_lp = max((lp[token] for token, _ in starts.values()),
                             default=NEG)
            allowed = {token for token, _ in starts.values()}
            qs.none_wins = (max(none_lp, stop_lp) > qs.best_lp
                            or self._leaves(tables, lp, allowed, qs.best_lp, stop_lp))
            if qs.none_wins:
                self._answer(doc, q)
                return None
            token = max(starts.values(), key=lambda s: lp[s[0]])[0]
            qs.live = make_streams(tables, doc.text, doc.tokens, starts, token)
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

    def answers(self, q: int) -> tuple:
        """Return every document's answer and span for question q."""
        return ([doc.questions[q].answer for doc in self.docs],
                [doc.questions[q].span for doc in self.docs])

    def counts(self, q: int) -> dict:
        """Return how many documents took each path and how much was fed."""
        states = [doc.questions[q] for doc in self.docs]
        return {
            "located": sum(qs.path == "located" for qs in states),
            "fallback": sum(qs.path == "fallback" for qs in states),
            "scored": sum(qs.path == "scored" for qs in states),
            "none": sum(qs.answer is None for qs in states),
            "steps": sum(qs.steps for qs in states),
            "answer_tokens": sum(len(qs.written) for qs in states),
            "suffix_tokens": sum(qs.fed_tokens for qs in states),
        }
