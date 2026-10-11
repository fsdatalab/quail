"""Copy each document's answer to a question with the stage scheduler.

A document with LOCATE_MIN_LINES lines or more is shown as numbered
lines: the model writes the lines that hold the answer, then the
executor reads the likeliest start positions in those lines after
the phrase cue. A shorter document is shown plain and the start
positions are read over the whole of it. From each start, the
document's own tokens are fed CHUNK at a time; each row gives the
probability of copying the next token and of stopping, so every span
from the start is scored in one pass, and a start whose copy
probability falls below the best span's score is dropped. The
document, the lines, and the cue stay in KV across the steps; the
candidate spans are fed after them and read whole.
"""

import re
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from quail.backends.quail.executor.model import full_output_head
from quail.backends.quail.executor.readout import AsyncTopLogprobs
from quail.backends.quail.executor.stages import Stage
from quail.execution.spans import line_span
from quail.logical.prompts import EXTRACT_PHRASE_CUE
from quail.progress import say

TOP_TOKENS = 8      # start tokens read at the cue
MAX_STARTS = 8      # candidate start positions kept after expansion
CUE_TOP = 1024      # vocabulary entries read at the cue
FEED_TOP = 32       # vocabulary entries read at each fed row
CHUNK = 16          # document tokens fed per pass from each start
MAX_PASSES = 8      # passes a document may take: a runaway guard
LOCATE_MIN_LINES = 3    # documents with fewer lines are shown plain
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


@dataclass
class Candidate:
    """A start position: its first token and how likely the model begins there."""

    char: int
    end: int
    first: int
    score: float
    filler: list


@dataclass
class Branch:
    """The spans from one start, scored along the document's tokens."""

    char: int
    first: int
    stream: list        # the tokens from the start to the document's end
    ends: list          # the document character each stream token ends at
    skip: int           # tokens before the last line, whose ends are not scored
    start: float = NEG  # the first token's log probability at the cue
    copy: list = field(default_factory=list)
    stop: list = field(default_factory=list)
    fed: int = 0

    def scores(self) -> list:
        """Return the score of the span ending after each fed token."""
        total, out = self.start, []
        for copy, stop in zip(self.copy, self.stop):
            out.append(total + stop)
            total += copy
        return out


@dataclass
class Extraction:
    """One document's prompt, its steps' answers, and its result."""

    index: int
    text: str
    lines: list
    numbered: bool
    prefix: list
    tokens: list        # (prefix index, (start, end)) per document token
    tail: list
    key: tuple
    range_tokens: list = field(default_factory=list)
    range_done: bool = False
    range: tuple | None = None
    region: tuple | None = None
    skip_to: int | None = None
    path: str = "scored"
    none_lp: float = NEG
    best_lp: float = NEG
    cue_lp: dict = field(default_factory=dict)
    starts: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    branches: list = field(default_factory=list)
    open: list = field(default_factory=list)
    passes: int = 0
    fed_tokens: int = 0
    answer: str | None = None
    span: tuple | None = None
    none_wins: bool = False


def best_start(tables: Tables, lp, key, token) -> tuple:
    """Return the likeliest token that begins a document token, and the rest.

    The model may write a word without its leading space, in another
    case, or split differently from the document: any token whose
    text is a prefix of the word, up to case, can start it. A shorter
    one is followed by the rest of the word as the document writes it.
    """
    if not isinstance(key, str) or not key.isascii():
        ids = tables.variants.get(key, [token]) if isinstance(key, str) else [token]
        return max(ids, key=lambda i: lp.get(i, NEG)), []
    # the whole word first, so it wins a tie with a prefix of itself
    options = [(i, j) for j in range(len(key), 0, -1)
               for i in tables.variants.get(key[:j], [])]
    first, j = max(options, key=lambda o: lp.get(o[0], NEG))
    text = tables.vocab[token]
    text = text[1:] if text.startswith("Ġ") else text
    rest = tables.encode_ids(text[j:]) if j < len(key) else []
    return first, rest


def rank_starts(tables: Tables, lp, prefix, tokens, region) -> dict:
    """Rank the document tokens the answer can start with.

    Every document token in the region is scored as the best of the
    tokens that can begin it. The TOP_TOKENS likeliest start tokens
    are kept; one that occurs at several positions is ambiguous and
    gets one more step, the log probability of each position's next
    token after it, so the positions are told apart.

    Returns:
        The ranked start keys, each key's best first token and filler,
        the positions of each key, the ambiguous keys, the positions
        in the region, and the none and best log probabilities.
    """
    inside = [(i, span) for i, span in tokens
              if region is None or region[0] <= span[0] < region[1]]
    groups = {}
    for i, span in inside:
        key = same_word(tables.vocab[prefix[i]]) or prefix[i]
        groups.setdefault(key, []).append((i, span))
    best, filler = {}, {}
    for key, occurrences in groups.items():
        best[key], filler[key] = best_start(
            tables, lp, key, prefix[occurrences[0][0]])
    ranked = sorted(groups, key=lambda k: -lp.get(best[k], NEG))[:TOP_TOKENS]
    return {"ranked": ranked, "best": best, "filler": filler,
            "groups": groups, "inside": inside,
            "ambiguous": [key for key in ranked if len(groups[key]) > 1],
            "none_lp": lp.get(tables.none_id, NEG),
            "best_lp": lp.get(best[ranked[0]], NEG) if ranked else NEG}


def choose_candidates(tables: Tables, starts: dict, lp, follow, prefix) -> list:
    """Return the MAX_STARTS likeliest start positions, the region's first always.

    Args:
        tables: The token tables.
        starts: The ranked starts from rank_starts.
        lp: The cue row's log probabilities.
        follow: Per ambiguous key, the log probabilities after its token.
        prefix: The prompt's tokens.
    """
    best, filler, groups = starts["best"], starts["filler"], starts["groups"]

    def candidate(key, i, span):
        score = lp.get(best[key], NEG)
        if key in follow:
            after = (filler[key] + prefix[i + 1:i + 2] + [tables.quote_id])[0]
            score += follow[key].get(after, NEG)
        return Candidate(span[0], span[1], best[key], score, filler[key])

    scored = sorted((candidate(key, i, span) for key in starts["ranked"]
                     for i, span in groups[key]), key=lambda c: -c.score)
    scored = scored[:MAX_STARTS]
    inside = starts["inside"]
    if inside and all(c.char != inside[0][1][0] for c in scored):
        i, span = inside[0]
        key = same_word(tables.vocab[prefix[i]]) or prefix[i]
        if key not in best:
            best[key], filler[key] = best_start(tables, lp, key, prefix[i])
        scored.append(candidate(key, i, span))
    return scored


def make_branches(tables: Tables, text: str, candidates, skip_to) -> list:
    """Return one branch per candidate: its tokens to the document's end."""
    branches = []
    for c in candidates:
        ids, offsets = tables.encode(text[c.end:])
        stream = [c.first, *c.filler, *ids]
        ends = [c.end] * (1 + len(c.filler)) + [c.end + b for _, b in offsets]
        skip = sum(end <= skip_to for end in ends) if skip_to else 0
        branches.append(Branch(c.char, c.first, stream, ends,
                               min(skip, len(stream))))
    return branches


def best_span(branches) -> tuple | None:
    """Return the branch and fed token index of the best span, or None."""
    spans = [(score, index, j) for index, branch in enumerate(branches)
             for j, score in enumerate(branch.scores())]
    if not spans:
        return None
    _, index, j = max(spans, key=lambda item: item[0])
    return branches[index], j


class ExtractStages:
    """The stages of one extraction and the answers they produce.

    Args:
        state: The loaded model and current query state.
        spec: The ExtractSpec.
        tables: The model's token tables.
        document_ids: The documents' ids in their table.
        documents: The table's tokenized documents.
        texts: The table's document texts.
        readouts: Optional (cue, feed) readouts in place of the model's.
            The cue readout's extras are the none and quote tokens, then
            the lines answer's targets.
    """

    def __init__(self, state, spec, tables: Tables, document_ids, documents,
                 texts, readouts=None):
        self.spec = spec
        self.tables = tables
        self.docs = [self._prepare(index, document, texts[document],
                                   None if documents is None
                                   else documents[document])
                     for index, document in enumerate(document_ids)]
        self.by_key = {doc.key: doc for doc in self.docs}
        self.prefixes = [doc.prefix for doc in self.docs]
        self.keys = [doc.key for doc in self.docs]
        if readouts is None:
            readouts = self._readouts(state)
        self.cue_readout, self.feed_readout = readouts
        longest = max((len(doc.tokens) for doc in self.docs), default=0)
        tails = max((len(doc.tail) for doc in self.docs), default=1)
        name = spec.name
        # every document appends its tail first: a numbered one reads
        # the lines answer's first token there, a plain one its start
        self.stages = [Stage(
            suffixes=self._tail_suffix, suffix_tokens=tails,
            readout=self.cue_readout, decide=self._tail_decide,
            single=True, append=True, label=f"{name} tail")]
        self.stages.extend(
            Stage(suffixes=self._range_suffix, suffix_tokens=1,
                  readout=self.cue_readout, decide=self._range_decide,
                  single=True, append=True, label=f"{name} lines {r}")
            for r in range(1, LINE_MAX_TOKENS))
        self.stages.append(Stage(
            suffixes=self._cue_suffix, suffix_tokens=len(tables.begins_cue),
            readout=self.cue_readout, decide=self._cue_decide,
            single=True, append=True, label=f"{name} start"))
        self.stages.append(Stage(
            suffixes=self._step_suffix, suffix_tokens=1,
            readout=self.cue_readout, decide=self._step_decide,
            label=f"{name} start step"))
        # a branch's tokens are the document's text from its start,
        # tokenized afresh, so they may outnumber the document's tokens;
        # a suffix is fed whole, so a pass never exceeds the chunk budget
        self.pass_bound = {p: min(2 * longest + CHUNK * (p + 1), state.chunk_tokens)
                           for p in range(1, MAX_PASSES + 1)}
        self.stages.extend(
            Stage(suffixes=partial(self._pass_suffix, p),
                  suffix_tokens=self.pass_bound[p],
                  readout=self.feed_readout, decide=partial(self._pass_decide, p),
                  read_all_rows=True, label=f"{name} pass {p}")
            for p in range(1, MAX_PASSES + 1))

    def _readouts(self, state) -> tuple:
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        F = torch.nn.functional
        tables = self.tables
        return (AsyncTopLogprobs(torch, F, head, CUE_TOP,
                                 [tables.none_id, tables.quote_id,
                                  *tables.range_targets]),
                AsyncTopLogprobs(torch, F, head, FEED_TOP, [tables.quote_id],
                                 ragged=True))

    def _prepare(self, index, document, text, stored) -> Extraction:
        """Build a document's prefix: the head and its plain or numbered text.

        A plain document whose tokens are the table's keeps the table's
        arena key, so KV an earlier operator left resident is read;
        any other prefix gets a key of its own.
        """
        spec, tables = self.spec, self.tables
        text = text.as_py() if hasattr(text, "as_py") else str(text)
        lines = document_lines(text)
        numbered = len(lines) >= LOCATE_MIN_LINES
        if numbered:
            head, tail = spec.numbered_token_parts
            body, pieces = numbered_text(text, lines)
        else:
            head, tail = spec.prompt_token_parts
            body, pieces = text, [(0, 0, len(text))]
        ids, offsets = tables.encode(body)
        key = ("extract", spec.name, index)
        if not numbered and stored is not None and [int(t) for t in stored] == ids:
            key = (spec.alias, int(document))
        prefix = list(head) + ids
        tokens = document_tokens(ids, offsets, pieces, len(head))
        return Extraction(index=index, text=text, lines=lines,
                          numbered=numbered, prefix=prefix, tokens=tokens,
                          tail=list(tail), key=key,
                          path="located" if numbered else "scored")

    # ---- the tail, then the lines range -----------------------------------

    def _tail_suffix(self, key):
        return [self.by_key[key].tail]

    def _tail_decide(self, anchor, row):
        doc = self.docs[anchor]
        doc.fed_tokens += len(doc.tail)
        if doc.numbered:
            return self._range_decide(anchor, row, first=True)
        return self._cue_decide(anchor, row)

    def _range_suffix(self, key):
        doc = self.by_key[key]
        if not doc.numbered or doc.range_done:
            return Stage.SKIP
        return [[doc.range_tokens[-1]]]

    def _extras(self, values) -> np.ndarray:
        """Return a cue row's extra values: none, quote, then the lines targets."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return values[:len(values) - 2 * self.cue_readout.k]

    def _range_decide(self, anchor, row, first=False):
        doc = self.docs[anchor]
        tables = self.tables
        if not first:
            doc.fed_tokens += 1
        token = tables.range_targets[int(np.argmax(self._extras(row)[2:]))]
        if token in tables.range_text:
            doc.range_tokens.append(token)
        else:
            doc.range_done = True
        if len(doc.range_tokens) == LINE_MAX_TOKENS:
            doc.range_done = True
        if doc.range_done:
            answer = "".join(tables.range_text[t] for t in doc.range_tokens)
            doc.range = parse_lines(answer, len(doc.lines))
            if doc.range is None:
                doc.path = "fallback"
            else:
                a, b = doc.range
                doc.region = (doc.lines[a - 1][0], doc.lines[b - 1][1])
                doc.skip_to = doc.lines[b - 1][0] if b > a else None
        return True

    # ---- the start -------------------------------------------------------

    def _cue_suffix(self, key):
        doc = self.by_key[key]
        return [self.tables.begins_cue] if doc.numbered else Stage.SKIP

    def _lp(self, values) -> dict:
        """Return a cue row's log probabilities by token id."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        k = self.cue_readout.k
        extras = self._extras(values)
        lp = dict(zip(values[-2 * k:-k].astype(np.int64).tolist(),
                      values[-k:].tolist()))
        lp.setdefault(self.tables.none_id, float(extras[0]))
        lp.setdefault(self.tables.quote_id, float(extras[1]))
        return lp

    def _cue_decide(self, anchor, row):
        doc = self.docs[anchor]
        if doc.numbered:
            doc.fed_tokens += len(self.tables.begins_cue)
        if doc.region is not None and not any(
                doc.region[0] <= span[0] < doc.region[1] for _, span in doc.tokens):
            doc.region = doc.skip_to = None      # the range maps to no tokens
            doc.path = "fallback"
        lp = self._lp(row)
        doc.starts = rank_starts(self.tables, lp, doc.prefix, doc.tokens,
                                 doc.region)
        doc.none_lp, doc.best_lp = doc.starts["none_lp"], doc.starts["best_lp"]
        doc.none_wins = doc.none_lp > doc.best_lp
        doc.cue_lp = lp
        if doc.none_wins:
            # the model would write none: no start is read
            doc.starts["ambiguous"] = []
            doc.answer, doc.span = None, None
        elif not doc.starts["ambiguous"]:
            self._finish_candidates(doc, {})
        return True

    def _step_suffix(self, key):
        doc = self.by_key[key]
        if not doc.starts.get("ambiguous"):
            return Stage.SKIP
        return [[doc.starts["best"][k]] for k in doc.starts["ambiguous"]]

    def _step_decide(self, anchor, row):
        doc = self.docs[anchor]
        rows = np.asarray(row, dtype=np.float32).reshape(-1, self.cue_readout.width)
        doc.fed_tokens += len(rows)
        follow = {key: self._lp(values)
                  for key, values in zip(doc.starts["ambiguous"], rows)}
        self._finish_candidates(doc, follow)
        return True

    def _finish_candidates(self, doc, follow):
        doc.candidates = choose_candidates(
            self.tables, doc.starts, doc.cue_lp, follow, doc.prefix)
        doc.branches = make_branches(self.tables, doc.text, doc.candidates,
                                     doc.skip_to)
        doc.open = list(range(len(doc.branches)))
        if not doc.open:
            doc.answer, doc.span = None, None

    # ---- the spans -------------------------------------------------------

    def _pass_suffix(self, p, key):
        doc = self.by_key[key]
        doc.open = [i for i in doc.open if self._take(doc, i, p) > 0]
        if not doc.open:
            self._answer(doc)
            return Stage.SKIP
        return [doc.branches[i].stream[:doc.branches[i].fed + self._take(doc, i, p)]
                for i in doc.open]

    def _take(self, doc, i, p) -> int:
        """Return how many more tokens the branch feeds at pass p."""
        branch = doc.branches[i]
        take = (branch.skip if p == 1 else 0) + CHUNK
        take = min(take, len(branch.stream) - branch.fed,
                   self.pass_bound[p] - branch.fed)
        return max(take, 0)

    def _pass_decide(self, p, anchor, row):
        doc = self.docs[anchor]
        k = self.feed_readout.k
        for i, values in zip(doc.open, row):
            branch = doc.branches[i]
            values = np.asarray(values, dtype=np.float32).reshape(-1, 1 + 2 * k)
            take = self._take(doc, i, p)
            doc.fed_tokens += len(values)
            for j in range(branch.fed, branch.fed + take):
                ids = values[j, 1:1 + k].astype(np.int64).tolist()
                lps = values[j, 1 + k:].tolist()
                following = branch.stream[j + 1] if j + 1 < len(branch.stream) else None
                copy = NEG
                if following is not None and following in ids:
                    copy = lps[ids.index(following)]
                branch.copy.append(copy)
                branch.stop.append(float(values[j, 0]) if j >= branch.skip else NEG)
            if branch.start == NEG:
                branch.start = doc.cue_lp.get(branch.first, NEG)
            branch.fed += take
        doc.passes = p
        best = max((max(b.scores()) for b in doc.branches if b.fed), default=NEG)
        doc.open = [i for i in doc.open
                    if doc.branches[i].fed < len(doc.branches[i].stream)
                    and doc.branches[i].start + sum(doc.branches[i].copy) >= best]
        if not doc.open or p == MAX_PASSES:
            self._answer(doc)
        return True

    def _answer(self, doc):
        """Set the document's answer: its best span, or none."""
        doc.open = []
        found = best_span(doc.branches)
        if found is None or doc.none_wins:
            doc.answer, doc.span = None, None
            return
        branch, j = found
        start, end = branch.char, branch.ends[j]
        text = doc.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if not self.spec.trim:
            start, end = line_span(text, (start, end))
        doc.answer, doc.span = text[start:end], (start, end)

    def finish(self) -> None:
        """Answer every document the passes left open."""
        for doc in self.docs:
            if doc.open:
                self._answer(doc)

    def counts(self) -> dict:
        """Return how many documents took each path and how much was fed."""
        return {
            "located": sum(doc.path == "located" for doc in self.docs),
            "fallback": sum(doc.path == "fallback" for doc in self.docs),
            "scored": sum(doc.path == "scored" for doc in self.docs),
            "none": sum(doc.answer is None for doc in self.docs),
            "passes": sum(doc.passes for doc in self.docs),
            "suffix_tokens": sum(doc.fed_tokens for doc in self.docs),
        }
