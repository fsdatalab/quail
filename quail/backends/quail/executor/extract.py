"""Copy each document's answers to its questions with the stage scheduler.

With locating on, a document is shown as numbered lines: the model
writes the lines that hold the answer, then the executor reads the
likeliest start positions in those lines after the phrase cue. With
locating off, the document is shown plain and the start positions are
read over the whole of it. From each start, the document's own tokens
are fed CHUNK at a time; each row gives the probability of copying the
next token and of stopping, so every span from the start is scored in
one pass, and a start whose copy probability falls below the best
span's score is dropped. The document stays in KV across the steps;
the candidate spans are fed after the cue and read whole.

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
from quail.execution.spans import line_span
from quail.logical.prompts import EXTRACT_PHRASE_CUE
from quail.progress import say

TOP_TOKENS = 8      # start tokens read at the cue
MAX_STARTS = 8      # candidate start positions kept after expansion
CUE_TOP = 1024      # vocabulary entries read at the cue
FEED_TOP = 128      # vocabulary entries read at each fed row
CHUNK = 16          # document tokens fed per pass from each start
MAX_PASSES = 8      # passes a document may take: a runaway guard
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
    """The span from one start, read along the document's tokens."""

    char: int
    first: int
    stream: list        # the tokens from the start to the document's end
    ends: list          # the document character each stream token ends at
    start: float = NEG  # the first token's log probability at the cue
    rank: float = NEG   # the candidate's score, which orders the starts
    copy: list = field(default_factory=list)    # per fed row, the next token
    stop: list = field(default_factory=list)    # per fed row, closing the quote
    merged: list = field(default_factory=list)  # per fed row, the next token
    #                                             and the quote in one token
    fed: int = 0

    @property
    def order(self) -> tuple:
        """The branch's place in the model's order of starts."""
        return self.start, self.rank

    def ending(self, force: bool = False) -> tuple | None:
        """Return where the span ends, or None while it copies on.

        The span ends where a decoder stops: at the first fed row
        after which closing the quote is at least as likely as copying
        the next token, or after the next token when copying it and
        closing the quote in one token is likelier than either.

        Args:
            force: Whether a span still copying ends after its last fed
                token.

        Returns:
            The stream token the span ends after, the fed row that
            decided it, and the log probability of stopping there.
        """
        for j, (copy, stop) in enumerate(zip(self.copy, self.stop)):
            merged = self.merged[j] if j < len(self.merged) else NEG
            if merged > max(copy, stop):
                return j + 1, j, merged
            if stop >= copy:
                return j, j, stop
        if force and self.copy:
            return len(self.copy) - 1, len(self.copy) - 1, self.stop[-1]
        return None

    def end(self, force: bool = False) -> int | None:
        """Return the stream token the span ends after, or None."""
        ending = self.ending(force)
        return None if ending is None else ending[0]

    def score(self, force: bool = False) -> float:
        """Return the ended span's score: its start, its copies, and its stop."""
        ending = self.ending(force)
        if ending is None:
            return NEG
        _, row, stop = ending
        return self.start + sum(self.copy[:row]) + stop


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
    are kept; one that occurs at several positions, or that begins
    another kept word too, as "the" begins "their", is ambiguous and
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
    firsts = [best[key] for key in ranked]
    return {"ranked": ranked, "best": best, "filler": filler,
            "groups": groups, "inside": inside,
            "ambiguous": [key for key in ranked if len(groups[key]) > 1
                          or firsts.count(best[key]) > 1],
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


def make_branches(tables: Tables, text: str, candidates, lp) -> list:
    """Return one branch per candidate: its tokens to the document's end.

    Args:
        tables: The token tables.
        text: The document's text.
        candidates: The start positions, from choose_candidates.
        lp: The cue row's log probabilities.
    """
    branches = []
    for c in candidates:
        ids, offsets = tables.encode(text[c.end:])
        stream = [c.first, *c.filler, *ids]
        ends = [c.end] * (1 + len(c.filler)) + [c.end + b for _, b in offsets]
        branches.append(Branch(c.char, c.first, stream, ends,
                               start=lp.get(c.first, NEG), rank=c.score))
    return branches


def open_branches(branches) -> list:
    """Return the indexes of the branches to feed next.

    The starts are fed in the model's order, the likeliest first, as a
    decoder writes its likeliest token. A start is fed while no
    better one has ended with a span; one below an ended span is never
    fed. Positions that tie, the same start and the same token after
    it, are fed together and told apart by what follows.
    """
    done = max((b.order for b in branches
                if b.end() is not None and b.score() > NEG), default=None)
    live = [(b.order, i) for i, b in enumerate(branches)
            if b.end() is None and b.fed < len(b.stream)
            and (done is None or b.order >= done)]
    if not live:
        return []
    top = max(order for order, _ in live)
    return [i for order, i in live if order == top]


def best_span(branches, force: bool = False) -> tuple | None:
    """Return the branch and fed token index of the answer, or None.

    The answer begins with the start the model ranks first among those
    whose span ended; among that start's positions, the span the model
    copies best wins.

    Args:
        branches: The candidates' branches.
        force: Whether a span still copying ends at its last fed token.
    """
    ended = [(branch.order, branch.score(force), index)
             for index, branch in enumerate(branches)
             if branch.score(force) > NEG]
    if not ended:
        return None
    _, _, index = max(ended)
    return branches[index], branches[index].end(force)


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
        readouts: Optional (cue, feed) readouts in place of the model's.
            The cue readout's extras are the none and quote tokens, then
            the lines answer's targets.

    Raises:
        ValueError: The specs differ in their table, locating, or head.
    """

    def __init__(self, state, specs, tables: Tables, document_ids, documents,
                 texts, readouts=None):
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
        if readouts is None:
            readouts = self._readouts(state)
        self.cue_readout, self.feed_readout = readouts
        longest = max((len(doc.tokens) for doc in self.docs), default=0)
        # a branch's tokens are the document's text from its start,
        # tokenized afresh, so they may outnumber the document's tokens;
        # a suffix is fed whole, so a pass never exceeds the chunk budget
        self.pass_bound = {p: min(2 * longest + CHUNK * (p + 1), state.chunk_tokens)
                           for p in range(1, MAX_PASSES + 1)}
        self.stages = []
        for q, (spec, tail) in enumerate(zip(self.specs, self.tails)):
            self.stages.extend(self._question_stages(q, spec.name, tail))

    def _question_stages(self, q: int, name: str, tail: list) -> list:
        """Build one question's stages: its tail, lines, start, and passes.

        The tail but its last token is the frame of every stage, so
        the scheduler writes it once after the document; the first
        stage rewinds past what the question before appended. The
        last token of the tail is the stage's suffix, whose row is
        read: the lines answer's first token for a numbered document,
        the start for a plain one.
        """
        frame, last = tail[:-1], tail[-1]
        stages = [Stage(
            suffixes=[[last]], frame=frame, rewind=True,
            readout=self.cue_readout, decide=partial(self._tail_decide, q),
            single=True, append=True, label=f"{name} tail")]
        stages.extend(
            Stage(suffixes=partial(self._range_suffix, q), suffix_tokens=1,
                  frame=frame, readout=self.cue_readout,
                  decide=partial(self._range_decide, q),
                  single=True, append=True, label=f"{name} lines {r}")
            for r in range(1, LINE_MAX_TOKENS))
        stages.append(Stage(
            suffixes=partial(self._cue_suffix, q),
            suffix_tokens=len(self.tables.begins_cue), frame=frame,
            readout=self.cue_readout, decide=partial(self._cue_decide, q),
            single=True, append=True, label=f"{name} start"))
        stages.append(Stage(
            suffixes=partial(self._step_suffix, q), suffix_tokens=1,
            frame=frame, readout=self.cue_readout,
            decide=partial(self._step_decide, q), label=f"{name} start step"))
        stages.extend(
            Stage(suffixes=partial(self._pass_suffix, q, p),
                  suffix_tokens=self.pass_bound[p], frame=frame,
                  readout=self.feed_readout,
                  decide=partial(self._pass_decide, q, p),
                  read_all_rows=True, label=f"{name} pass {p}")
            for p in range(1, MAX_PASSES + 1))
        return stages

    def _readouts(self, state) -> tuple:
        torch = state.torch
        head = full_output_head(state.loaded_model.model)
        F = torch.nn.functional
        tables = self.tables
        return (AsyncTopLogprobs(torch, F, head, CUE_TOP,
                                 [tables.none_id, tables.quote_id,
                                  *tables.range_targets]),
                AsyncTopLogprobs(torch, F, head, FEED_TOP, [tables.stop_ids],
                                 ragged=True))

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

    def _extras(self, values) -> np.ndarray:
        """Return a cue row's extra values: none, quote, then the lines targets."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        return values[:len(values) - 2 * self.cue_readout.k]

    def _range_decide(self, q, anchor, row, first=False):
        doc, qs = self._question(q, anchor)
        tables = self.tables
        if not first:
            qs.fed_tokens += 1
        token = tables.range_targets[int(np.argmax(self._extras(row)[2:]))]
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

    # ---- the start -------------------------------------------------------

    def _cue_suffix(self, q, key):
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

    def _cue_decide(self, q, anchor, row):
        doc, qs = self._question(q, anchor)
        if doc.numbered:
            qs.fed_tokens += len(self.tables.begins_cue)
        if qs.region is not None and not any(
                qs.region[0] <= span[0] < qs.region[1] for _, span in doc.tokens):
            qs.region = None      # the range maps to no tokens
            qs.path = "fallback"
        lp = self._lp(row)
        qs.starts = rank_starts(self.tables, lp, doc.prefix, doc.tokens,
                                qs.region)
        qs.none_lp, qs.best_lp = qs.starts["none_lp"], qs.starts["best_lp"]
        qs.none_wins = qs.none_lp > qs.best_lp
        qs.cue_lp = lp
        if qs.none_wins:
            # the model would write none: no start is read
            qs.starts["ambiguous"] = []
            qs.answer, qs.span = None, None
        elif not qs.starts["ambiguous"]:
            self._finish_candidates(doc, qs, {})
        return True

    def _step_suffix(self, q, key):
        qs = self.by_key[key].questions[q]
        if not qs.starts.get("ambiguous"):
            return Stage.SKIP
        return [[qs.starts["best"][k]] for k in qs.starts["ambiguous"]]

    def _step_decide(self, q, anchor, row):
        doc, qs = self._question(q, anchor)
        rows = np.asarray(row, dtype=np.float32).reshape(-1, self.cue_readout.width)
        qs.fed_tokens += len(rows)
        follow = {key: self._lp(values)
                  for key, values in zip(qs.starts["ambiguous"], rows)}
        self._finish_candidates(doc, qs, follow)
        return True

    def _finish_candidates(self, doc, qs, follow):
        qs.candidates = choose_candidates(
            self.tables, qs.starts, qs.cue_lp, follow, doc.prefix)
        qs.branches = make_branches(self.tables, doc.text, qs.candidates,
                                    qs.cue_lp)
        qs.open = open_branches(qs.branches)
        if not qs.open:
            qs.answer, qs.span = None, None

    # ---- the spans -------------------------------------------------------

    def _pass_suffix(self, q, p, key):
        doc = self.by_key[key]
        qs = doc.questions[q]
        qs.open = [i for i in qs.open if self._take(qs, i, p) > 0]
        if not qs.open:
            self._answer(doc, q)
            return Stage.SKIP
        return [qs.branches[i].stream[:qs.branches[i].fed + self._take(qs, i, p)]
                for i in qs.open]

    def _take(self, qs, i, p) -> int:
        """Return how many more tokens the branch feeds at pass p."""
        branch = qs.branches[i]
        take = min(CHUNK, len(branch.stream) - branch.fed,
                   self.pass_bound[p] - branch.fed)
        return max(take, 0)

    def _pass_decide(self, q, p, anchor, row):
        doc, qs = self._question(q, anchor)
        k = self.feed_readout.k
        for i, values in zip(qs.open, row):
            branch = qs.branches[i]
            values = np.asarray(values, dtype=np.float32).reshape(-1, 1 + 2 * k)
            take = self._take(qs, i, p)
            qs.fed_tokens += len(values)
            for j in range(branch.fed, branch.fed + take):
                ids = values[j, 1:1 + k].astype(np.int64).tolist()
                lps = values[j, 1 + k:].tolist()
                following = branch.stream[j + 1] if j + 1 < len(branch.stream) else None
                copy = merged = NEG
                if following is not None and following in ids:
                    copy = lps[ids.index(following)]
                joined = self.tables.merged.get(following)
                if joined is not None and joined in ids:
                    merged = lps[ids.index(joined)]
                branch.copy.append(copy)
                branch.merged.append(merged)
                branch.stop.append(float(values[j, 0]))
            branch.fed += take
        qs.passes = p
        qs.open = open_branches(qs.branches)
        if not qs.open or p == MAX_PASSES:
            self._answer(doc, q, force=p == MAX_PASSES)
        return True

    def _answer(self, doc, q, force: bool = False):
        """Set the question's answer for the document: its span, or none.

        Args:
            doc: The extraction.
            q: The question.
            force: Whether a span still copying at the pass limit ends
                at its last fed token.
        """
        qs = doc.questions[q]
        qs.open = []
        found = best_span(qs.branches, force)
        if found is None or qs.none_wins:
            qs.answer, qs.span = None, None
            return
        branch, j = found
        start, end = branch.char, branch.ends[j]
        text = doc.text
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if not self.specs[q].trim:
            start, end = line_span(text, (start, end))
        qs.answer, qs.span = text[start:end], (start, end)

    def finish(self) -> None:
        """Answer every question the passes left open."""
        for doc in self.docs:
            for q, qs in enumerate(doc.questions):
                if qs.open:
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
            "passes": sum(qs.passes for qs in states),
            "suffix_tokens": sum(qs.fed_tokens for qs in states),
        }
