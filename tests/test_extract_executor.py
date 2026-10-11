"""The Quail executor's AI.EXTRACT: its steps on a scripted model."""

import re
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch, letter_tokens

import quail
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor import extract as extract_mod
from quail.backends.quail.executor.extract import (
    CHUNK,
    LINE_MAX_TOKENS,
    MAX_PASSES,
    ExtractStages,
    Tables,
    best_span,
    choose_candidates,
    document_lines,
    document_tokens,
    make_branches,
    numbered_text,
    parse_lines,
    rank_starts,
    same_word,
)
from quail.backends.quail.executor.operators.extract import (
    execute_extract,
    extract_answer_table,
)
from quail.backends.quail.executor.readout import split_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.catalog import DocumentProvider
from quail.execution.execute import execute_query
from quail.execution.runner import (
    ExecutionContext,
    GenericRunner,
    NodeMetrics,
    NodeResult,
    compute_subgraph,
    scalar_node_metrics,
)
from quail.execution.types import PhysicalResponse, export_physical_outputs
from quail.physical import AiExtract, ExtractSpec
from quail.planner.plan import EngineConfig

NEG = float("-inf")
QUESTION = "When does the lease end?"
LEASE = ("Contract of lease.\nThe lease ends on\nMarch 3, 2027 at noon.\n"
         "Rent is $2,000.")
TERM = "Term: five years"
NOTHING = "No dates here."
HEAD = "DOCUMENT:\n"
NUMBERED_HEAD = "DOCUMENT, as numbered lines:\n"
TAIL = ' question\nPhrase: "'
NUMBERED_TAIL = " question\nLines: "


class FakeTokenizer:
    """Words, numbers, and punctuation as tokens, spaced like a byte-level BPE.

    A token after a space starts with "Ġ", a line break is "Ċ", and the
    offsets of a token include its leading space, as a Hugging Face
    fast tokenizer reports them.
    """

    PIECES = re.compile(r"\n| ?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]| ")

    def __init__(self, texts):
        self.tokens = ['"', "none", "-", "\n", "<|im_end|>", "Phrase", ":",
                       " question", "Lines"]
        self.tokens.extend(str(digit) for digit in range(10))
        for text in texts:
            for piece in self.PIECES.findall(text):
                if piece not in self.tokens:
                    self.tokens.append(piece)
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.eos_token_id = self.ids["<|im_end|>"]
        self.unk_token_id = None

    def __len__(self):
        return len(self.tokens)

    def _pieces(self, text):
        at = 0
        for match in self.PIECES.finditer(text):
            if match.start() != at:
                raise ValueError(f"cannot tokenize {text[at:match.start()]!r}")
            yield match.group(0), match.span()
            at = match.end()
        if at != len(text):
            raise ValueError(f"cannot tokenize {text[at:]!r}")

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        ids, offsets = [], []
        for piece, span in self._pieces(text):
            ids.append(self.ids[piece])
            offsets.append(span)
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def convert_ids_to_tokens(self, ids):
        return [self.spelled(index) for index in ids]

    def spelled(self, index):
        token = self.tokens[index]
        if token == "\n":
            return "Ċ"
        return "Ġ" + token[1:] if token.startswith(" ") else token

    def convert_tokens_to_ids(self, token):
        text = ("\n" if token == "Ċ" else " " + token[1:]
                if token.startswith("Ġ") else token)
        return self.ids.get(text, -1)


RANKED = "The lease ends on March 3. The rent is due."


def _tables(*extra):
    tokenizer = FakeTokenizer([LEASE, TERM, NOTHING, HEAD, NUMBERED_HEAD, TAIL,
                               NUMBERED_TAIL, "March", "five", RANKED,
                               numbered_text(LEASE, document_lines(LEASE))[0],
                               *extra])
    return Tables(tokenizer)


def _spec(trim=True, *extra):
    tables = _tables(*extra)
    return ExtractSpec(
        name="ends", aliases=("c",), query_template="{0}",
        arguments=(("c", "text"),), expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=(tuple(tables.encode_ids(HEAD)),
                            tuple(tables.encode_ids(TAIL))),
        question=QUESTION, trim=trim,
        numbered_token_parts=(tuple(tables.encode_ids(NUMBERED_HEAD)),
                              tuple(tables.encode_ids(NUMBERED_TAIL)))), tables


def test_lines_numbering_parsing_and_token_offsets():
    assert document_lines("a\n\n  \nbb\nc") == [(0, 1), (6, 8), (9, 10)]
    assert document_lines("one line") == [(0, 8)]
    assert document_lines("") == []
    text, pieces = numbered_text("ab\n\ncd", [(0, 2), (4, 6)])
    assert text == "1: ab\n2: cd"
    assert pieces == [(3, 0, 2), (9, 4, 2)]
    assert parse_lines("3-4", 5) == (3, 4) and parse_lines("3", 5) == (3, 3)
    for bad in ("4-3", "0-1", "2-9", "-3", "3-", "", "3-4-5", "a"):
        assert parse_lines(bad, 5) is None, bad
    assert same_word("ĠMarch") == "march" and same_word("the") == "the"
    assert same_word("Ċ") == "" and same_word(5) == "" and same_word("ĠaĠb") == ""
    tables = _tables()
    body, pieces = numbered_text(LEASE, document_lines(LEASE))
    ids, offsets = tables.encode(body)
    tokens = document_tokens(ids, offsets, pieces, 3)
    # a line's first token carries the space after its number, so it
    # is clipped to the line's text; no number token is a document token
    spelled = {index - 3: tables.vocab[ids[index - 3]] for index, _ in tokens}
    assert "1" not in spelled.values() and ":" not in spelled.values()
    assert [LEASE[a:b] for _, (a, b) in tokens][:4] == [
        "Contract", " of", " lease", "."]
    assert all(LEASE[a:b].strip() == tables.vocab[ids[index - 3]].lstrip("Ġ")
               for index, (a, b) in tokens
               if tables.vocab[ids[index - 3]] != "Ċ")


def test_start_ranking_tells_repeated_words_apart_and_keeps_the_first():
    tables = _tables()
    text = RANKED
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(ids, offsets, [(0, 0, len(text))], 1)
    the, march = tables.encode_ids("The")[0], tables.encode_ids("March")[0]
    space_march = tables.encode_ids(" March")[0]
    lp = {the: -0.5, march: -1.0, space_march: -3.0, tables.none_id: -4.0}
    starts = rank_starts(tables, lp, prefix, tokens, None)
    # the model writes "March" without its space; the document's token
    # has one, and both count as the same start
    assert starts["ranked"][:2] == ["the", "march"]
    assert starts["best"]["march"] == march and starts["filler"]["march"] == []
    # "The" and "." each occur twice, so both get the next-token step
    assert starts["ambiguous"] == ["the", "."]
    assert (starts["none_lp"], starts["best_lp"]) == (-4.0, -0.5)
    # the second "The" is followed by " rent", which the model prefers
    rent = tables.encode_ids(" rent")[0]
    follow = {"the": {rent: -0.1, tables.encode_ids(" lease")[0]: -2.0}}
    candidates = choose_candidates(tables, starts, lp, follow, prefix)
    # a candidate's offset includes its token's leading space
    assert [(c.char, c.score) for c in candidates[:3]] == [
        (26, -0.6), (17, -1.0), (0, -2.5)]
    assert candidates[0].first == the and candidates[1].first == march
    # within a region, its first token is always a candidate, even
    # when the TOP_TOKENS likeliest start tokens leave it out
    region = (text.index(" on"), len(text))
    inside = rank_starts(tables, lp, prefix, tokens, region)
    assert inside["ranked"][:2] == ["the", "march"] and "on" in inside["ranked"]
    likely = {starts["best"][key]: -1.0 for key in starts["groups"]
              if key != "the"}
    assert len(likely) > 8
    inside = rank_starts(tables, likely, prefix, tokens, None)
    assert len(inside["ranked"]) == 8 and "the" not in inside["ranked"]
    forced = choose_candidates(tables, inside, likely, {}, prefix)
    assert forced[-1].char == 0 and forced[-1].score == NEG
    assert len(forced) == 9
    branches = make_branches(tables, text, candidates[:2], skip_to=None)
    assert branches[0].stream[0] == the and branches[0].char == 26
    # the first token ends at the candidate's end; the rest follow
    assert branches[1].ends[:4] == [23, 24, 25, 26]
    assert text[branches[1].char:branches[1].ends[3]] == " March 3."
    assert branches[0].skip == 0
    assert make_branches(tables, text, candidates[:1], skip_to=30)[0].skip == 1


def test_scores_and_best_span_follow_the_worked_example():
    branch = extract_mod.Branch(char=0, first=1, stream=[1, 2, 3, 4, 5],
                                ends=[1, 2, 3, 4, 5], skip=0, start=0.0,
                                copy=[np.log(.85), np.log(.90), np.log(.95),
                                      np.log(.05), np.log(.30)],
                                stop=[np.log(.10), np.log(.05), np.log(.02),
                                      np.log(.90), np.log(.60)], fed=5)
    scores = branch.scores()
    assert np.exp(scores[3]) == pytest.approx(.85 * .90 * .95 * .90)
    assert best_span([branch]) == (branch, 3)
    assert best_span([]) is None


class ScriptedModel:
    """Writes the gold answer: the lines that hold it, its words, then stops.

    The model prefers the gold span's next token while the fed tokens
    are a prefix of it, and the closing quote right after it.
    """

    def __init__(self, tables, docs, gold, cue_k, feed_k):
        self.tables = tables
        self.docs = docs            # key -> (text, numbered, range text)
        self.gold = gold            # key -> gold answer text, or None
        self.cue_k, self.feed_k = cue_k, feed_k
        self.targets = len(tables.range_targets)
        self.width = max(2 + self.targets + 2 * cue_k, 1 + 2 * feed_k)
        # the phrase cue ends in the quote that follows a space, which
        # is not the quote that closes an answer
        self.cue_token = tables.encode_ids(TAIL)[-1]
        self.kv = {}                # key -> tokens appended after the prefix
        self.packed = []

    def _row(self, values):
        row = np.full(self.width, np.nan, dtype=np.float32)
        row[:len(values)] = values
        return row

    def _top(self, lps, k, extras):
        """A cue or feed row: the extras, then the k likeliest ids and values."""
        best = sorted(lps.items(), key=lambda item: -item[1])[:k]
        while len(best) < k:
            best.append((0, -30.0))
        return self._row(list(extras) + [float(i) for i, _ in best]
                         + [v for _, v in best])

    def _cue(self, lps, none_lp, quote_lp, digits=None):
        """A cue row: none, quote, the lines targets, then the top tokens."""
        digits = digits or [-9.0] * self.targets
        return self._top(lps, self.cue_k, [none_lp, quote_lp, *digits])

    def _range_row(self, wanted):
        return self._cue({}, -9.0, -9.0, [0.0 if token == wanted else -9.0
                                         for token in self.tables.range_targets])

    def _gold_keys(self, key):
        gold = self.gold[key]
        ids = self.tables.encode_ids(gold) if gold else []
        return [same_word(self.tables.vocab[i]) or i for i in ids]

    def forward_chunk(self, chunk):
        rows = []
        for spec in chunk.specs:
            key = spec["key"]
            text, numbered, range_text = self.docs[key]
            for suffix in spec["suffixes"]:
                suffix = [int(t) for t in suffix]
                self.packed.append((key, suffix, spec.get("write_suffix_tokens", 0),
                                    bool(spec.get("read_all_rows"))))
                held = self.kv.get(key, [])
                path = held + suffix
                if spec.get("read_all_rows"):
                    rows.extend(self._feed_rows(key, suffix))
                elif path[-1] == self.cue_token:
                    rows.append(self._cue_row(key))
                elif held and held[-1] == self.cue_token:
                    rows.append(self._follow_row(key, suffix[-1]))
                elif numbered:
                    rows.append(self._next_digit(path, range_text))
                else:
                    raise AssertionError(f"unexpected request {suffix}")
                if spec.get("write_suffix_tokens", 0):
                    self.kv[key] = path
        return np.stack(rows)

    def _next_digit(self, path, range_text):
        """The lines answer, one digit or dash at a time, then a line break."""
        tables = self.tables
        written = "".join(tables.range_text.get(t, "") for t in path
                          if t in tables.range_text)
        # digits before the cue belong to the numbered lines, not the answer
        cue = tables.encode_ids(NUMBERED_TAIL)[-1]
        after = path[len(path) - 1 - path[::-1].index(cue):]
        written = "".join(tables.range_text.get(t, "") for t in after[1:])
        if written == range_text:
            return self._range_row(tables.encode_ids("\n")[0])
        return self._range_row(tables.encode_ids(range_text[len(written)])[0])

    def _cue_row(self, key):
        tables = self.tables
        gold = self._gold_keys(key)
        lps = {}
        if gold:
            # the model writes the first word without its space, and
            # every other document word is far behind
            for index in tables.variants.get(gold[0], []):
                lps[index] = -0.5 if not tables.vocab[index].startswith("Ġ") else -0.9
            the = tables.variants.get("the", [])
            for index in the:
                lps[index] = -2.0
        none_lp = -0.2 if not gold else -5.0
        return self._cue(lps, none_lp, -6.0)

    def _follow_row(self, key, token):
        return self._cue({}, -7.0, -7.0)

    def _feed_rows(self, key, suffix):
        """Feed rows: the gold span's next token is likely, then the quote.

        The last fed token's row names no next token, as the fake does
        not look past the suffix; a gold span that runs past a pass
        would lose its branch there.
        """
        tables = self.tables
        gold = self._gold_keys(key)
        rows = []
        for j in range(len(suffix)):
            fed = [same_word(tables.vocab[t]) or t for t in suffix[:j + 1]]
            on_gold = fed == gold[:len(fed)] and len(fed) <= len(gold)
            following = suffix[j + 1] if j + 1 < len(suffix) else None
            if following is None:
                following_key = None
            else:
                following_key = same_word(tables.vocab[following]) or following
            lps = {}
            if following is not None:
                still = (on_gold and len(fed) < len(gold)
                         and following_key == gold[len(fed)])
                lps[following] = -0.2 if still else -2.5
            stop = -0.1 if on_gold and len(fed) == len(gold) else -4.0
            rows.append(self._top(lps, self.feed_k, [stop]))
        return rows


def _fake_readouts(tables, cue_k, feed_k):
    def readout(k, width_of, ragged=False):
        value = SimpleNamespace(k=k, width=width_of, ragged=ragged)
        value.dtype = (np.dtype(object) if ragged
                       else np.dtype((np.float32, (width_of,))))

        def submit(rows, rows_per_answer=None):
            rows = np.asarray(rows, dtype=np.float32)[:, :width_of]
            return rows, rows_per_answer

        def result(handle):
            rows, rows_per_answer = handle
            return split_rows(rows, rows_per_answer) if ragged else rows

        value.submit, value.result = submit, result
        return value

    return (readout(cue_k, 2 + len(tables.range_targets) + 2 * cue_k),
            readout(feed_k, 1 + 2 * feed_k, ragged=True))


def _state(model, chunk_tokens=400):
    return QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(256),
            pipeline=fake_pipeline(forward_chunk=model.forward_chunk),
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object()),
        torch=fake_torch(), chunk_tokens=chunk_tokens, answer_rows=object(),
        async_answers=object())


DOCS = [LEASE, TERM, NOTHING]
GOLD = ["March 3, 2027", "five years", None]
RANGES = ["2-3", "", ""]


def _run(monkeypatch, trim=True, documents=None):
    spec, tables = _spec(trim)
    documents = DOCS if documents is None else documents
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    cue_k, feed_k = 4, 3
    model = ScriptedModel(tables, {}, {}, cue_k, feed_k)
    readouts = _fake_readouts(tables, cue_k, feed_k)
    monkeypatch.setattr(ExtractStages, "_readouts", lambda self, state: readouts)
    state = _state(model)
    state.loaded_model.extract_tables = tables
    node = AiExtract(node_id="ai-extract:0", backend_name="quail",
                     model="qwen3-4b-fp8", spec=spec)
    tokens = {"c": [tables.encode_ids(text) for text in documents]}
    stages_seen = {}
    original = ExtractStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for doc in self.docs:
            model.docs[doc.key] = (doc.text, doc.numbered,
                                   RANGES[documents.index(doc.text)])
            model.gold[doc.key] = GOLD[documents.index(doc.text)]
        stages_seen["stages"] = self

    monkeypatch.setattr(ExtractStages, "__init__", init)
    result = execute_extract(state, node, {
        "document_ids": list(range(len(documents))), "documents": tokens,
        "texts": {"c": documents}, "pre": [], "gpu_timing": False})
    return result, stages_seen["stages"], model


def test_extractor_locates_lines_reads_the_start_and_scores_the_spans(monkeypatch):
    result, stages, model = _run(monkeypatch)
    table = result.outputs["scores"].to_pydict()
    assert table == {
        "c": [0, 1, 2],
        "ends": ["March 3, 2027", "five years", None],
        "ends_span": [{"start": LEASE.index("March"),
                       "end": LEASE.index("March") + len("March 3, 2027")},
                      {"start": 6, "end": 16}, None]}
    assert result.outputs["ids:c"] == [0, 1, 2]
    lease, term, nothing = stages.docs
    # the four-line lease is shown numbered; the model names lines 2-3,
    # so the starts are read in them and the ends from line 3 on
    assert lease.numbered and lease.path == "located"
    assert lease.range == (2, 3)
    assert lease.region == (LEASE.index("The lease"), LEASE.index("\nRent"))
    assert lease.skip_to == LEASE.index("March")
    assert [c.char for c in lease.candidates][0] == LEASE.index("March")
    assert lease.passes == 1 and not lease.open
    # the two short documents are shown plain; the model writes none
    # for the one without dates, and no start is read for it
    assert not term.numbered and term.path == "scored"
    assert not nothing.numbered and nothing.none_wins
    assert nothing.candidates == [] and nothing.passes == 0
    # the steps: the numbered tail and each digit are appended to the
    # lease's KV, then the phrase cue; the candidates are fed after it
    tables = stages.tables
    lease_key = lease.key
    appended = [suffix for key, suffix, kept, _ in model.packed
                if key == lease_key and kept]
    assert appended[0] == list(lease.tail)
    assert [tables.vocab[s[0]] for s in appended[1:4]] == ["2", "-", "3"]
    assert appended[4] == list(tables.begins_cue)
    # a plain document appends its tail, whose last row is its cue
    term_appended = [suffix for key, suffix, kept, _ in model.packed
                     if key == term.key and kept]
    assert term_appended == [list(term.tail)]
    # the candidate spans are read whole; a start token the lines
    # repeat (the space before a number) got its one-token step first
    steps = [suffix for key, suffix, kept, whole in model.packed
             if key == lease_key and not kept and not whole]
    assert steps == [[tables.encode_ids(" ")[0]]]
    fed = [suffix for key, suffix, kept, whole in model.packed
           if key == lease_key and whole]
    # the likeliest start, "March" as the model writes it, is fed with
    # the document's tokens after it, CHUNK past the start of line 3
    march = tables.encode_ids("March")[0]
    assert lease.branches[0].first == march
    (first,) = [suffix for suffix in fed if suffix[0] == march]
    assert len(first) == lease.branches[0].fed <= lease.branches[0].skip + CHUNK
    assert len(fed) == len(lease.candidates) <= 8
    assert lease.key == ("extract", "ends", 0)
    assert term.key == ("c", 1) and nothing.key == ("c", 2)
    assert result.metrics.extension["located"] == 1
    assert result.metrics.extension["scored"] == 2
    assert result.metrics.extension["none"] == 1
    assert result.metrics.fresh_tokens > 0
    assert result.metrics.output_rows == 3


def test_extractor_returns_whole_lines_with_trim_off(monkeypatch):
    result, stages, _ = _run(monkeypatch, trim=False)
    table = result.outputs["scores"].to_pydict()
    line = LEASE.index("March 3, 2027 at noon.")
    assert table["ends"] == ["March 3, 2027 at noon.", "Term: five years", None]
    assert table["ends_span"][0] == {"start": line, "end": line + 22}
    assert table["ends_span"][1] == {"start": 0, "end": len(TERM)}


def test_extractor_stops_after_the_pass_limit(monkeypatch):
    # a model that never closes the quote keeps every start open; the
    # passes stop at MAX_PASSES and the best span so far answers
    long = " ".join(["word"] * 200)
    spec, tables = _spec(True, long)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4, 3)

    word = tables.encode_ids(" word")[0]

    def feed_rows(key, suffix):
        # every next token is " word", the last fed one's included
        return [model._top({suffix[j + 1] if j + 1 < len(suffix) else word: -0.01},
                           3, [-3.0]) for j in range(len(suffix))]

    model._feed_rows = feed_rows
    readouts = _fake_readouts(tables, 4, 3)
    monkeypatch.setattr(ExtractStages, "_readouts", lambda self, state: readouts)
    state = _state(model, chunk_tokens=2000)
    state.loaded_model.extract_tables = tables
    texts = [long]
    stages = ExtractStages(state, spec, tables, [0],
                           [tables.encode_ids(long)], texts)
    model.docs[stages.docs[0].key] = (long, False, "")
    model.gold[stages.docs[0].key] = "word"
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys)
    stages.finish()
    doc = stages.docs[0]
    assert doc.passes == MAX_PASSES and not doc.open
    assert doc.answer is not None
    assert len(stages.stages) == LINE_MAX_TOKENS + 2 + MAX_PASSES
    assert [stage.label for stage in stages.stages][:2] == [
        "ends tail", "ends lines 1"]


def test_answer_table_and_session_project_the_answer_columns():
    spec, _ = _spec()
    table = extract_answer_table(spec, [3, 1], ["x", None], [(0, 1), None])
    assert table.column_names == ["c", "ends", "ends_span"]
    assert table.to_pydict() == {"c": [3, 1], "ends": ["x", None],
                                 "ends_span": [{"start": 0, "end": 1}, None]}

    session = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    session.register("contracts", DocumentProvider.from_table(
        pa.table({"id": ["a", "b"], "text": [LEASE, NOTHING]}), id_col="id"))
    query = (session.docs("contracts").alias("c")
             .ai_extract("c.text", QUESTION, name="ends")
             .select("c.id", "ends"))

    class Execution:
        documents = {}

        def execute(self, node, inputs):
            assert isinstance(node, AiExtract)
            (ids,) = inputs.values()
            answers = [("March 3, 2027", (20, 33)) if document == 0 else (None, None)
                       for document in ids]
            return NodeResult(
                {"scores": extract_answer_table(
                    node.spec, list(ids), [a for a, _ in answers],
                    [s for _, s in answers]),
                 "ids:c": list(ids)},
                NodeMetrics(input_rows=len(ids), output_rows=len(ids)))

    def execute(request):
        graph = compute_subgraph(query.plan().graph)
        run = GenericRunner().run(graph, ExecutionContext(
            runtimes=session.registry.runtimes, model_execution=Execution(),
            sources={"c": range(2), **request.relations}))
        return PhysicalResponse(
            export_physical_outputs(graph, run),
            {"backend": "quail", "wall_s": 0.1, "fresh_tokens": 12,
             "cached_tokens": 3, "node_metrics": scalar_node_metrics(run.nodes)})

    result = execute_query(query, physical_executor=execute)
    rows = result.collect()
    assert rows.column_names == ["c.id", "ends", "ends_span"]
    assert rows.column("c.id").to_pylist() == ["a", "b"]
    assert rows.column("ends").to_pylist() == ["March 3, 2027", None]
    assert rows.column("ends_span").to_pylist() == [{"start": 20, "end": 33}, None]
    answers = result.answer_tables["extracts"]["ends"]
    assert answers.column_names == ["c", "ends", "ends_span"]
    session.close()
