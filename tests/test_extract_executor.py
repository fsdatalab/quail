"""The Quail executor's AI.EXTRACT: its steps on a scripted model."""

import re
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch, letter_tokens

import quail
from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.extract import (
    ExtractStages,
    Tables,
    document_tokens,
    first_tokens,
    make_streams,
    same_word,
    squeezed,
)
from quail.backends.quail.executor.operators.extract import (
    execute_extract,
    execute_extractions,
    extract_answer_table,
)
from quail.backends.quail.executor.readout import split_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.catalog import DocumentProvider
from quail.cost.extract import ANSWER_MAX_TOKENS, DRAFT_TOKENS
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
from quail.physical import (
    AiExtract,
    AiFilter,
    ExtractSpec,
    FilterStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.planner.plan import EngineConfig
from quail.specs import DEVICES, MODELS

NEG = float("-inf")
QUESTION = "When does the lease end?"
LEASE = ("Contract of lease.\nThe lease ends on\nMarch 3, 2027 at noon.\n"
         "Rent is $2,000.")
TERM = "Term: five years"
NOTHING = "No dates here."
HEAD = "DOCUMENT:\n"
TAIL = ' question\nPhrase: "'


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
        if isinstance(text, list):
            each = [self(one) for one in text]
            return {"input_ids": [e["input_ids"] for e in each],
                    "offset_mapping": [e["offset_mapping"] for e in each]}
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
    # a stream writes a run of whitespace as one space, so the
    # vocabulary holds the squeezed texts' tokens too
    tokenizer = FakeTokenizer([LEASE, TERM, NOTHING, HEAD, TAIL, "March",
                               "five", RANKED,
                               *(squeezed(text)[0] for text in (LEASE, TERM,
                                                                NOTHING)),
                               *extra])
    return Tables(tokenizer)


def _spec(trim=True, *extra):
    tables = _tables(*extra)
    return ExtractSpec(
        name="ends", aliases=("c",), query_template="{0}",
        arguments=(("c", "text"),), expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=(tuple(tables.encode_ids(HEAD)),
                            tuple(tables.encode_ids(TAIL))),
        question=QUESTION, trim=trim), tables


def test_word_keys_and_token_offsets():
    assert same_word("ĠMarch") == "march" and same_word("the") == "the"
    assert same_word("Ċ") == "" and same_word(5) == "" and same_word("ĠaĠb") == ""
    tables = _tables()
    ids, offsets = tables.encode(LEASE)
    tokens = document_tokens(offsets, 3)
    assert [LEASE[a:b] for _, (a, b) in tokens][:4] == [
        "Contract", " of", " lease", "."]
    assert [index for index, _ in tokens] == list(range(3, 3 + len(ids)))


def test_first_tokens_follow_the_models_spelling():
    tables = _tables()
    text = RANKED
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    the, march = tables.encode_ids("The")[0], tables.encode_ids("March")[0]
    space_march = tables.encode_ids(" March")[0]
    lp = {the: -0.5, march: -1.0, space_march: -3.0}
    starts = first_tokens(tables, lp, prefix, tokens)
    by_char = {tokens[i - 1][1][0]: value for i, value in starts.items(tables)}
    # the model writes "March" without its space; the document's token
    # has one, and the likelier spelling starts it; "The" twice
    assert by_char[text.index(" March")] == (march, [])
    assert by_char[0] == (the, []) and by_char[26] == (the, [])
    assert set(by_char) == {0, 26, text.index(" March")}
    assert {tokens[i - 1][1][0] for i in starts} == set(by_char)
    # the chosen first token names the streams, each the text from
    # its start as the model would write it, with its character ends
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 26]
    # a stream holds its first token until the writing reaches it
    assert streams[1].stream == [the]
    streams[1].extend(tables)
    assert streams[1].stream[:3] == tables.encode_ids("The rent is")
    # a start's offset includes its token's leading space
    assert text[streams[1].char:streams[1].ends[2]] == " The rent is"
    (march_stream,) = make_streams(tables, text, starts, march)
    march_stream.extend(tables)
    assert march_stream.ends[:4] == [23, 24, 25, 26]
    assert text[march_stream.char:march_stream.ends[3]] == " March 3."


def test_a_stream_writes_a_run_of_whitespace_as_one_space():
    text = "laws of the State of          Arkansas\nshall apply."
    tables = _tables(text, "laws of the State of Arkansas shall apply.")
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    laws = tables.encode_ids("laws")[0]
    starts = first_tokens(tables, {laws: -0.5}, prefix, tokens)
    (stream,) = make_streams(tables, text, starts, laws)
    stream.extend(tables)
    # the model writes " Arkansas" right after "of"; the stream does too,
    # and its end is the document's, past the spaces
    assert stream.stream == tables.encode_ids(
        "laws of the State of Arkansas shall apply.")
    assert text[stream.char:stream.ends[5]] == "laws of the State of          Arkansas"
    assert text[stream.char:stream.ends[6]].endswith("Arkansas\nshall")
    assert squeezed("a  b\n\n c") == ("a b c", [1, 3, 4, 7, 8])


def test_a_stream_tokenizes_a_window_at_a_time():
    from quail.backends.quail.executor import extract as extract_mod

    text = " ".join(["word"] * 300)
    tables = _tables(text)
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    word = tables.encode_ids("word")[0]
    starts = first_tokens(tables, {word: -0.5}, prefix, tokens)
    streams = make_streams(tables, text, starts, word)
    assert len(streams) == 300
    first = streams[0]
    # one window of the text is tokenized ahead; the rest on demand
    window_tokens = len(first.stream)
    assert window_tokens < 300 and not first.exhausted
    assert first.next(window_tokens + 5, tables) == tables.encode_ids(" word")[0]
    assert len(first.stream) > window_tokens
    assert first.next(10_000, tables) is None and first.exhausted
    assert text[first.char:first.ends[-1]] == text
    assert extract_mod.STREAM_WINDOW == 512


def test_writing_that_begins_outside_the_document_answers_none(monkeypatch):
    # at the cue the model prefers a word that begins no position: its
    # answer is not in the document, and neither extraction answers
    state, model, run, _ = _setup(monkeypatch)
    tables = model.tables
    stray = tables.encode_ids(" five")[0]
    cue_row = model._cue_row

    def straying(key):
        if key[1] != 0:
            return cue_row(key)
        march = tables.encode_ids("March")[0]
        return model._cue({stray: -0.05, march: -0.5}, -5.0, -6.0)

    model._cue_row = straying
    result, stages = run()
    assert result.outputs["scores"].column("ends").to_pylist() == [
        None, "five years", None]
    assert stages.docs[0].questions[0].none_wins


def test_writing_that_leaves_the_document_later_ends_the_answer(monkeypatch):
    # after "March" the model would rather write a word the document
    # does not continue with: the answer is what it wrote within the
    # document, as a contract's first party before an "and" the
    # contract does not have
    state, model, run, _ = _setup(monkeypatch)
    tables = model.tables
    stray = tables.encode_ids(" lease")[0]
    answer_row = model._answer_row

    def straying(key, written):
        row = answer_row(key, written)
        if key[1] == 0 and len(written) == 1:
            gold = model._gold_ids(key)
            return model._cue({stray: -0.05, gold[1]: -1.0}, -9.0, -4.0)
        return row

    model._answer_row = straying
    result, stages = run()
    assert result.outputs["scores"].column("ends").to_pylist() == [
        "March", "five years", None]
    assert not stages.docs[0].questions[0].none_wins


def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch):
    # "five" starts two positions; the draft follows the first, "five
    # days", and the model chooses " years" instead: the round ends
    # there, and the next round feeds " years" in the draft's place
    text = "five days or five years"
    spec, tables = _spec(True, text)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.extract_tables = tables
    stages = ExtractStages(state, [spec], tables, [0],
                           [tables.encode_ids(text)], [text])
    key = stages.docs[0].key
    model.docs[key] = text
    model.gold[key] = "five years"
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys,
               appended_tokens=stages.appended_tokens)
    stages.finish()
    asked = stages.docs[0].questions[0]
    assert asked.answer == "five years" and asked.span == (13, 23)
    assert asked.written == tables.encode_ids("five years")
    five, days, years = (tables.encode_ids(w)[0] for w in ("five", " days", " years"))
    fed = [(suffix, f) for k, suffix, kept, _, f in model.packed if k == key and kept]
    # the cue, then the two rounds: the second starts where the first
    # verified, one token past the first round's position
    assert fed[1][0][:2] == [five, days]
    assert fed[2][0][:1] == [years] and fed[2][1] == fed[1][1] + 1
    assert asked.steps == 3
    assert asked.fed_tokens == (len(tables.encode_ids(TAIL)) + len(fed[1][0])
                                + len(fed[2][0]))


def test_a_first_token_that_begins_a_longer_word_carries_its_rest():
    text = "The lease is their lease."
    tables = _tables(text, "their", "ir")
    ids, offsets = tables.encode(text)
    prefix = [99] + ids
    tokens = document_tokens(offsets, 1)
    the = tables.encode_ids("The")[0]
    starts = first_tokens(tables, {the: -0.5}, prefix, tokens)
    by_char = {tokens[i - 1][1][0]: value for i, value in starts.items(tables)}
    # "the" begins "their" too: that start continues with "ir"
    assert by_char[0] == (the, [])
    assert by_char[12] == (the, tables.encode_ids("ir"))
    streams = make_streams(tables, text, starts, the)
    assert [stream.char for stream in streams] == [0, 12]
    assert streams[1].stream[:2] == [the, tables.encode_ids("ir")[0]]
    assert text[streams[1].char:streams[1].ends[1]] == " their"


class ScriptedModel:
    """Writes the gold answer: the lines that hold it, its tokens, then stops.

    At the cue the model prefers the gold answer's first word, written
    without its space; at each later step it prefers the gold answer's
    next token while the tokens written so far are a prefix of it,
    and closes the quote once the whole answer is written.
    """

    def __init__(self, tables, docs, gold, k):
        self.tables = tables
        self.docs = docs            # key -> text
        self.gold = gold            # key -> gold answer text, or None
        self.k = k
        self.filter_truth = {}      # key -> a filter question's answer
        self.reads = Counter()      # key -> times its prefix was packed
        self.width = 2 + 2 * k
        # the phrase cue ends in the quote that follows a space, which
        # is not the quote that closes an answer
        self.cue_token = tables.encode_ids(TAIL)[-1]
        self.kv = {}                # key -> tokens appended after the prefix
        self.prefix_len = {}        # key -> the prefix's tokens
        self.packed = []

    def _row(self, values):
        row = np.full(self.width, np.nan, dtype=np.float32)
        row[:len(values)] = values
        return row

    def _top(self, lps, extras):
        """A row: the extras, then the k likeliest ids and values."""
        best = sorted(lps.items(), key=lambda item: -item[1])[:self.k]
        while len(best) < self.k:
            best.append((0, -30.0))
        return self._row(list(extras) + [float(i) for i, _ in best]
                         + [v for _, v in best])

    def _cue(self, lps, none_lp, stop_lp):
        """A row: none, the stop, then the top tokens."""
        return self._top(lps, [none_lp, stop_lp])

    def _gold_ids(self, key):
        gold = self.gold[key]
        return self.tables.encode_ids(gold) if gold else []

    def forward_chunk(self, chunk):
        rows = []
        for spec in chunk.specs:
            key = spec["key"]
            for suffix in spec["suffixes"]:
                suffix = [int(t) for t in suffix]
                read_all = bool(spec.get("read_all_rows"))
                self.packed.append((key, suffix, spec.get("write_suffix_tokens", 0),
                                    read_all, spec.get("f")))
                # the request's rows follow position f: a frame, a
                # rewinding stage, or a discarding one drops what an
                # earlier stage appended
                if spec.get("prefix") is not None:
                    self.prefix_len[key] = len(spec["prefix"])
                    self.reads[key] += 1
                held = self.kv.get(key, [])
                if spec.get("f") is not None and key in self.prefix_len:
                    held = held[:spec["f"] - self.prefix_len[key]]
                path = held + suffix
                # every row of a draft is read; otherwise the last
                for end in (range(len(held) + 1, len(path) + 1) if read_all
                            else [len(path)]):
                    rows.append(self._row_after(key, path[:end]))
                if spec.get("write_suffix_tokens", 0):
                    self.kv[key] = path
        return np.stack(rows)

    def _row_after(self, key, path):
        if path[-1] == self.cue_token:
            return self._cue_row(key)
        if self.cue_token in path:
            cue = len(path) - 1 - path[::-1].index(self.cue_token)
            return self._answer_row(key, path[cue + 1:])
        if key in self.filter_truth:
            # a filter's question: the row's first value is the answer
            row = self._cue({}, -9.0, -9.0)
            row[:] = float(self.filter_truth[key])
            return row
        raise AssertionError(f"unexpected request {path}")

    def _cue_row(self, key):
        tables = self.tables
        gold = self._gold_ids(key)
        lps = {}
        if gold:
            # the model writes the first word without its space, and
            # every other document word is far behind
            first = same_word(tables.vocab[gold[0]]) or gold[0]
            for index in tables.variants.get(first, [gold[0]]):
                lps[index] = -0.5 if not tables.vocab[index].startswith("Ġ") else -0.9
            for index in tables.variants.get("the", []):
                lps[index] = -2.0
        none_lp = -0.2 if not gold else -5.0
        return self._cue(lps, none_lp, -6.0)

    def _answer_row(self, key, written):
        """The answer's next token while the writing fits the gold, then the stop."""
        tables = self.tables
        gold = self._gold_ids(key)
        keys = [same_word(tables.vocab[t]) or t for t in gold]
        fed = [same_word(tables.vocab[t]) or t for t in written]
        if fed == keys:
            return self._cue({}, -9.0, -0.1)
        if fed == keys[:len(fed)]:
            return self._cue({gold[len(fed)]: -0.2}, -9.0, -4.0)
        return self._cue({}, -9.0, -0.5)


def _fake_readout(tables, k):
    width = 2 + 2 * k
    value = SimpleNamespace(k=k, width=width, ragged=True)
    value.dtype = np.dtype(object)

    def submit(rows, rows_per_answer=None):
        return split_rows(np.asarray(rows, dtype=np.float32)[:, :width],
                          rows_per_answer)

    value.submit, value.result = submit, lambda handle: handle
    return value


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


def _setup(monkeypatch, trim=True, documents=None):
    """A state, its scripted model, and a node builder for one spec name."""
    spec, tables = _spec(trim)
    documents = DOCS if documents is None else documents
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.extract_tables = tables
    tokens = {"c": [tables.encode_ids(text) for text in documents]}
    stages_seen = {}
    original = ExtractStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for doc in self.docs:
            model.docs[doc.key] = doc.text
            model.gold[doc.key] = GOLD[documents.index(doc.text)]
        stages_seen["stages"] = self

    monkeypatch.setattr(ExtractStages, "__init__", init)

    def node(name):
        return AiExtract(node_id=f"ai-extract:{name}", backend_name="quail",
                         model="qwen3-4b-fp8", spec=replace(spec, name=name))

    inputs = {"document_ids": list(range(len(documents))), "documents": tokens,
              "texts": {"c": documents}, "pre": [], "gpu_timing": False}

    def run(name="ends"):
        result = execute_extract(state, node(name), inputs)
        return result, stages_seen["stages"]

    def run_pipeline(*names):
        results = execute_extractions(state, [node(name) for name in names],
                                      inputs)
        return results, stages_seen["stages"]

    return state, model, run, run_pipeline


def _run(monkeypatch, trim=True, documents=None):
    _, model, run, _ = _setup(monkeypatch, trim, documents)
    result, stages = run()
    return result, stages, model


def tables_of(stages):
    return stages.tables


def test_extractor_writes_the_answer_along_the_document(monkeypatch):
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
    asked = lease.questions[0]
    # the model writes "March" without its space, then the document's
    # tokens after it, and closes the quote after "2027"
    march = tables_of(stages).encode_ids("March")[0]
    assert asked.written == [march, *tables_of(stages).encode_ids(" 3, 2027")]
    # the cue's row chooses "March"; one round feeds it with the
    # document's next tokens as a draft and verifies the rest
    assert asked.done and asked.steps == 2
    # the model writes none for the document without dates, and no
    # token is written for it
    assert nothing.questions[0].none_wins
    assert nothing.questions[0].written == []
    assert nothing.questions[0].steps == 1
    # the steps: the tail is appended to the lease's KV, then the
    # round of the answer
    (tail,) = stages.tails
    lease_key = lease.key
    appended = [suffix for key, suffix, kept, _, _ in model.packed
                if key == lease_key and kept]
    assert appended[0] == tail
    (round_fed,) = appended[1:]
    assert round_fed[:len(asked.written)] == asked.written
    # the draft runs to the document's end, short of DRAFT_TOKENS
    assert len(asked.written) < len(round_fed) <= 1 + DRAFT_TOKENS
    # the draft's rows are read whole; the tail reads its last
    assert [whole for key, _, kept, whole, _ in model.packed
            if key == lease_key and kept] == [False, True]
    # the keys name the document under the extraction's head, not the
    # output, so another extraction of the table reads the KV left
    # resident
    assert [doc.key for doc in stages.docs] == [
        (("extract", "c"), index) for index in range(3)]
    assert result.metrics.extension["none"] == 1
    assert result.metrics.extension["answer_tokens"] == (
        len(asked.written) + len(term.questions[0].written))
    assert result.metrics.fresh_tokens > 0
    assert result.metrics.output_rows == 3


def test_a_pipeline_of_extractions_reads_each_document_once(monkeypatch):
    state, model, _, run_pipeline = _setup(monkeypatch)
    results, stages = run_pipeline("ends", "starts")
    first, second = results["ai-extract:ends"], results["ai-extract:starts"]
    assert second.outputs["scores"].column("starts").to_pylist() == \
        first.outputs["scores"].column("ends").to_pylist()
    assert second.outputs["scores"].column("starts_span").to_pylist() == \
        first.outputs["scores"].column("ends_span").to_pylist()
    # one stage list asks both questions of a document; the second
    # question's tail is written after the document in place of the
    # first question's steps, the same frame rewound
    assert len(stages.stages) == 2 * (1 + ANSWER_MAX_TOKENS)
    assert stages.tails[0] == stages.tails[1]
    lease = stages.docs[0]
    (tail,) = set(map(tuple, stages.tails))
    appended = [suffix for key, suffix, kept, _, _ in model.packed
                if key == lease.key and kept]
    written = lease.questions[0].written
    round_fed = appended[1]
    assert round_fed[:len(written)] == written
    # the frame is written once, with the first question's cue token;
    # the second question feeds its cue token alone after the frame,
    # past the first question's round
    assert appended == [list(tail), round_fed, [tail[-1]], round_fed]
    # the documents are read once: the second extraction is charged
    # only the tokens fed after them
    prefixes = sum(len(doc.prefix) for doc in stages.docs)
    fed = [stages.counts(q)["suffix_tokens"] for q in range(2)]
    assert first.metrics.fresh_tokens == prefixes + fed[0]
    assert second.metrics.fresh_tokens == fed[1] > 0
    assert second.metrics.wall_s == 0.0 and first.metrics.wall_s > 0.0


def test_an_extraction_after_a_filter_asks_the_filtered_documents_once(
        monkeypatch):
    # the filter writes each document under the query's shared head;
    # the plain extraction continues its pipeline, so a document is
    # packed once and the extraction feeds only its question
    spec, tables = _spec()
    plain = replace(spec, shares_head=True)
    head = list(plain.prompt_token_parts[0])
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractStages, "_readout", lambda self, state: readout)
    for index, text in enumerate(DOCS):
        model.docs[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
        model.filter_truth[("c", index)] = index != 2
    question = tables.encode_ids(" question")
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiFilter(node_id="ai_filter:c",
                 inputs=input_ports((PortRef("input:c", "ids:c"),)),
                 alias="c", arena_writes=True,
                 stages=(FilterStage(0, len(question), len(head), 0.7, 2.0),),
                 question_token_ids=(tuple(question),)),
        AiExtract(node_id="ai-extract:ends",
                  inputs=input_ports((PortRef("ai_filter:c", "ids:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8", spec=plain),
    ), PortRef("ai-extract:ends", "scores"))
    torch, arena = fake_torch(), cpu_arena(256)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.loaded_model.extract_tables = tables
    execution.loaded_model.input_staging = SimpleNamespace(fixed_tokens=set())
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=SimpleNamespace(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"torch": torch, "arena": arena, "pipeline": pipeline,
             "model_execution": execution,
             "runtimes": built_in_registry().runtimes,
             "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
             "chunk_tokens": 400,
             "docs": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}}
    report = execute_single_graph(
        state, {"filter_limit": None, "pre_ids": head, "retention": {}}, graph)
    table = report["_outputs"][PortRef("ai-extract:ends", "scores")].to_pydict()
    assert table["c"] == [0, 1]
    assert table["ends"] == ["March 3, 2027", "five years"]
    # every document was packed once: the extraction's tail followed
    # the filter's question over the resident KV
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    metrics = report["node_metrics"]
    documents = sum(len(head) + len(tokens) for tokens in state["docs"]["c"])
    assert metrics["ai-extract:ends"]["fresh_tokens"] < documents
    assert metrics["ai_filter:c"]["fresh_tokens"] >= documents
    assert metrics["ai-extract:ends"]["evaluated_documents"] == 2
    assert not arena.accounting.owned


def test_a_second_extraction_reads_the_documents_resident_kv(monkeypatch):
    state, model, run, _ = _setup(monkeypatch)
    first, stages = run("ends")
    arena = state.loaded_model.arena
    # every document's KV stays resident after the extraction
    assert all(arena.is_resident(doc.key) for doc in stages.docs)
    assert all(arena.prefix_tokens(doc.key) == len(doc.prefix)
               for doc in stages.docs)
    model.kv = {}
    second, again = run("starts")
    assert [doc.key for doc in again.docs] == [doc.key for doc in stages.docs]
    assert second.outputs["scores"].column("starts").to_pylist() == \
        first.outputs["scores"].column("ends").to_pylist()
    # the second extraction feeds no document: only its tail, lines,
    # cue, and spans are fresh
    prefixes = sum(len(doc.prefix) for doc in stages.docs)
    assert first.metrics.fresh_tokens > prefixes
    assert second.metrics.fresh_tokens == first.metrics.fresh_tokens - prefixes


def test_extractor_returns_whole_lines_with_trim_off(monkeypatch):
    result, stages, _ = _run(monkeypatch, trim=False)
    table = result.outputs["scores"].to_pydict()
    line = LEASE.index("March 3, 2027 at noon.")
    assert table["ends"] == ["March 3, 2027 at noon.", "Term: five years", None]
    assert table["ends_span"][0] == {"start": line, "end": line + 22}
    assert table["ends_span"][1] == {"start": 0, "end": len(TERM)}


def test_the_table_key_names_kv_written_under_the_shared_head():
    spec, tables = _spec(True, LEASE)
    state = SimpleNamespace(chunk_tokens=400)
    documents = [tables.encode_ids(LEASE)]
    readout = SimpleNamespace(k=4)
    head = list(spec.prompt_token_parts[0])
    # under the query's shared head, the document's KV is the table's,
    # as a filter writes it
    shared = ExtractStages(state, [spec], tables, [0], documents, [LEASE],
                           readout=readout, shared_head=head)
    assert shared.docs[0].key == ("c", 0)
    # under another head, or with no shared head, the extraction's own
    own = ExtractStages(state, [spec], tables, [0], documents, [LEASE],
                        readout=readout, shared_head=head[1:])
    assert own.docs[0].key == (("extract", "c"), 0)
    alone = ExtractStages(state, [spec], tables, [0], documents, [LEASE],
                          readout=readout)
    assert alone.docs[0].key == (("extract", "c"), 0)
    # a pipeline's extractions share a prompt head
    other = replace(spec, prompt_token_parts=(tuple(head[1:]),
                                              spec.prompt_token_parts[1]))
    with pytest.raises(ValueError, match="prompt head"):
        ExtractStages(state, [spec, other], tables, [0], documents, [LEASE],
                      readout=readout)


def test_extractor_stops_at_the_answer_token_cap(monkeypatch):
    # a model that never closes the quote keeps writing the document;
    # the answer ends at ANSWER_MAX_TOKENS
    long = " ".join(["word"] * 200)
    spec, tables = _spec(True, long)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractStages, "_readout", lambda self, state: readout)
    state = _state(model, chunk_tokens=2000)
    state.loaded_model.extract_tables = tables
    stages = ExtractStages(state, [spec], tables, [0],
                           [tables.encode_ids(long)], [long])
    model.docs[stages.docs[0].key] = long
    model.gold[stages.docs[0].key] = long
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys)
    stages.finish()
    asked = stages.docs[0].questions[0]
    assert asked.done and len(asked.written) == ANSWER_MAX_TOKENS
    assert asked.answer == " ".join(["word"] * ANSWER_MAX_TOKENS)
    assert len(stages.stages) == 1 + ANSWER_MAX_TOKENS
    assert [stage.label for stage in stages.stages][:2] == [
        "ends tail", "ends round 1"]
    assert stages.stages[-1].label == f"ends round {ANSWER_MAX_TOKENS}"


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


def test_top_logprob_readout_sums_a_group_of_extras():
    import torch

    from quail.backends.quail.executor.readout import AsyncTopLogprobs

    head = torch.eye(6, dtype=torch.float32)
    readout = AsyncTopLogprobs(torch, torch.nn.functional, head, 2,
                               [(0, 1), 2])
    normed = torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]])
    (row,) = readout.values(normed).tolist()
    logprobs = torch.log_softmax(normed[0], dim=0).tolist()
    assert row[0] == pytest.approx(np.logaddexp(logprobs[0], logprobs[1]))
    assert row[1] == pytest.approx(logprobs[2])
    assert row[2:4] == [0.0, 1.0] and row[4:] == pytest.approx(logprobs[:2])
    assert readout.width == 2 + 2 * 2


def test_tables_pair_each_token_with_its_quote_closing_merge():
    tokenizer = FakeTokenizer([LEASE, TAIL])
    tokenizer.tokens.extend(['."', '""', 'Ġ"'])
    tokenizer.ids = {token: index for index, token in enumerate(tokenizer.tokens)}
    tables = Tables(tokenizer)
    ids = tokenizer.ids
    assert tables.merged == {ids["."]: ids['."'], ids['"']: ids['""'],
                             ids[" "]: ids['Ġ"']}
    assert set(tables.stop_ids) == {ids['"'], ids['""']}
