"""The Quail executor's AI.EXTRACT: its steps on a scripted model."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from extract_fakes import (
    LEASE,
    NOTHING,
    TAIL,
    TERM,
    make_spec,
)
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.extract.candidates import Stream
from quail.backends.quail.executor.extract.decode import AnswerState, Decoder
from quail.backends.quail.executor.extract.stages import ExtractionStages
from quail.backends.quail.executor.extract.vocabulary import same_word
from quail.backends.quail.executor.operators.extract import (
    execute_extract,
    execute_extractions,
)
from quail.backends.quail.executor.readout import split_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.execution.tokens import TokenOffsets, TokenStore
from quail.extraction import ExtractionConfig
from quail.physical import (
    AiExtract,
    AiFilter,
    FilterStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS


@pytest.mark.parametrize(("scores", "stop", "chosen", "done"), [
    ([-1, -1, -9, -10], -8, 1, False),
    ([-2, -1, -0.5, -10], -8, 1, True),
    ([-2, -1, -9, -10], -1, None, True),
    ([-2, -1, -9, -0.5], -8, None, True),
])
@pytest.mark.parametrize("tokens", [[1], [33, 1, 1]])
def test_continuation_arrays_handle_duplicates_ties_quotes_and_stops(
        scores, stop, chosen, done, tokens):
    merged = np.full(64, -1, dtype=np.int64)
    merged[1] = 40
    vocabulary = SimpleNamespace(vocab=range(64), merged=merged,
                                 is_stop=np.zeros(64, dtype=bool), none_id=63)
    vocabulary.is_stop[62] = True
    decoder = Decoder(vocabulary, 4, 10, [True], ExtractionConfig())
    text = "abcdefghij"
    streams = [Stream(at, [0, token], [at + 1, at + 2], text, exhausted=True)
               for at, token in enumerate(tokens)]
    if len(tokens) > 1:
        streams.append(Stream(5, [0], [6], text, exhausted=True))
    answer = AnswerState(written=[0], live=streams)
    doc = SimpleNamespace(answers=[answer], text=text)
    actual = decoder.choose(doc, 0, [-20, stop, 33, 1, 40, 50, *scores])
    assert actual == (chosen if not done else None)
    assert answer.done == done
    assert answer.written == ([0, chosen] if chosen is not None else [0])
    if chosen is not None:
        assert [s.char for s in answer.live] == [
            index for index, token in enumerate(tokens) if token == chosen]
    if chosen is not None and done:
        start = tokens.index(chosen)
        assert answer.span == (start, start + 2)


def test_readout_scores_reset_between_rows_and_keep_token_order():
    _, vocabulary = make_spec()
    decoder = Decoder(vocabulary, 4, 10, [True], ExtractionConfig())
    scores, none, stop = decoder.lp([-4, -5, 5, 2, 3, 4, -1, -1, -2, -3])
    assert decoder.top == 5 and none == -4 and stop == -5
    assert scores[5] == scores[2] == -1
    reused, _, _ = decoder.lp([-6, -7, 6, 1, 0, 0, -0.5, -1, -20, -30])
    assert reused is scores
    assert decoder.top == 6
    assert reused[5] == reused[2] == -np.inf
    assert reused[0] == -30


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
    assert stages.docs[0].answers[0].none_wins


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
    assert not stages.docs[0].answers[0].none_wins


def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch):
    # "five" starts two positions; the draft follows the first, "five
    # days", and the model chooses " years" instead: the round ends
    # there, and the next round feeds " years" in the draft's place
    text = "five days or five years"
    spec, tables = make_spec(True, text)
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    stages = ExtractionStages(state, [spec], tables, [0],
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
    asked = stages.docs[0].answers[0]
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


def _setup(monkeypatch, trim=True, documents=None, token_documents=None,
           share_prefixes=False, document_ids=None):
    """A state, its scripted model, and a node builder for one spec name."""
    spec, tables = make_spec(trim)
    documents = DOCS if documents is None else documents
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model)
    state.loaded_model.vocabulary = tables
    tokens = {"c": ([tables.encode_ids(text) for text in documents]
                    if token_documents is None else token_documents)}
    stages_seen = {}
    original = ExtractionStages.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for doc in self.docs:
            model.docs[doc.key] = doc.text
            model.gold[doc.key] = GOLD[documents.index(doc.text)]
        stages_seen["stages"] = self

    monkeypatch.setattr(ExtractionStages, "__init__", init)

    def node(name):
        return AiExtract(node_id=f"ai-extract:{name}", backend_name="quail",
                         model="qwen3-4b-fp8", spec=replace(
                             spec, name=name, share_prefixes=share_prefixes))

    inputs = {"document_ids": (list(range(len(documents))) if document_ids is None
                               else document_ids), "documents": tokens,
              "texts": {"c": documents}, "pre": [], "gpu_timing": False}

    def run(name="ends"):
        result = execute_extract(state, node(name), inputs)
        return result, stages_seen["stages"]

    def run_pipeline(*names):
        results = execute_extractions(state, [node(name) for name in names],
                                      inputs)
        return results, stages_seen["stages"]

    return state, model, run, run_pipeline


def test_extraction_borrows_prefix_pages_and_keeps_decode_kv(monkeypatch):
    documents = [LEASE, LEASE, NOTHING]
    _, _, plain, _ = _setup(monkeypatch, documents=documents)
    unshared, _ = plain()
    _, model, shared, _ = _setup(
        monkeypatch, documents=documents, share_prefixes=True)
    result, stages = shared()
    assert result.outputs["scores"].equals(unshared.outputs["scores"])
    borrowed = result.metrics.extension["borrowed_prefix_tokens"]
    assert borrowed > 0
    assert result.metrics.fresh_tokens == unshared.metrics.fresh_tokens - borrowed
    for doc in stages.docs:
        rounds = [entry for entry in model.packed if entry[0] == doc.key]
        if doc.answers[0].answer is not None:
            assert len(rounds) > 1
        assert model.reads[doc.key] <= 1
        assert all(entry[2] > 0 for entry in rounds)



@pytest.mark.parametrize("document_ids", [[2, 0], [1], []])
def test_extraction_preserves_selected_document_order(monkeypatch, document_ids):
    _, _, run, _ = _setup(monkeypatch, document_ids=document_ids)
    result, _ = run()
    table = result.outputs["scores"]
    assert table.column("c").to_pylist() == document_ids
    assert table.column("ends").to_pylist() == [GOLD[i] for i in document_ids]
    assert table.column("ends_span").is_null().to_pylist() == [
        GOLD[i] is None for i in document_ids]

def _run(monkeypatch, trim=True, documents=None):
    _, model, run, _ = _setup(monkeypatch, trim, documents)
    result, stages = run()
    return result, stages, model


def test_extraction_reuses_arrow_tokens_and_offsets(monkeypatch, tmp_path):
    _, vocabulary = make_spec()

    offsets = TokenOffsets({index: text.encode() for index, text
                            in enumerate(vocabulary.tokenizer.tokens)})

    store = TokenStore.write(
        str(tmp_path / "extract.arrow"), [pa.record_batch({"text": DOCS})],
        document_column="text", tokenizer=vocabulary.encode_ids,
        token_type=pa.int32(), offsets_encoder=offsets)
    _, model, run, _ = _setup(monkeypatch, token_documents=store)
    encode_batch = model.tables.encode_batch

    def windows_only(texts):
        assert all(text not in DOCS for text in texts)
        return encode_batch(texts)

    monkeypatch.setattr(model.tables, "encode_batch", windows_only)
    result, stages = run()
    assert result.outputs["scores"].column("ends").to_pylist() == GOLD
    assert result.outputs["scores"].column("ends_span").to_pylist() == [
        {"start": LEASE.index("March"),
         "end": LEASE.index("March") + len(GOLD[0])},
        {"start": 6, "end": 16}, None]
    for index, doc in enumerate(stages.docs):
        assert np.shares_memory(doc.scan.tokens, store[index].numpy())
        assert np.shares_memory(doc.scan.starts, store[index].character_offsets)
    store.close()


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
    lease, term, _ = stages.docs
    asked = lease.answers[0]
    assert result.metrics.extension["none"] == 1
    assert result.metrics.extension["answer_tokens"] == (
        len(asked.written) + len(term.answers[0].written))
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
    lease = stages.docs[0]
    (tail,) = set(map(tuple, stages.tails))
    appended = [suffix for key, suffix, kept, _, _ in model.packed
                if key == lease.key and kept]
    written = lease.answers[0].written
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
    spec, tables = make_spec()
    plain = replace(spec, shares_head=True)
    head = list(plain.prompt_token_parts[0])
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
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
    execution.loaded_model.vocabulary = tables
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


def test_extraction_pipeline_rejects_mixed_heads_or_decoding_settings():
    spec, tables = make_spec(True, LEASE)
    state = SimpleNamespace(chunk_tokens=400)
    documents = [tables.encode_ids(LEASE)]
    readout = SimpleNamespace(k=4, width=2 + 2 * 4)
    head = list(spec.prompt_token_parts[0])
    other = replace(spec, prompt_token_parts=(tuple(head[1:]),
                                              spec.prompt_token_parts[1]))
    with pytest.raises(ValueError, match="prompt head"):
        ExtractionStages(state, [spec, other], tables, [0], documents, [LEASE],
                      readout=readout)
    other = replace(spec, config=ExtractionConfig(draft_tokens=2))
    with pytest.raises(ValueError, match="decoding settings"):
        ExtractionStages(state, [spec, other], tables, [0], documents, [LEASE],
                         readout=readout)


@pytest.mark.parametrize("limit,draft", [(64, 16), (5, 1), (3, 0), (1, 0)])
def test_extractor_stops_at_the_answer_token_cap(monkeypatch, limit, draft):
    # a model that never closes the quote keeps writing the document;
    # the answer ends at stages.config.max_answer_tokens
    long = " ".join(["word"] * 200)
    spec, tables = make_spec(True, long)
    spec = replace(spec, config=ExtractionConfig(
        max_answer_tokens=limit, draft_tokens=draft))
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    model = ScriptedModel(tables, {}, {}, 4)
    readout = _fake_readout(tables, 4)
    monkeypatch.setattr(ExtractionStages, "_readout", lambda self, state: readout)
    state = _state(model, chunk_tokens=2000)
    state.loaded_model.vocabulary = tables
    stages = ExtractionStages(state, [spec], tables, [0],
                           [tables.encode_ids(long)], [long])
    model.docs[stages.docs[0].key] = long
    model.gold[stages.docs[0].key] = long
    from quail.backends.quail.executor.stages import run_stages

    run_stages(state.torch, state.loaded_model.arena,
               state.loaded_model.pipeline, stages.stages, stages.prefixes,
               state.chunk_tokens, anchor_keys=stages.keys)
    stages.finish()
    asked = stages.docs[0].answers[0]
    assert asked.done and len(asked.written) == stages.config.max_answer_tokens
    assert asked.answer == " ".join(["word"] * stages.config.max_answer_tokens)
    assert all(len(suffix) <= 1 + draft
               for _, suffix, _, whole, _ in model.packed if whole)


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
