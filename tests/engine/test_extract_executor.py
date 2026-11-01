"""Shared AI.EXTRACT execution on a scripted model."""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch as tensor_torch
from helpers.extract_fakes import (
    LEASE,
    NOTHING,
    TAIL,
    TERM,
)
from helpers.extract_fakes import (
    make_spec as make_extract_spec,
)
from helpers.extract_reference import AnswerState, Decoder
from helpers.fakes import fake_pipeline, fake_readout
from helpers.kv_checker import Setup, make_arena
from test_map_decode import cpu_staging, events_torch

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.distributed import (
    execute_distributed_graph,
)
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.extract.candidates import Stream
from quail.backends.quail.executor.extract.vocabulary import same_word
from quail.backends.quail.executor.operators.extract import (
    ExtractPart,
    execute_extractions,
)
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.backends.quail.graph import execute_single_graph
from quail.backends.quail.worker import ChildState, _child_execute
from quail.builtins import built_in_registry
from quail.logical import And, ColumnRef, GetField, Not, Or, RegularPredicate
from quail.physical import (
    AiExtract,
    AiFilter,
    PhysicalGraph,
    PortRef,
    Scan,
    plan_envelope,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS, QWEN3_4B_FP8


def make_spec(*texts):
    spec, vocabulary = make_extract_spec(*texts)
    return replace(spec, top_tokens=4), vocabulary


@pytest.fixture(autouse=True)
def packed_scripted_inputs(monkeypatch):
    cpu_staging(monkeypatch)
    monkeypatch.setattr("quail.backends.quail.executor.parts.input_staging",
                        lambda state: None)
    pack = chunk_mod.pack_chunk

    def capture(torch, arena, groups, **kwargs):
        chunk = pack(torch, arena, groups, **kwargs)
        chunk.specs = groups
        chunk.answer_keys = getattr(arena, "answer_keys", ())
        return chunk

    monkeypatch.setattr(chunk_mod, "pack_chunk", capture)


@pytest.mark.parametrize(("scores", "stop", "chosen", "done"), [
    ([-1, -1, -9, -10], -8, 1, False),
    ([-2, -1, -9, -0.5], -8, None, True),
])
def test_continuation_arrays_handle_duplicates_ties_and_stops(
        scores, stop, chosen, done):
    tokens = [33, 1, 1]
    vocabulary = SimpleNamespace(vocab=range(64), is_stop=np.zeros(64, dtype=bool))
    vocabulary.is_stop[62] = True
    decoder = Decoder(vocabulary, 4, 9, 1, make_spec()[0])
    text = "abcdefghij"
    streams = [Stream(at, [0, token], [at + 1, at + 2], text, exhausted=True)
               for at, token in enumerate(tokens)]
    streams.append(Stream(5, [0], [6], text, exhausted=True))
    answer = AnswerState(written=[0], streams=streams)
    document = SimpleNamespace(answers=[answer], text=text)
    actual = decoder.choose(document, 0, [stop, 33, 1, 40, 50, *scores])
    assert actual == (chosen if not done else None)
    assert answer.done == done
    assert answer.written == ([0, chosen] if chosen is not None else [0])
    if chosen is not None:
        assert [s.char for s in answer.streams] == [
            index for index, token in enumerate(tokens) if token == chosen]


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
        return model._cue({stray: -0.05, march: -0.5}, -6.0)

    model._cue_row = straying
    result, stages = run()
    assert result.outputs["rows:c"].column("ends").to_pylist() == [
        None, "five years", None]
    assert stages.documents[0].answers[0].none_wins


@pytest.mark.parametrize("recurrent", [False])
def test_a_rejected_draft_is_replaced_by_the_next_round(monkeypatch, recurrent):
    # "five" starts two positions; the draft follows the first, "five
    # days", and the model chooses " years" instead: the round ends
    # there, and the next round feeds " years" in the draft's place
    text = "five days or five years"
    asked, model = _extract_spec(monkeypatch, text, "five years", recurrent=recurrent)
    assert asked.answer == "five years" and asked.span == (13, 23)
    assert asked.written == model.tables.encode_ids(" five years")
    assert asked.steps >= 2
    assert any(model.tables.encode_ids(" days")[0] in suffix
               for _, suffix, _, _, _ in model.packed)


class ScriptedModel:
    """Writes the gold answer: the lines that hold it, its tokens, then stops.

    At the cue the model prefers the gold answer's first word, written
    with its space; at each later step it prefers the gold answer's
    next token while the tokens written so far are a prefix of it,
    and ends its turn once the whole answer is written. As in the
    model's readout, the stop value includes the probabilities of
    the stop tokens among the top tokens.
    """

    def __init__(self, tables, documents, gold, k):
        self.tables = tables
        self.documents = documents            # key -> text
        self.gold = gold            # key -> gold answer text, or None
        self.k = k
        self.filter_truth = {}      # key -> a filter question's answer
        self.reads = Counter()      # key -> times its prefix was packed
        self.width = 1 + 2 * k
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

    def _cue(self, lps, stop_lp):
        """A row: the stop, then the top tokens."""
        for token, value in lps.items():
            if self.tables.is_stop[token]:
                stop_lp = np.logaddexp(stop_lp, value)
        return self._top(lps, [stop_lp])

    def _gold_ids(self, key):
        gold = self.gold[key]
        return self.tables.encode_ids(gold) if gold else []

    def forward_chunk(self, chunk):
        specs = list(chunk.specs)
        if chunk.decode_rows is not None:
            decode = chunk.decode_rows
            tokens = chunk.input_ids[-len(decode):].tolist()
            for begin, end in zip(decode.sequence_offsets[:-1],
                                  decode.sequence_offsets[1:]):
                specs.append(dict(key=chunk.answer_keys[decode.slots[begin]],
                                  suffixes=[tokens[begin:end]], read_all_rows=True,
                                  f=int(decode.positions[begin]),
                                  write_suffix_tokens=int(end - begin)))
        rows = []
        for spec in specs:
            key = spec["key"]
            if key not in self.gold:
                parent = next(candidate for candidate in self.gold
                              if candidate[1] == key[1])
                self.gold[key] = self.gold[parent]
                self.prefix_len[key] = self.prefix_len[parent]
            if spec.get("prefix") is not None:
                self.prefix_len[key] = spec["f"]
                if key in self.documents:
                    self.reads[key] += 1
            for index, suffix in enumerate(spec["suffixes"]):
                suffix = list(map(int, suffix))
                read_all = bool(spec.get("read_all_rows"))
                self.packed.append((key, suffix, spec.get("write_suffix_tokens", 0),
                                    read_all, spec["f"]))
                held = self.kv.get(key, [])[:spec["f"] - self.prefix_len[key]]
                path = held + suffix
                count = (spec["read_rows"][index] if spec.get("read_rows") is not None
                         else len(suffix) if read_all else 1)
                for end in range(len(path) - count + 1, len(path) + 1):
                    values = self._row_after(key, path[:end])
                    if self.cue_token not in path[:end]:
                        rows.append(np.full(len(self.tables.vocab), values[0]))
                        continue
                    logits = np.full(len(self.tables.vocab), -30.0, dtype=np.float32)
                    for token, value in zip(values[1:1 + self.k], values[1 + self.k:]):
                        logits[int(token)] = max(logits[int(token)], value)
                    logits[self.tables.stop_ids[0]] = values[0]
                    rows.append(logits)
                if spec.get("write_suffix_tokens", 0):
                    self.kv[key] = path
        return tensor_torch.tensor(np.stack(rows), dtype=tensor_torch.float32)

    def _row_after(self, key, path):
        if path[-1] == self.cue_token:
            return self._cue_row(key)
        if self.cue_token in path:
            cue = len(path) - 1 - path[::-1].index(self.cue_token)
            return self._answer_row(key, path[cue + 1:])
        if key in self.filter_truth:
            # a filter's question: the row's first value is the answer
            row = self._cue({}, -9.0)
            row[:] = float(self.filter_truth[key])
            return row
        raise AssertionError(f"unexpected request {path}")

    def _cue_row(self, key):
        tables = self.tables
        gold = self._gold_ids(key)
        lps = {}
        if gold:
            # the model writes the first word with its space after the
            # cue, and every other document word is far behind
            first = same_word(tables.vocab[gold[0]]) or gold[0]
            for index in tables.variants.get(first, [gold[0]]):
                lps[index] = -0.5 if tables.vocab[index].startswith("Ġ") else -0.9
            for index in tables.variants.get("the", []):
                lps[index] = -2.0
            return self._cue(lps, -6.0)
        # without an answer the model leaves the phrase empty
        return self._cue({}, -0.2)

    def _answer_row(self, key, written):
        """The answer's next token while the writing fits the gold, then the stop."""
        tables = self.tables
        gold = self._gold_ids(key)
        keys = [same_word(tables.vocab[t]) or t for t in gold]
        fed = [same_word(tables.vocab[t]) or t for t in written]
        if fed == keys:
            return self._cue({}, -0.1)
        if fed == keys[:len(fed)]:
            return self._cue({gold[len(fed)]: -0.2}, -4.0)
        return self._cue({}, -0.5)


def _state(model, chunk_tokens=400, recurrent=False):
    arena = make_arena(Setup(path="unified", pages=256,
                             state_slots=24 if recurrent else 0))
    return QueryExecutionState(
        loaded_model=LoadedModelState(
            model_spec=QWEN3_4B_FP8, arena=arena,
            pipeline=fake_pipeline(forward_chunk=model.forward_chunk),
            input_staging=None,
            model=SimpleNamespace(quail_full_head=tensor_torch.eye(len(model.tables.vocab)))),
        torch=events_torch(0), chunk_tokens=chunk_tokens, answer_rows=object(),
        async_answers=fake_readout())


def _snapshot(part):
    for d, document in enumerate(part.extraction.documents):
        document.answers = []
        for q in range(len(part.nodes)):
            slot = part.slots[q, d]
            index = part.first + q
            values, spans = part.answers(q, [d])
            written = part.decoder.answers(index, [slot])[0].tolist()
            document.answers.append(SimpleNamespace(
                written=written, steps=int(part.decoder.steps[index][slot]),
                done=part.decoder.lengths[index][slot] >= 0,
                answer=values[0], span=spans[0], none_wins=not written))
    return SimpleNamespace(**vars(part.extraction), counts=lambda q: part.counts[q])


def _extract_spec(monkeypatch, text, gold, spec=None, vocabulary=None, recurrent=False):
    if spec is None:
        spec, vocabulary = make_spec(text)
    model = ScriptedModel(vocabulary, {}, {}, 4)
    state = _state(model, chunk_tokens=2000, recurrent=recurrent)
    state.loaded_model.vocabulary = vocabulary
    seen = []
    original = ExtractPart.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for document in self.extraction.documents:
            model.documents[document.key] = text
            model.gold[document.key] = gold
        seen.append(self)

    monkeypatch.setattr(ExtractPart, "__init__", initialize)
    node = AiExtract(node_id="extract", backend_name="quail", model=QWEN3_4B_FP8.name,
                     spec=spec)
    execute_extractions(state, [node], {
        "document_ids": [0], "documents": {"c": [vocabulary.encode_ids(text)]},
        "texts": {"c": [text]}})
    return _snapshot(seen[-1]).documents[0].answers[0], model


DOCS = [LEASE, TERM, NOTHING]
GOLD = ["March 3, 2027", "five years", None]


def _setup(monkeypatch, documents=None, token_documents=None,
           share_prefixes=False, document_ids=None, gpu_timing=False,
           policy="fixed", recurrent=False):
    """A state, its scripted model, and a node builder for one spec name."""
    spec, tables = make_spec()
    spec = replace(spec, draft_policy=policy, round_cost_per_token=0.05)
    documents = DOCS if documents is None else documents
    model = ScriptedModel(tables, {}, {}, 4)
    state = _state(model, recurrent=recurrent)
    state.loaded_model.vocabulary = tables
    tokens = {"c": ([tables.encode_ids(text) for text in documents]
                    if token_documents is None else token_documents)}
    stages_seen = {}
    original = ExtractPart.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for document in self.extraction.documents:
            model.documents[document.key] = document.text
            model.gold[document.key] = GOLD[documents.index(document.text)]
        stages_seen["stages"] = self

    monkeypatch.setattr(ExtractPart, "__init__", init)

    def node(name):
        return AiExtract(node_id=f"ai-extract:{name}", backend_name="quail",
                         model="qwen3-4b-fp8", spec=replace(
                             spec, name=name, share_prefixes=share_prefixes))

    inputs = {"document_ids": (list(range(len(documents))) if document_ids is None
                               else document_ids), "documents": tokens,
              "texts": {"c": documents}, "gpu_timing": gpu_timing}

    def run(name="ends", keep_kv=False):
        result = execute_extractions(
            state, [replace(node(name), keep_kv=keep_kv)], inputs)[node(name).node_id]
        return result, _snapshot(stages_seen["stages"])

    def run_pipeline(*names):
        results = execute_extractions(state, [node(name) for name in names],
                                      inputs)
        return results, _snapshot(stages_seen["stages"])

    return state, model, run, run_pipeline


def test_extraction_borrows_prefix_pages_and_keeps_decode_kv(monkeypatch):
    documents = [LEASE, LEASE, NOTHING]
    _, _, plain, _ = _setup(monkeypatch, documents=documents)
    unshared, _ = plain()
    _, model, shared, _ = _setup(
        monkeypatch, documents=documents, share_prefixes=True)
    result, stages = shared()
    assert result.outputs["rows:c"].table().equals(unshared.outputs["rows:c"].table())
    borrowed = result.metrics.extension["borrowed_prefix_tokens"]
    assert borrowed > 0
    assert result.metrics.fresh_tokens == unshared.metrics.fresh_tokens - borrowed
    for document in stages.documents:
        rounds = [entry for entry in model.packed if entry[0] == document.key]
        if document.answers[0].answer is not None:
            assert len(rounds) > 1
        assert model.reads[document.key] <= 1
        assert all(entry[2] > 0 for entry in rounds)


@pytest.mark.parametrize("gpu_timing,recurrent", [(True, True)])
def test_a_pipeline_of_extractions_reads_each_document_once(
        monkeypatch, gpu_timing, recurrent):
    state, model, _, run_pipeline = _setup(
        monkeypatch, gpu_timing=gpu_timing, recurrent=recurrent)
    timing_calls = []
    event = state.torch.cuda.Event

    def timed_event(**kwargs):
        timing_calls.append("event" if kwargs.get("enable_timing") else "copy")
        return event(**kwargs)

    monkeypatch.setattr(state.torch.cuda, "Event", timed_event)
    monkeypatch.setattr(state.torch.cuda, "synchronize",
                        lambda: timing_calls.append("sync"))
    results, stages = run_pipeline("ends", "starts")
    first, second = results["ai-extract:ends"], results["ai-extract:starts"]
    assert second.outputs["rows:c"].column("starts").to_pylist() == \
        first.outputs["rows:c"].column("ends").to_pylist()
    assert second.outputs["rows:c"].column("starts_span").to_pylist() == \
        first.outputs["rows:c"].column("ends_span").to_pylist()
    assert dict(model.reads) == {document.key: 1 for document in stages.documents}
    # Whole pages are shared; each branch recomputes the partial document tail.
    prefixes = sum(len(document.prefix) for document in stages.documents)
    fed = [stages.counts(q)["suffix_tokens"] for q in range(2)]
    borrowed = first.metrics.extension["borrowed_prefix_tokens"]
    assert first.metrics.fresh_tokens == 2 * prefixes - borrowed + fed[0]
    assert second.metrics.fresh_tokens == fed[1] > 0
    assert (first.metrics.gpu_s > 0) == gpu_timing
    assert (second.metrics.gpu_s > 0) == gpu_timing
    if gpu_timing:
        assert "event" in timing_calls and timing_calls.count("sync") == 1
    else:
        assert set(timing_calls) == {"copy"}


@pytest.mark.parametrize("gpus", [2])
def test_distributed_extraction_preserves_spans_order_and_kv(monkeypatch, gpus):
    import pickle

    state, model, _, run_pipeline = _setup(monkeypatch, share_prefixes=True)
    expected, _ = run_pipeline("ends", "starts")
    spec, tables = make_spec()
    nodes = [Scan(node_id="input:c", alias="c", input_id="c")]
    for name in ("ends", "starts"):
        previous = nodes[-1]
        nodes.append(AiExtract(
            node_id=f"ai-extract:{name}", backend_name="quail", model="qwen3-4b-fp8",
            inputs=input_ports((PortRef(previous.node_id, "rows:c"),)),
            spec=replace(spec, name=name, share_prefixes=True),
            keep_kv=name == "starts"))
    graph = PhysicalGraph(tuple(nodes), PortRef(nodes[-1].node_id, "rows:c"))
    registry = built_in_registry()
    children = []
    for _ in range(gpus):
        local = _state(model)
        torch, arena = local.torch, local.loaded_model.arena
        pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
        execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
        execution.bind_loaded_model(model=local.loaded_model.model,
                                    arena=arena, pipeline=pipeline)
        execution.loaded_model.vocabulary = tables
        execution.loaded_model.input_staging = None
        execution.bind_query(torch=torch, answer_rows=object(),
                             chunk_tokens=state.chunk_tokens,
                             async_answers=fake_readout())
        children.append(ChildState(gpu_index=0, registry=registry, gpu=SimpleNamespace(
            execution=execution, torch=torch, arena=arena, pipeline=pipeline,
            spec=MODELS["qwen3-4b-fp8"])))
    payload = {
        "model": "qwen3-4b-fp8", "chunk_tokens": state.chunk_tokens,
        "true_ids": [1], "false_ids": [2],
        "documents": {"c": tables.document_batch(DOCS, [None] * len(DOCS))},
        "texts": {"c": pa.array(DOCS)},
        "physical_plan": plan_envelope(
            backend="quail", model="qwen3-4b-fp8", device="h100-sxm",
            gpus=gpus, graph=graph, codecs=registry.codecs),
    }
    rounds = []

    def round_fn(kind, subs):
        assert kind == "execute"
        subs = pickle.loads(pickle.dumps(subs))
        rounds.append(subs)
        return [_child_execute(child, sub) for child, sub in zip(children, subs)]

    result = execute_distributed_graph(
        payload, graph, gpus, round_fn, DEVICES["h100-sxm"], registry.runtimes,
        registry)
    assert len(rounds) == 1
    # every worker ran the whole pipeline over its own documents
    shards = [sub["inputs"]["ai-extract:ends"]["input:0"].ids["c"].to_pylist()
              for sub in rounds[0]]
    assert sorted(document for shard in shards for document in shard) == [0, 1, 2]
    assert all(sub["pipeline"] and sub["final"] for sub in rounds[0])
    for node in nodes[1:]:
        actual = result["_outputs"][PortRef(node.node_id, "rows:c")]
        assert actual.equals(expected[node.node_id].outputs["rows:c"].table())
    assert result["fresh_tokens"] == sum(
        item.metrics.fresh_tokens for item in expected.values())
    assert result["evaluated_documents"] == 2 * len(DOCS)
    assert result["backend_metrics"]["inferences"][0]["none"] == 1
    assert all(not child.gpu.arena.accounting.owned for child in children)


def test_an_extraction_after_a_filter_asks_the_filtered_documents_once(
        monkeypatch):
    # the filter writes each document under the query's shared head;
    # the plain extraction continues its pipeline, so a document is
    # packed once and the extraction feeds only its question
    spec, tables = make_spec()
    plain = spec
    head = list(plain.prompt_token_parts[0])
    model = ScriptedModel(tables, {}, {}, 4)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
        model.filter_truth[("c", index)] = index != 2
    question = tables.encode_ids(" question")
    graph = PhysicalGraph(
        (
            Scan(node_id="input:c", alias="c", input_id="c"),
            AiFilter(
                node_id="ai_filter:c",
                inputs=input_ports((PortRef("input:c", "rows:c"),)),
                alias="c",
                arena_writes=True,
                question_token_ids=tuple(question),
                written_position=0,
                question_tokens=len(question),
                preamble_tokens=len(head),
                document_head=tuple(head),
                selectivity=0.7,
                expected_rows=2.0,
            ),
            AiExtract(
                node_id="ai-extract:ends",
                inputs=input_ports((PortRef("ai_filter:c", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=plain,
            ),
            AiExtract(
                node_id="ai-extract:starts",
                inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=replace(plain, name="starts"),
            ),
        ),
        PortRef("ai-extract:starts", "rows:c"),
    )
    local = _state(model)
    torch, arena = local.torch, local.loaded_model.arena
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=local.loaded_model.model,
                                    arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = None
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=fake_readout(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"],
             "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}, "gpu_timing": True}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    table = report["_outputs"][PortRef("ai-extract:ends", "rows:c")].to_pydict()
    assert table["c"] == [0, 1]
    assert table["ends"] == ["March 3, 2027", "five years"]
    # every document was packed once: the extraction's tail followed
    # the filter's question over the resident KV
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    metrics = report["node_metrics"]
    documents = sum(len(head) + len(tokens) for tokens in state["documents"]["c"])
    assert metrics["ai-extract:ends"]["fresh_tokens"] < documents
    assert metrics["ai_filter:c"]["fresh_tokens"] >= documents
    assert metrics["ai-extract:ends"]["evaluated_documents"] == 2
    later = report["_outputs"][PortRef("ai-extract:starts", "rows:c")].to_pydict()
    assert later["starts"] == table["ends"]
    assert later["starts_span"] == table["ends_span"]
    assert metrics["ai-extract:ends"]["gpu_s"] > 0
    assert metrics["ai-extract:starts"]["gpu_s"] > 0
    assert not arena.accounting.owned


def test_a_filter_on_answers_drops_documents_before_the_next_question(
        monkeypatch):
    # the second question is asked of the documents whose first answer
    # passed, over the KV the first question left; the dropped
    # document is packed once and its KV freed
    from quail.physical import Filter
    from quail.physical.pipelines import build_pipelines

    spec, tables = make_spec()
    model = ScriptedModel(tables, {}, {}, 4)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiExtract(node_id="ai-extract:ends",
                  inputs=input_ports((PortRef("input:c", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(node_id="filter:c:0",
               inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
               predicate=Not(RegularPredicate(
                   ColumnRef("c", "contracts", "ends"), "like", "five%")),
               aliases=("c",), written_position=0),
        AiExtract(node_id="ai-extract:starts",
                  inputs=input_ports((PortRef("filter:c:0", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8",
                  spec=replace(spec, name="starts")),
        Filter(node_id="filter:c:1",
               # a condition on two extractions reads each as the
               # document settles: starts from the part before it, ends
               # by document ID from the port that still carries it
               inputs=input_ports((PortRef("ai-extract:starts", "rows:c"),
                                   PortRef("filter:c:0", "rows:c"))),
               predicate=Or((
                   RegularPredicate(ColumnRef("c", "contracts", "starts"), "is null"),
                   And((RegularPredicate(ColumnRef("c", "contracts", "ends"),
                                         "like", "March%"),
                        RegularPredicate(
                            GetField(ColumnRef("c", "contracts", "starts_span"),
                                     "start"), "<", 40))))),
               aliases=("c",), written_position=1),
    ), PortRef("filter:c:1", "rows:c"))
    assert [member.node_id for member in build_pipelines(graph)["ai-extract:ends"]
            ] == ["ai-extract:ends", "filter:c:0", "ai-extract:starts", "filter:c:1"]
    local = _state(model)
    torch, arena = local.torch, local.loaded_model.arena
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=local.loaded_model.model,
                                    arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = None
    execution.bind_query(torch=torch, answer_rows=object(), chunk_tokens=400,
                         async_answers=fake_readout())
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    outputs = report["_outputs"]
    first = outputs[PortRef("ai-extract:ends", "rows:c")].to_pydict()
    assert first["ends"] == ["March 3, 2027", "five years", None]
    kept = outputs[PortRef("filter:c:0", "rows:c")].to_pydict()
    assert kept["c"] == [0] and kept["ends"] == ["March 3, 2027"]
    # the second question was asked of the survivor only
    second = outputs[PortRef("ai-extract:starts", "rows:c")].to_pydict()
    assert second == {"c": [0], "starts": ["March 3, 2027"],
                      "starts_span": [first["ends_span"][0]]}
    assert first["ends_span"][0] == {"start": 37, "end": 50}
    assert outputs[PortRef("filter:c:1", "rows:c")].to_pydict()["c"] == [0]
    assert outputs[PortRef("filter:c:1", "rows:c")].column("c").to_pylist() == [0]
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    metrics = report["node_metrics"]
    assert metrics["ai-extract:starts"]["evaluated_documents"] == 1
    assert metrics["filter:c:0"]["output_rows"] == 1
    assert not arena.accounting.owned


def test_a_filter_after_an_extraction_rewinds_the_kv_past_the_answer(
        monkeypatch):
    # the filter's question follows the document in KV, in place of
    # the answer the extraction wrote; the document is packed once
    from quail.physical import Filter
    from quail.physical.pipelines import build_pipelines

    spec, tables = make_spec()
    head = list(spec.prompt_token_parts[0])
    model = ScriptedModel(tables, {}, {}, 4)
    for index, text in enumerate(DOCS):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = GOLD[index]
        model.filter_truth[("c", index)] = index != 1
    question = tables.encode_ids(" question")
    graph = PhysicalGraph(
        (
            Scan(node_id="input:c", alias="c", input_id="c"),
            AiExtract(
                node_id="ai-extract:ends",
                inputs=input_ports((PortRef("input:c", "rows:c"),)),
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=spec,
            ),
            Filter(
                node_id="filter:c:1",
                inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
                predicate=RegularPredicate(
                    ColumnRef("c", "contracts", "ends"), "is not null"
                ),
                aliases=("c",),
                written_position=1,
            ),
            AiFilter(
                node_id="ai_filter:c",
                inputs=input_ports((PortRef("filter:c:1", "rows:c"),)),
                alias="c",
                arena_writes=True,
                question_token_ids=tuple(question),
                written_position=0,
                question_tokens=len(question),
                preamble_tokens=len(head),
                document_head=tuple(head),
                selectivity=0.7,
                expected_rows=2.0,
            ),
        ),
        PortRef("ai_filter:c", "rows:c"),
    )
    assert [member.node_id for member in build_pipelines(graph)["ai-extract:ends"]
            ] == ["ai-extract:ends", "filter:c:1", "ai_filter:c"]
    local = _state(model)
    torch, arena = local.torch, local.loaded_model.arena
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=local.loaded_model.model,
                                    arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = None
    execution.bind_query(
        torch=torch, answer_rows=object(), chunk_tokens=400,
        async_answers=fake_readout(
            submit=lambda rows: [float(np.asarray(r).ravel()[0]) for r in rows],
            result=lambda v: v, dtype=None))
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": 400,
             "documents": {"c": [tables.encode_ids(text) for text in DOCS]},
             "texts": {"c": list(DOCS)}}
    report = execute_single_graph(
        state, {"filter_limit": None, "retention": {}}, graph)
    outputs = report["_outputs"]
    assert outputs[PortRef("ai-extract:ends", "rows:c")].to_pydict()["ends"] == [
        "March 3, 2027", "five years", None]
    # the test drops the third document; the filter asks the other two
    # and keeps the first
    assert outputs[PortRef("filter:c:1", "rows:c")].column("c").to_pylist() == [0, 1]
    assert outputs[PortRef("ai_filter:c", "rows:c")].column("c").to_pylist() == [0]
    assert dict(model.reads) == {("c", 0): 1, ("c", 1): 1, ("c", 2): 1}
    # the filter's question was fed right after the document, in place
    # of the extraction's cue and answer
    fed = [(key, suffix, f) for key, suffix, kept, _, f in model.packed
           if suffix == question]
    assert [(key, f) for key, _, f in fed] == [
        (("c", index), len(head) + len(state["documents"]["c"][index]))
        for index in (0, 1)]
    assert report["node_metrics"]["ai_filter:c"]["evaluated_documents"] == 2
    assert not arena.accounting.owned


@pytest.mark.parametrize("limit", [2])
def test_a_limit_ends_the_pipeline_once_enough_documents_pass(monkeypatch, limit):
    # the result's limit stops the run after that many documents
    # passed every operator, the trailing test included; the documents
    # still queued are not read
    from quail.physical import Filter

    spec, tables = make_spec()
    model = ScriptedModel(tables, {}, {}, 4)
    # the first four documents have no answer, so the test drops them;
    # some of the rest are still queued when enough have passed
    texts = [NOTHING] * 4 + [LEASE] * 4 + [TERM] * 4
    gold = [None] * 4 + [GOLD[0]] * 4 + [GOLD[1]] * 4
    for index, text in enumerate(texts):
        model.documents[("c", index)] = text
        model.gold[("c", index)] = gold[index]
    graph = PhysicalGraph((
        Scan(node_id="input:c", alias="c", input_id="c"),
        AiExtract(node_id="ai-extract:ends",
                  inputs=input_ports((PortRef("input:c", "rows:c"),)),
                  backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(node_id="filter:c:0",
               inputs=input_ports((PortRef("ai-extract:ends", "rows:c"),)),
               predicate=RegularPredicate(
                   ColumnRef("c", "contracts", "ends"), "is not null"),
               aliases=("c",), written_position=0),
    ), PortRef("filter:c:0", "rows:c"))
    local = _state(model)
    torch, arena = local.torch, local.loaded_model.arena
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(model=MODELS["qwen3-4b-fp8"]))
    execution.bind_loaded_model(model=local.loaded_model.model,
                                    arena=arena, pipeline=pipeline)
    execution.loaded_model.vocabulary = tables
    execution.loaded_model.input_staging = None
    documents = [tables.encode_ids(text) for text in texts]
    # a chunk holds about one document and its question
    chunk = max(len(document) for document in documents) + len(
        spec.prompt_token_parts[0]) + len(spec.prompt_token_parts[1]) + 8
    execution.bind_query(torch=torch, answer_rows=object(), chunk_tokens=chunk,
                         async_answers=fake_readout())
    state = {"gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                                    execution=execution, spec=MODELS["qwen3-4b-fp8"]),
             "runtimes": built_in_registry().runtimes,
             "device": DEVICES["h100-sxm"], "chunk_tokens": chunk,
             "documents": {"c": documents}, "texts": {"c": list(texts)}}
    report = execute_single_graph(
        state, {"filter_limit": limit, "retention": {}}, graph)
    outputs = report["_outputs"]
    kept = outputs[PortRef("filter:c:0", "rows:c")].to_pydict()
    # the documents that passed before the run ended have their whole
    # answers; the queued ones are never read
    assert limit <= len(kept["c"]) < len(texts)
    assert all(answer is not None for answer in kept["ends"])
    assert kept["ends"][:limit] == [gold[index] for index in kept["c"][:limit]]
    assert sum(model.reads.values()) < len(texts)
    assert not arena.accounting.owned
