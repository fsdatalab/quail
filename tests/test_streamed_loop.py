"""The filter chain streamed into a join on a CPU arena, without torch."""

import random
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from quail.executor import loop
from quail.executor.arena import KVArena, PageArena
from quail.executor.attention import JOIN_ATTENTION
from quail.executor.pack import JoinAdmission

DOC = 1000       # document d packs tokens [DOC + d] * length
QUESTION = 2000  # stage s asks [QUESTION + s]
PARTNER = 3000   # partner i streams [PARTNER + i] * length
FRAME = 4000     # the join frame


def cpu_arena(pages):
    arena = KVArena.__new__(KVArena)
    arena.accounting = PageArena(pages, 16)
    arena._rows = {}
    arena._capacity_rows = {}
    arena._refresh_rows = lambda *args: None
    arena.reset_stats()

    def allocate(key, tokens, capacity_tokens=None):
        got = arena.accounting.alloc(key, tokens, capacity_tokens)
        if got is not None:
            arena._rows[key] = None
            arena._capacity_rows[key] = None
        return got

    arena.alloc = allocate
    return arena


class FakeModel:
    """Answer packed suffixes from planted truth tables."""

    def __init__(self, filter_truth, join_truth):
        self.filter_truth = filter_truth
        self.join_truth = join_truth
        self.launched = []      # ("filter" | "join", specs)

    def forward_chunk(self, chunk):
        bits = []
        kind = None
        for spec in chunk["specs"]:
            for suffix in spec["suffixes"]:
                head = suffix[0]
                if head >= FRAME:
                    bits.append(0)
                    kind = "join"
                elif head >= PARTNER:
                    bits.append(self.join_truth[spec["key"]][head - PARTNER])
                    kind = "join"
                else:
                    document = spec["key"][1]
                    bits.append(self.filter_truth[document][head - QUESTION])
                    kind = "filter"
            if kind == "join" and spec["prefix"] is not None:
                raise AssertionError("a streamed anchor packed its prefix")
        self.launched.append((kind, chunk["specs"]))
        return bits


def fake_torch():
    return SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(Event=lambda **kw: SimpleNamespace(
            record=lambda: None)))


def fake_pack(torch, arena, specs, **kw):
    tokens = sum(
        (len(spec["prefix"]) if spec["prefix"] is not None else 0)
        + sum(len(suffix) for suffix in spec["suffixes"])
        for spec in specs)
    return {"specs": specs, "tokens": tokens}


def run_streamed(monkeypatch, *, doc_lengths, filter_truth, partner_lengths,
                 join_truth, budget, pages, frame_tokens=3, stages=2):
    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    model = FakeModel(filter_truth, join_truth)
    pipeline = SimpleNamespace(attention_mode=JOIN_ATTENTION,
                               forward_chunk=model.forward_chunk)
    answers = SimpleNamespace(submit=lambda v: v, result=lambda v: v)
    arena = cpu_arena(pages)
    docs = [[DOC + d] * n for d, n in enumerate(doc_lengths)]
    questions = [[QUESTION + s] for s in range(stages)]
    keys = [("r", d) for d in range(len(docs))]
    frame = [FRAME] * frame_tokens
    suffixes = [[PARTNER + i] * n for i, n in enumerate(partner_lengths)]
    stream = loop.FilterStream(
        fake_torch(), arena, pipeline, answers, docs, questions, budget,
        arena_writes=True, arena_keys=keys, hold_survivors=True,
        hold_extra_tokens=len(frame))
    blocked_seen = []
    real_next = stream.next

    def counting_next(evict_retained=False):
        items, blocked = real_next(evict_retained=evict_retained)
        blocked_seen.append(blocked)
        return items, blocked

    stream.next = counting_next
    anchor_keys, anchor_prefixes = [], []
    settled = {}

    def anchor_done(local, row):
        settled[anchor_keys[local]] = list(row)
        arena.free_key(anchor_keys[local])

    join_answers, _, join_tokens = loop.run_join(
        fake_torch(), arena, pipeline, answers, anchor_prefixes,
        [suffixes], budget, stage_frames=[frame], anchor_keys=anchor_keys,
        anchor_done=anchor_done, anchor_source=stream)
    return dict(stream=stream, model=model, arena=arena,
                join_answers=join_answers, join_tokens=join_tokens,
                anchor_keys=anchor_keys, settled=settled,
                blocked=blocked_seen)


def expected_filter_rows(filter_truth):
    rows = {}
    for d, truth in enumerate(filter_truth):
        row = []
        for bit in truth:
            row.append(bit)
            if not bit:
                break
        rows[d] = row
    return rows


def test_streamed_filter_feeds_the_join_and_frees_everything(monkeypatch):
    rng = random.Random(7)
    n_docs, n_partners = 60, 5
    doc_lengths = [rng.randrange(20, 90) for _ in range(n_docs)]
    filter_truth = [[1 if rng.random() < 0.8 else 0, 1 if rng.random() < 0.7 else 0]
                    for _ in range(n_docs)]
    partner_lengths = [rng.randrange(5, 20) for _ in range(n_partners)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)]
                  for d in range(n_docs)}
    # the arena holds about a dozen documents: the filter blocks on
    # pages before the corpus is through, and the join has to drain
    out = run_streamed(
        monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
        partner_lengths=partner_lengths, join_truth=join_truth,
        budget=400, pages=80)

    stream, model = out["stream"], out["model"]
    assert stream.done
    assert stream.answers == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    assert sorted(stream.held) == survivors
    assert out["anchor_keys"] == [("r", d) for d in stream.held]
    # every survivor's row answered from the planted truth, in
    # admission order
    rows = out["join_answers"][0]
    assert sorted(rows) == list(range(len(survivors)))
    for local, key in enumerate(out["anchor_keys"]):
        assert rows[local] == join_truth[key]
        assert out["settled"][key] == join_truth[key]
    # nothing left in the arena: held keys were freed by anchor_done,
    # rejected documents by the chain
    assert not out["arena"].accounting.owned
    assert out["arena"].accounting.free_pages == 80
    # join chunks ran before the chain finished, and the chain reported
    # a page shortfall at least once
    kinds = [kind for kind, _ in model.launched]
    assert "join" in kinds[:-1] and kinds[-1] == "join"
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
    assert any(out["blocked"])
    # streamed anchors pack the frame and partner suffixes only
    anchors = sum(len(specs) for kind, specs in model.launched
                  if kind == "join")
    assert anchors > 0
    assert out["join_tokens"] == sum(
        3 + sum(partner_lengths) for _ in survivors)


def test_streamed_loop_random_shapes(monkeypatch):
    rng = random.Random(11)
    for _ in range(25):
        n_docs = rng.randrange(1, 40)
        n_partners = rng.randrange(0, 6)
        stages = rng.randrange(1, 4)
        doc_lengths = [rng.randrange(8, 120) for _ in range(n_docs)]
        filter_truth = [[1 if rng.random() < 0.7 else 0
                         for _ in range(stages)] for _ in range(n_docs)]
        partner_lengths = [rng.randrange(4, 30) for _ in range(n_partners)]
        join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                                 for _ in range(n_partners)]
                      for d in range(n_docs)}
        frame_tokens = rng.randrange(0, 20)
        budget = max(doc_lengths) + 1 + rng.randrange(0, 400)
        budget = max(budget, frame_tokens + max(partner_lengths, default=0))
        pages = max(-(-(max(doc_lengths) + max(1, frame_tokens)) // 16),
                    rng.randrange(6, 40))
        out = run_streamed(
            monkeypatch, doc_lengths=doc_lengths, filter_truth=filter_truth,
            partner_lengths=partner_lengths, join_truth=join_truth,
            budget=budget, pages=pages, frame_tokens=frame_tokens,
            stages=stages)
        stream = out["stream"]
        assert stream.done
        assert stream.answers == expected_filter_rows(filter_truth)
        survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
        assert sorted(stream.held) == survivors
        assert out["anchor_keys"] == [("r", d) for d in stream.held]
        assert not out["arena"].accounting.owned
        assert out["arena"].accounting.free_pages == pages
        if n_partners:
            rows = out["join_answers"][0]
            assert sorted(rows) == list(range(len(survivors)))
            for local, key in enumerate(out["anchor_keys"]):
                assert rows[local] == join_truth[key]
        else:
            assert out["join_answers"] == [{}]
            assert set(out["settled"]) == set(out["anchor_keys"])


def test_join_admission_admits_incrementally_and_prices_room():
    sched = JoinAdmission([], [[10, 10, 10]], 100, 50, 16,
                          frame_tokens=[5])
    assert sched.done()
    assert sched.buildable_tokens() == 0
    first = sched.admit(40, resident_pages=3)
    assert first == 0
    assert sched.buildable_tokens() == 5 + 30
    second = sched.admit(80)
    assert second == 1
    assert sched.buildable_tokens() == 100
    groups = sched.next_chunk(free_pages=50)
    assert groups[0] == (0, 0, 0, 3, False)
    assert sched.report(0, 0, 0, 3, [0, 1, 0]) == [("finished", 0)]
    # the resident anchor packed no prefix; the fresh one still waits
    # for chunk room and counts its prefix as buildable work
    assert sched.buildable_tokens() == 100
    with pytest.raises(ValueError):
        sched.admit(5000)


def test_streamed_edge_runs_through_the_quail_graph(monkeypatch):
    """A filter and a join with a streamed edge, through the real graph runtime."""
    from quail.backends.quail import QuailModelExecution
    from quail.backends.quail.graph import execute_single_graph
    from quail.builtins import built_in_registry
    from quail.physical import (
        AnchoredJoin,
        DocumentInput,
        FilterStage,
        JoinStage,
        PackedFilter,
        PhysicalGraph,
        PortRef,
    )
    from quail.physical.base import input_ports
    from quail.specs import DEVICES, MODELS

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    rng = random.Random(3)
    n_docs, n_partners = 14, 3
    docs = {
        "r": [[DOC + d] * rng.randrange(10, 40) for d in range(n_docs)],
        "p": [[PARTNER + i] * 35 for i in range(n_partners)],
    }
    filter_truth = [[1 if rng.random() < 0.8 else 0,
                     1 if rng.random() < 0.7 else 0] for _ in range(n_docs)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)]
                  for d in range(n_docs)}
    model = FakeModel(filter_truth, join_truth)
    torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(
            Event=lambda **kw: SimpleNamespace(record=lambda: None),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0))
    arena = cpu_arena(48)
    pipeline = SimpleNamespace(attention_mode=JOIN_ATTENTION,
                               forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v),
        chunk_tokens=120)
    registry = built_in_registry()
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution, "runtimes": registry.runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
    }
    scan_r = DocumentInput(node_id="input:r", alias="r", input_id="r")
    scan_p = DocumentInput(node_id="input:p", alias="p", input_id="p")
    chain = PackedFilter(
        node_id="filter:r",
        inputs=input_ports((PortRef("input:r", "ids:r"),)),
        alias="r", arena_writes=True,
        stages=(FilterStage(0, 1, 0, 0.8, n_docs),
                FilterStage(1, 1, 0, 0.7, n_docs * 0.8)),
        question_token_ids=((QUESTION,), (QUESTION + 1,)))
    join = AnchoredJoin(
        node_id="group:0", anchor="r", anchor_resident="filter",
        stream_anchor=True,
        inputs=input_ports((PortRef("filter:r", "ids:r"),
                            PortRef("input:p", "ids:p"))),
        stages=(JoinStage(
            written_pos=0, exec_idx=0, anchor="r", partners=("p",),
            semantics="full", selectivity=0.5, expected_tuples=1,
            anchor_frame_tokens=1, pair_tail_tokens=0,
            anchor_resident="filter", tuple_tokens=0,
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
            tail_token_ids=()),))
    graph = PhysicalGraph((scan_r, chain, scan_p, join),
                          PortRef("group:0", "ids:r"))
    assert join.streamed_inputs() == ("input:0",)

    result = execute_single_graph(
        state, {"filter_limit": None, "pre_ids": [], "retention": {}}, graph)

    assert result["filters"]["r"] == expected_filter_rows(filter_truth)
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    rows = result["joins"][0]
    assert sorted(rows["anchor_index"]) == survivors
    for local, document in enumerate(rows["anchor_index"]):
        assert rows["rows"][local] == join_truth[("r", document)]
    metrics = result["node_metrics"]
    assert metrics["filter:r"]["evaluated_documents"] == n_docs
    assert metrics["filter:r"]["fresh_tokens"] > 0
    assert metrics["group:0"]["kv_hits"] == len(survivors)
    assert metrics["group:0"]["kv_misses"] == 0
    assert result["regret_tokens"] == 0
    assert result["fresh_tokens"] == (
        metrics["filter:r"]["fresh_tokens"] + metrics["group:0"]["fresh_tokens"])
    assert result["kv_manager"]["join_anchor_hits"] == len(survivors)
    assert not arena.accounting.owned
    kinds = [kind for kind, _ in model.launched]
    assert kinds.index("join") < len(kinds) - 1 - kinds[::-1].index("filter")
