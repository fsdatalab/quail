"""A hybrid run under checkpoint-slot pressure, through the real stage executor.

The model and the chunk packer are fakes; planning, admission, slot
allocation, chunk splitting, and settling are the real code.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fakes import bare_arena, fake_pipeline, fake_torch

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.chunk import Suffixes
from quail.backends.quail.executor.gdn import plan_phases
from quail.backends.quail.executor.stages import (
    Stage,
    checkpoint_slot_limit,
    run_stages,
)


class _Rows(dict):
    def pop(self, key, *default):
        return super().pop(key, None)


def _arena(slots, pages=100_000):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting = PageArena(pages, 16, n_slots=slots)
    arena._rows = _Rows()
    arena._capacity_rows = _Rows()
    return arena


def _pack(torch, arena, specs, **options):
    groups, rows, fresh = [], 0, []
    for spec in specs:
        is_fresh = spec.get("prefix") is not None
        length = len(spec["prefix"]) if is_fresh else 0
        row0 = rows
        rows += length
        end = rows
        suffixes = spec["suffixes"]
        if not isinstance(suffixes, Suffixes):
            suffixes = Suffixes.of(suffixes)
        ends = np.cumsum(suffixes.lengths)
        begins = ends - suffixes.lengths
        groups.append(dict(
            key=spec["key"], fresh=is_fresh, f=spec["f"], row0=row0,
            start=spec.get("start", 0), prefix_end=end,
            row1=rows + (int(ends[-1]) if len(ends) else 0),
            suffix_spans=(rows + begins, rows + ends), canvas=False,
            keeps=bool(spec.get("write_suffix_tokens")),
            kept=int(spec.get("write_suffix_tokens") or 0),
            borrow_key=spec.get("borrow_key"),
            share_points=spec.get("share_points", ())))
        rows += int(suffixes.lengths.sum())
        if is_fresh:
            fresh.append(spec["key"])
    phases = plan_phases(arena, groups, lambda name, values, dtype: np.asarray(values))
    return SimpleNamespace(
        specs=specs, tokens=rows, temporary_keys=(), fresh_keys=tuple(fresh),
        rows_per_answer=(), meta={"gdn": phases},
        n=sum(len(spec["suffixes"]) for spec in specs))


class _Readout:
    dtype = None

    def submit(self, normed, rows_per_answer=None):
        return normed

    def result(self, handle):
        return handle


def _run(arena, keys, stages, budget, monkeypatch, prefix_tree=None,
         prefixes=None):
    # plan_phases names torch dtypes and never runs a torch operation
    stub = ModuleType("torch")
    stub.int64 = stub.int32 = None
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setattr(chunk_mod, "pack_chunk", _pack)
    pipeline = fake_pipeline(canvas_ids=(), canvas_answer_row=0,
                             tree_attention=False)
    pipeline.forward_chunk = lambda chunk: [True] * chunk.n

    lengths = [len(p) for p in (prefixes or [[1] * 64 for _ in keys])]

    def settled(anchor, survivors, row):
        if arena.is_resident(keys[anchor]):
            arena.retain(keys[anchor], lengths[anchor])

    return run_stages(
        fake_torch(), arena, pipeline, stages,
        prefixes or [[1] * 64 for _ in keys],
        budget, anchor_keys=keys, on_settled=settled,
        attention_mode="unified", prefix_tree=prefix_tree)


def _stages(count=2, frame=(7, 7), question=3):
    return [Stage(suffixes=[[100 + j] * question], readout=_Readout(),
                  frame=list(frame), single=True) for j in range(count)]


def _retained_run(slots, retained, documents, stages, monkeypatch):
    arena = _arena(slots)
    keys = [("d", i) for i in range(documents)]
    for key in keys[documents - retained:]:
        arena.alloc(key, 64)
        arena.accounting.alloc_slot(key)
        arena.record_checkpoint(key, 64)
        arena.retain(key, 64)
    _run(arena, keys, stages, 72 * 12, monkeypatch)
    return arena


def test_slot_limit_leaves_out_the_slots_pinned_residents_hold():
    assert checkpoint_slot_limit(40, 2, False) == 20
    assert checkpoint_slot_limit(40, 2, False, held=30) == 5
    assert checkpoint_slot_limit(40, 1, True, held=39) == 1


@pytest.mark.parametrize("retained", [10, 30, 36, 38, 39])
@pytest.mark.parametrize("stage_count", [1, 2])
def test_retained_documents_do_not_exhaust_the_checkpoint_slots(
        retained, stage_count, monkeypatch):
    # 40 slots, 160 documents, `retained` of them resident from an
    # earlier operator, and a frame that needs a second slot each
    arena = _retained_run(40, retained, 160, _stages(stage_count),
                          monkeypatch)
    assert arena.accounting.free_slots


def test_the_arena_counts_the_slots_a_set_of_keys_holds():
    arena = _arena(8)
    first, second = ("d", 0), ("d", 1)
    for key in (first, second):
        arena.alloc(key, 64)
        arena.accounting.alloc_slot(key)
    arena.accounting.alloc_extension(first)
    arena.accounting.alloc_share(second, 32)
    assert arena.slots_held([first]) == 2
    assert arena.slots_held([second]) == 2
    assert arena.slots_held([first, second]) == 4
    assert arena.slots_held([]) == 0


def _group(key, f, row0, prefix_end, *, fresh, kept=0, slen=2):
    begins = np.array([prefix_end])
    return dict(key=key, fresh=fresh, f=f, start=0, row0=row0,
                prefix_end=prefix_end, row1=prefix_end + slen,
                suffix_spans=(begins, begins + slen), kept=kept,
                keeps=bool(kept), borrow_key=None, share_points=())


def _plan(arena, groups, monkeypatch):
    stub = ModuleType("torch")
    stub.int64 = stub.int32 = None
    monkeypatch.setitem(sys.modules, "torch", stub)
    return plan_phases(arena, groups, lambda name, values, dtype: np.asarray(values))


def test_a_question_cannot_share_a_chunk_with_an_in_place_advance_of_its_slot(
        monkeypatch):
    arena = _arena(4)
    arena.alloc("a", 64)
    # the document ends at 32 tokens and its frame of 4 advances a second slot
    _plan(arena, [_group("a", 32, 0, 32, fresh=True, kept=4, slen=4)],
          monkeypatch)
    assert arena.extension("a")[1] == 36
    question = _group("a", 36, 0, 0, fresh=False, slen=2)
    advance = _group("a", 36, 0, 0, fresh=False, kept=3, slen=3)
    with pytest.raises(ValueError, match="advances a checkpoint in place"):
        _plan(arena, [question, advance], monkeypatch)
    # a chunk with the advance alone, or the question alone, plans
    _plan(arena, [advance], monkeypatch)
    assert arena.extension("a")[1] == 39


def test_an_empty_document_has_no_state_to_checkpoint(monkeypatch):
    arena = _arena(2)
    arena.alloc(("d", 0), 16)
    with pytest.raises(ValueError, match="empty document"):
        _plan(arena, [_group(("d", 0), 0, 0, 0, fresh=True, slen=3)],
              monkeypatch)
    # nothing was allocated before the refusal
    assert ("d", 0) not in arena.accounting.slots
    assert len(arena.accounting.free_slots) == 2


def test_a_query_that_needs_more_slots_than_the_pool_holds_is_refused_early(
        monkeypatch):
    # a stage frame needs a second slot, and the pool has one
    arena = _arena(1)
    keys = [("d", i) for i in range(4)]
    with pytest.raises(ValueError, match="1 checkpoint slots"):
        _run(arena, keys, _stages(2), 72 * 12, monkeypatch)


def test_a_pool_that_holds_no_checkpoint_beside_a_document_is_refused():
    from quail.cost import budgets
    from quail.specs import H100_SXM, QWEN3_5_27B_FP8

    # a checkpoint is 146.8 pages and the pool is 12,108, so a document
    # of 10 million tokens leaves room for none
    with pytest.raises(ValueError, match="holds no checkpoint"):
        budgets.hybrid_arena_pages(QWEN3_5_27B_FP8, H100_SXM, 61_680, 1e7)
    assert budgets.hybrid_arena_pages(
        QWEN3_5_27B_FP8, H100_SXM, 61_680, 1_000) > 0


def test_slots_per_document_count_the_frame_and_every_share_length():
    from quail.backends.quail.executor.stages import slots_per_document

    assert slots_per_document(1, 0) == 1
    assert slots_per_document(2, 0) == 2
    # a written frame extends the document's checkpoint, even in one stage
    assert slots_per_document(1, 0, frames=True) == 2
    # borrowers at 16, 32, and 48 tokens each start from their own slot
    assert slots_per_document(1, 3) == 4
    assert slots_per_document(2, 3, frames=True) == 5


@pytest.mark.parametrize("slots", [20, 24, 30, 40])
def test_a_single_stage_with_a_frame_gets_its_second_slot(slots, monkeypatch):
    arena = _arena(slots, pages=4_000)
    keys = [("d", i) for i in range(200)]
    prefixes = [[i + 1] * 64 for i in range(200)]
    _run(arena, keys, _stages(1), 72 * 12, monkeypatch, prefixes=prefixes)


def test_a_retained_parent_is_not_evicted_by_its_childs_chunk(monkeypatch):
    # 6 slots hold the retained parent and 5 fresh documents; the child
    # starts from the parent's state at 64 tokens and is planned last
    slots, fresh = 6, 5
    keys = [("d", i) for i in range(2 + fresh)]
    arena = _arena(slots, pages=4_000)
    parent = keys[0]
    arena.alloc(parent, 64)
    arena.accounting.alloc_slot(parent)
    arena.record_checkpoint(parent, 64)
    arena.retain(parent, 64)
    prefixes = [[1] * 64, [1] * 64 + [2] * 16] + [
        [5 + i] * 64 for i in range(fresh)]
    tree = SimpleNamespace(
        parent=[None, 0] + [None] * fresh, shared=[0, 64] + [0] * fresh,
        order=[0, *range(2, 2 + fresh), 1], shared_tokens=64)
    stage = [Stage(suffixes=[[9, 9]], readout=_Readout(), single=True)]
    for budget in (100, 140, 200, 300, 600):
        run_arena = _arena(slots, pages=4_000)
        run_arena.alloc(parent, 64)
        run_arena.accounting.alloc_slot(parent)
        run_arena.record_checkpoint(parent, 64)
        run_arena.retain(parent, 64)
        _run(run_arena, keys, stage, budget, monkeypatch,
             prefix_tree=tree, prefixes=prefixes)


def test_a_failed_plan_leaves_the_extension_records_as_they_were(monkeypatch):
    arena = _arena(2)
    arena.alloc("a", 64)
    arena.alloc("b", 64)
    _plan(arena, [_group("a", 32, 0, 32, fresh=True, kept=4, slen=4)],
          monkeypatch)
    assert arena.extension("a")[1] == 36
    # the pool is full: a's frame advances, then a fresh document finds no slot
    arena.accounting.free_slots.clear()
    advance = _group("a", 36, 0, 0, fresh=False, kept=4, slen=4)
    fresh = _group("b", 32, 10, 42, fresh=True)
    with pytest.raises(Exception, match="no free checkpoint slot"):
        _plan(arena, [advance, fresh], monkeypatch)
    # the record is back at 36, so the retry asks at the length it holds
    assert arena.extension("a")[1] == 36
    _plan(arena, [advance], monkeypatch)
    assert arena.extension("a")[1] == 40


def _tree_run(slots, parents, children, monkeypatch, stages=1, frame=(),
              budget=1_000, spread=16):
    """Parents whose children share prefixes of increasing length."""
    docs, parent, shared = [], [], []
    for p in range(parents):
        base = [p + 1] * (spread * (children + 1))
        docs.append(base)
        parent.append(None)
        shared.append(0)
        index = len(docs) - 1
        for c in range(children):
            cut = spread * (c + 1)
            docs.append(base[:cut] + [100 + p * 10 + c] * 40)
            parent.append(index)
            shared.append(cut)
    firsts = [p * (children + 1) for p in range(parents)]
    rest = [i for i in range(len(docs)) if i not in firsts]
    tree = SimpleNamespace(parent=parent, shared=shared, order=firsts + rest,
                           shared_tokens=sum(shared))
    keys = [("t", i) for i in range(len(docs))]
    arena = _arena(slots, pages=100_000)
    if frame:
        run_stages_ = _stages(stages, frame=frame)
    else:
        run_stages_ = [Stage(suffixes=[[9, 9]], readout=_Readout(), single=True)
                       for _ in range(stages)]
    _run(arena, keys, run_stages_, budget, monkeypatch, prefix_tree=tree,
         prefixes=docs)


@pytest.mark.parametrize("slots", [3, 6, 10, 15, 30])
@pytest.mark.parametrize("children", [2, 6])
@pytest.mark.parametrize("stages,frame", [(1, ()), (1, (7, 7)), (2, (7, 7))])
def test_parents_with_many_share_lengths_run_in_a_small_pool(
        slots, children, stages, frame, monkeypatch):
    # a pool too small for every share point keeps the ones that save
    # the most, and the other borrowers compute their whole prefix
    _tree_run(slots, 6, children, monkeypatch, stages=stages, frame=frame)


def test_the_share_points_a_parent_keeps_are_the_ones_that_save_most():
    from quail.backends.quail.executor.stages import state_share_points

    parents = [None, 0, 0, 0, 0, 0]
    shared = [0, 16, 16, 32, 48, 48]
    assert state_share_points(parents, shared) == [(16, 32, 48), (), (), (),
                                                    (), ()]
    # 16 has 2 borrowers (32 tokens saved), 48 has 2 (96), 32 has 1 (32)
    assert state_share_points(parents, shared, limit=1)[0] == (48,)
    assert state_share_points(parents, shared, limit=2)[0] == (16, 48)
    assert state_share_points(parents, shared, limit=0)[0] == ()
