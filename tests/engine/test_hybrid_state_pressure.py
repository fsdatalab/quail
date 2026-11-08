"""A hybrid run under checkpoint-slot pressure, through the real stage executor.

The model and the chunk packer are fakes; planning, admission, slot
allocation, chunk splitting, and settling are the real code.
"""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from helpers.fakes import bare_arena, fake_pipeline, fake_torch

from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.chunk import Suffixes
from quail.backends.quail.executor.gdn import plan_phases
from quail.backends.quail.executor.stage_executor import StageExecutor
from quail.backends.quail.executor.stages import Stage


class _Rows(dict):
    def pop(self, key, *default):
        return super().pop(key, None)


def _arena(slots, pages=100_000):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting = PageArena(pages, 16, num_slots=slots)
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
        specs=specs, tokens=rows, layout=[], temporary_keys=(),
        fresh_keys=tuple(fresh),
        rows_per_answer=(), meta={"gdn": phases},
        n=sum(len(spec["suffixes"]) for spec in specs))


class _Readout:
    dtype = None
    trailing_rows = 1
    normalize = False

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

    return StageExecutor(
        fake_torch(), arena, pipeline, stages,
        prefixes or [[1] * 64 for _ in keys],
        budget, anchor_keys=keys, on_settled=settled,
        attention_mode="unified", prefix_tree=prefix_tree).run()


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


@pytest.mark.parametrize("retained,stage_count", [(39, 2)],
                         ids=["nearly-full-multistage"])
def test_retained_documents_do_not_exhaust_the_checkpoint_slots(
        retained, stage_count, monkeypatch):
    arena = _retained_run(40, retained, 160, _stages(stage_count),
                          monkeypatch)
    assert arena.accounting.free_slots


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
    return arena


@pytest.mark.parametrize("slots,children,stages,frame", [
    (10, 6, 2, (7, 7)),
], ids=["multistage-frame"])
def test_parents_with_many_share_lengths_run_in_a_small_pool(
        slots, children, stages, frame, monkeypatch):
    # a pool too small for every share point keeps the ones that save
    # the most, and the other borrowers compute their whole prefix
    _tree_run(slots, 6, children, monkeypatch, stages=stages, frame=frame)
