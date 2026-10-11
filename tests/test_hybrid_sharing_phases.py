"""Gated DeltaNet phases for documents that borrow a shared prefix."""

import sys
from types import ModuleType

import numpy as np
import pytest
from fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.gdn import plan_phases


@pytest.fixture
def torch_stub(monkeypatch):
    module = ModuleType("torch")
    module.int64 = module.int32 = None
    monkeypatch.setitem(sys.modules, "torch", module)


def arena_with_slots(slots=8):
    arena = bare_arena(KVArena.__new__(KVArena), 64)
    arena.accounting = PageArena(64, 16, n_slots=slots)
    return arena


def stage(name, values, dtype):
    return np.asarray(values)


def group(key, f, row0, prefix_end, *, start=0, borrow_key=None, points=()):
    return dict(key=key, fresh=True, f=f, start=start, row0=row0,
                prefix_end=prefix_end, row1=prefix_end + 2,
                suffix_spans=(np.array([prefix_end]),
                              np.array([prefix_end + 2])),
                kept=0, borrow_key=borrow_key, share_points=points)


def resident(arena, *keys):
    for key in keys:
        arena.accounting.alloc(key, 64)


def lengths(phases):
    return [phase.lengths for phase in phases]


def test_parent_writes_a_slot_at_the_shared_point(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p")
    phases = plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    assert lengths(phases) == [(32,), (16,), (2,)]
    shared = arena.shared_state("p", 32)
    assert shared is not None and shared != arena.accounting.slots["p"]
    assert phases[0].read_slots is None
    assert phases[0].write_slots.tolist() == [shared]
    assert phases[1].read_slots.tolist() == [shared]
    assert phases[1].write_slots.tolist() == [arena.accounting.slots["p"]]
    assert arena.checkpoint_tokens("p") == 48


def test_borrower_in_the_same_chunk_runs_one_level_after_the_state(
        torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    groups = [group("p", 48, 0, 48, points=(32,)),
              group("c", 64, 50, 82, start=32, borrow_key="p")]
    phases = plan_phases(arena, groups, stage)
    shared = arena.shared_state("p", 32)
    assert lengths(phases) == [(32,), (16, 32), (2, 2)]
    assert phases[1].read_slots.tolist() == [shared, shared]
    assert phases[1].write_slots.tolist() == [
        arena.accounting.slots["p"], arena.accounting.slots["c"]]
    # the child's rows are its tokens 32..64
    assert phases[1].rows.tolist()[16:] == list(range(50, 82))


def test_a_chain_in_one_chunk_takes_one_level_per_link(torch_stub):
    arena = arena_with_slots()
    resident(arena, "a", "b", "c")
    groups = [group("a", 32, 0, 32, points=(32,)),
              group("b", 64, 34, 66, start=32, borrow_key="a", points=(48,)),
              group("c", 80, 68, 100, start=48, borrow_key="b")]
    phases = plan_phases(arena, groups, stage)
    assert lengths(phases) == [(32,), (16,), (16, 32), (2, 2, 2)]
    assert phases[0].read_slots is None


def test_borrower_after_the_parent_chunk_reads_the_saved_state(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    shared = arena.shared_state("p", 32)
    phases = plan_phases(
        arena, [group("c", 64, 0, 32, start=32, borrow_key="p")], stage)
    assert lengths(phases) == [(32,), (2,)]
    assert phases[0].read_slots.tolist() == [shared]


def test_a_borrow_without_a_saved_state_is_refused(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    with pytest.raises(ValueError, match="no state of the parent"):
        plan_phases(
            arena, [group("c", 64, 0, 32, start=32, borrow_key="p")], stage)


def test_freeing_a_parent_frees_its_shared_slots(torch_stub):
    arena = arena_with_slots(slots=4)
    resident(arena, "p")
    plan_phases(arena, [group("p", 48, 0, 48, points=(16, 32))], stage)
    assert len(arena.accounting.free_slots) == 1
    arena.accounting.free_key("p")
    assert len(arena.accounting.free_slots) == 4
