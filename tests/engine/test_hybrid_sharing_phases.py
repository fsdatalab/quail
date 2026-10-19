"""Gated DeltaNet phases for documents that borrow a shared prefix."""

import sys
from types import ModuleType

import numpy as np
import pytest
from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.chunk import (
    ArenaFullError,
    BorrowStateMissingError,
)
from quail.backends.quail.executor.gdn import plan_phases


@pytest.fixture
def torch_stub(monkeypatch):
    module = ModuleType("torch")
    module.int64 = module.int32 = None
    monkeypatch.setitem(sys.modules, "torch", module)


def arena_with_slots(slots=8):
    arena = bare_arena(KVArena.__new__(KVArena), 64)
    arena.accounting = PageArena(64, 16, num_slots=slots)
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


def test_borrower_after_the_parent_chunk_reads_the_saved_state(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    shared = arena.shared_state("p", 32)
    phases = plan_phases(
        arena, [group("c", 64, 0, 32, start=32, borrow_key="p")], stage)
    assert lengths(phases) == [(32,), (2,)]
    assert phases[0].read_slots.tolist() == [shared]
    phases = plan_phases(arena, [resident_group("p", 48, 0, 2)], stage)
    assert lengths(phases) == [(2,)]
    assert phases[0].read_slots.tolist() == [arena.accounting.slots["p"]]
    with pytest.raises(ValueError, match="checkpoints cover"):
        plan_phases(arena, [resident_group("p", 32, 0, 2)], stage)


def test_a_later_group_does_not_evict_a_parent_an_earlier_group_reads(
        torch_stub):
    arena = arena_with_slots(slots=3)
    resident(arena, "p")
    plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    shared = arena.shared_state("p", 32)
    arena.accounting.retain("p")
    resident(arena, "c", "d")
    groups = [group("c", 64, 0, 32, start=32, borrow_key="p"),
              group("d", 48, 34, 82)]
    with pytest.raises(ArenaFullError):
        plan_phases(arena, groups, stage)
    assert arena.shared_state("p", 32) == shared
    assert not arena._reading


# ---- edge cases -------------------------------------------------------

def resident_group(key, f, row0, suffix, *, kept=0):
    """A group asked of a document an earlier chunk computed."""
    return dict(key=key, fresh=False, f=f, start=0, row0=row0,
                prefix_end=row0, row1=row0 + suffix,
                suffix_spans=(np.array([row0]), np.array([row0 + suffix])),
                kept=kept, borrow_key=None, share_points=())


def test_points_at_or_below_the_start_are_not_saved(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    phases = plan_phases(
        arena, [group("c", 80, 0, 48, start=32, borrow_key="p",
                      points=(16, 32, 48))], stage)
    assert lengths(phases)[:2] == [(16,), (32,)]
    assert list(arena.accounting.share_slots) == [("p", 32), ("c", 48)]


def test_only_distinct_interior_share_points_make_segments(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p")
    phases = plan_phases(
        arena, [group("p", 48, 0, 48, points=(0, 32, 16, 48, 64, 32, 16))], stage)
    assert lengths(phases) == [(16,), (16,), (16,), (2,)]
    assert sorted(t for _, t in arena.accounting.share_slots) == [16, 32]


def test_two_borrowers_of_one_point_run_in_the_same_phase(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "a", "b")
    groups = [group("p", 48, 0, 48, points=(32,)),
              group("a", 64, 50, 82, start=32, borrow_key="p"),
              group("b", 64, 84, 116, start=32, borrow_key="p")]
    phases = plan_phases(arena, groups, stage)
    shared = arena.shared_state("p", 32)
    assert lengths(phases) == [(32,), (16, 32, 32), (2, 2, 2)]
    assert phases[1].read_slots.tolist() == [shared] * 3


def test_a_borrower_of_the_parents_whole_prefix_reads_its_slot(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    groups = [group("p", 32, 0, 32),
              group("c", 64, 34, 66, start=32, borrow_key="p")]
    phases = plan_phases(arena, groups, stage)
    assert phases[1].read_slots.tolist() == [arena.accounting.slots["p"]]
    assert phases[1].write_slots.tolist() == [arena.accounting.slots["c"]]


def test_a_state_at_a_different_length_does_not_serve_a_borrow(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p", "c")
    plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    with pytest.raises(BorrowStateMissingError, match="no state of the parent"):
        plan_phases(
            arena, [group("c", 80, 0, 32, start=16, borrow_key="p")], stage)


def test_a_missing_share_slot_splits_and_the_retry_reuses_what_it_took(
        torch_stub):
    from quail.backends.quail.executor.chunk import ArenaFullError

    arena = arena_with_slots(slots=2)
    resident(arena, "p")
    full = group("p", 48, 0, 48, points=(16, 32))
    with pytest.raises(ArenaFullError, match="share"):
        plan_phases(arena, [full], stage)
    taken = dict(arena.accounting.share_slots)
    arena.accounting.num_slots += 1
    arena.accounting.free_slots.append(2)
    phases = plan_phases(arena, [full], stage)
    assert all(arena.accounting.share_slots[k] == v for k, v in taken.items())
    assert lengths(phases) == [(16,), (16,), (16,), (2,)]


def test_a_wholly_kept_suffix_continues_after_the_last_segment(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p")
    framed = group("p", 48, 0, 48, points=(32,))
    framed["kept"] = 2
    framed["suffix_spans"] = (np.array([48]), np.array([50]))
    phases = plan_phases(arena, [framed], stage)
    # prefix to the shared point, prefix to the end, then the frame
    # tokens from the document's slot into a second slot
    assert lengths(phases) == [(32,), (16,), (2,)]
    assert phases[2].read_slots.tolist() == [arena.accounting.slots["p"]]
    assert phases[2].write_slots.tolist() == [arena.accounting.ext_slots["p"]]
    assert arena.extension("p") == (arena.accounting.ext_slots["p"], 50)


def test_a_suffix_with_a_question_extends_after_it_ran(torch_stub):
    arena = arena_with_slots()
    resident(arena, "p")
    framed = group("p", 48, 0, 48, points=(32,))
    framed["kept"] = 2
    framed["suffix_spans"] = (np.array([48]), np.array([52]))
    phases = plan_phases(arena, [framed], stage)
    assert lengths(phases) == [(32,), (16,), (4,), (2,)]
    suffix, extension = phases[2], phases[3]
    assert suffix.write_slots is None and extension.output is False
    assert suffix.read_slots.tolist() == [arena.accounting.slots["p"]]
    assert extension.write_slots.tolist() == [arena.accounting.ext_slots["p"]]


def test_levels_cover_every_prefix_row_once_and_never_read_ahead(
        torch_stub):
    rng = np.random.default_rng(7)
    borrowers = 0
    for trial in range(40):
        arena = arena_with_slots(slots=64)
        count = int(rng.integers(1, 7))
        keys = [f"d{i}" for i in range(count)]
        resident(arena, *keys)
        lengths_ = [int(rng.integers(2, 6)) * 16 for _ in keys]
        groups, row = [], 0
        parent_of, start_of = {}, {}
        for i, key in enumerate(keys):
            parent = int(rng.integers(0, i)) if i and rng.random() < 0.7 \
                else None
            start = 0
            if parent is not None:
                low = start_of.get(parent, 0) // 16 + 1
                high = min(lengths_[parent], lengths_[i] - 16) // 16
                if high >= low:
                    start = int(rng.integers(low, high + 1)) * 16
            parent_of[i], start_of[i] = (parent if start else None), start
        points = {i: set() for i in range(count)}
        for i, parent in parent_of.items():
            if parent is not None:
                points[parent].add(start_of[i])
                borrowers += 1
        order = list(range(count))
        for i in order:
            n = lengths_[i] - start_of[i]
            groups.append(group(
                keys[i], lengths_[i], row, row + n, start=start_of[i],
                borrow_key=None if parent_of[i] is None
                else keys[parent_of[i]],
                points=tuple(sorted(points[i]))))
            row += n + 2
        phases = plan_phases(arena, groups, stage)
        prefix_rows = np.concatenate(
            [g["row0"] + np.arange(g["prefix_end"] - g["row0"])
             for g in groups])
        got = np.concatenate([p.rows for p in phases if p.output])
        suffix_rows = np.concatenate(
            [np.arange(g["prefix_end"], g["row1"]) for g in groups])
        assert sorted(got.tolist()) == sorted(
            prefix_rows.tolist() + suffix_rows.tolist()), trial
        written = set()
        for phase in phases:
            if phase.read_slots is not None:
                assert set(phase.read_slots.tolist()) <= written, trial
            if phase.write_slots is not None:
                written |= set(phase.write_slots.tolist())
    # the trials must exercise borrowing, or the checks above are empty
    assert borrowers > 20
