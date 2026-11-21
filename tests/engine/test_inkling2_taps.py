"""Inkling's convolution taps through the checkpoint phases of two chunks."""

import sys
from types import ModuleType

import numpy as np
import pytest
from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena, PageArena
from quail.backends.quail.executor.gdn import plan_phases
from quail.backends.quail.executor.models.inkling2 import conv_taps


@pytest.fixture
def torch_stub(monkeypatch):
    module = ModuleType("torch")
    module.int64 = module.int32 = None
    monkeypatch.setitem(sys.modules, "torch", module)


def stage(name, values, dtype):
    return np.asarray(values)


def group(key, f, row0, prefix_end, *, start=0, borrow_key=None, points=()):
    return dict(key=key, fresh=True, f=f, start=start, row0=row0,
                prefix_end=prefix_end, row1=prefix_end + 2,
                suffix_spans=(np.array([prefix_end]),
                              np.array([prefix_end + 2])),
                kept=0, borrow_key=borrow_key, share_points=points)


def values_of(taps, rows, saved):
    """Each tap's value: a row's value, zero, or a saved state column."""
    out = np.zeros(taps.shape)
    for index, source in np.ndenumerate(taps):
        if source >= 0:
            out[index] = rows[source]
        elif source <= -2:
            slot, column = divmod(-source - 2, 3)
            out[index] = saved[slot][column]
    return out


def save(saved, slots, sources, rows):
    new = values_of(sources, rows, saved)
    for slot, columns in zip(slots, new):
        saved[int(slot)] = columns


def test_a_borrower_and_the_questions_read_the_inputs_before_them(torch_stub):
    arena = bare_arena(KVArena.__new__(KVArena), 64)
    arena.accounting = PageArena(64, 16, num_slots=8)
    arena.accounting.alloc("p", 64)
    arena.accounting.alloc("c", 64)
    saved = {}

    # chunk 1: document p, 48 tokens with a share point at 32, then two
    # question tokens
    first = np.arange(1.0, 51.0)
    phases = plan_phases(arena, [group("p", 48, 0, 48, points=(32,))], stage)
    taps, slots, sources = conv_taps(phases, 50)
    assert values_of(taps, first, saved)[[0, 2, 47, 48, 49]].tolist() == [
        [0, 0, 0], [0, 1, 2], [45, 46, 47], [46, 47, 48], [47, 48, 49]]
    save(saved, slots, sources, first)

    # chunk 2: document c borrows p's first 32 tokens and runs 32 more,
    # then two question tokens
    second = np.arange(101.0, 135.0)
    phases = plan_phases(
        arena, [group("c", 64, 0, 32, start=32, borrow_key="p")], stage)
    taps, slots, sources = conv_taps(phases, 34)
    assert values_of(taps, second, saved)[[0, 1, 32, 33]].tolist() == [
        [30, 31, 32], [31, 32, 101], [130, 131, 132], [131, 132, 133]]
