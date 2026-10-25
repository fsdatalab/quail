"""KV pages that grow as documents write: who grows, and who is admitted."""

import numpy as np
from helpers.fakes import cpu_arena

from quail.backends.quail.executor.growth import KVGrowth


class Answers:
    """A growth model with planted needs: rows to cover, and rows left."""

    initial_tokens = 16

    def __init__(self):
        self.wanted = {}
        self.left = {}

    def capacities(self, anchors):
        return np.array([self.wanted.get(int(a), 0) for a in anchors], dtype=np.int64)

    def remaining(self, anchors):
        return np.array([self.left.get(int(a), 64) for a in anchors], dtype=np.int64)


def _growth(pages, documents):
    """An arena of pages, and documents of one 16-row page that can reach 80 rows."""
    arena = cpu_arena(pages)
    keys = [("d", index) for index in range(documents)]
    model = Answers()
    growth = KVGrowth(arena, keys, model, [16] * documents, [80] * documents)
    return arena, keys, model, growth


def _admit(arena, keys, growth, anchor):
    arena.activate(keys[anchor], 16, capacity_tokens=16)
    growth.admitted(anchor)


def test_the_oldest_document_grows_first_and_a_younger_one_waits():
    # nine pages: each document holds one and may need four more; the
    # younger may grow only while the older can still finish
    arena, keys, model, growth = _growth(9, 2)
    _admit(arena, keys, growth, 0)
    _admit(arena, keys, growth, 1)
    model.wanted = {0: 80, 1: 80}
    stalled = growth.grow([1, 0], arena.free_pages)
    assert stalled == {1}
    assert len(arena.accounting.table_pages(keys[0])) == 5
    # once the older document ends, its pages let the younger grow
    arena.free_key(keys[0])
    growth.settled(0)
    assert growth.grow([1], arena.free_pages) == set()
    assert len(arena.accounting.table_pages(keys[1])) == 5


def test_a_document_is_admitted_when_the_projected_peak_fits():
    # two documents hold a page each: one is expected to end within
    # 16 rows, the other within 64. A third, admitted with two pages,
    # peaks with them at five pages past what is free after it
    arena, keys, model, growth = _growth(12, 3)
    _admit(arena, keys, growth, 0)
    _admit(arena, keys, growth, 1)
    model.left = {0: 16, 1: 64, 2: 64}
    assert growth.admits(2, 2, arena.free_pages)
    # with four fewer pages, four are left after it: the peak of five
    # does not fit
    arena, keys, model, growth = _growth(8, 3)
    _admit(arena, keys, growth, 0)
    _admit(arena, keys, growth, 1)
    model.left = {0: 16, 1: 64, 2: 64}
    assert not growth.admits(2, 2, arena.free_pages)
