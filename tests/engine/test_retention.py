"""Shared retention priority, capacity, and eviction tests."""

from helpers.fakes import bare_arena

from quail.backends.quail.executor.arena import KVArena
from quail.backends.quail.retention import apply_retention
from quail.common.retention import RetentionPolicy
from quail.progress import set_answer_sink


def cpu_arena(pages, cap, uses):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting.configure_retention(RetentionPolicy(1.0, 0.01, uses), cap)
    return arena


def allocate(arena, key, tokens):
    assert arena.accounting.alloc(key, tokens) is not None
    arena._rows[key] = None
    arena._capacity_rows[key] = None


def test_retention_priority_capacity_eviction_and_its_report():
    seen = []
    set_answer_sink(seen.append)
    try:
        arena = cpu_arena(8, 2, {"a": (1, 0)})
        allocate(arena, ("a", 0), 32)
        arena.retain(("a", 0), 32)
        allocate(arena, ("a", 1), 32)
        arena.retain(("a", 1), 32)
    finally:
        set_answer_sink(None)
    assert seen == [{"kind": "evict", "alias": "a", "document": 1, "tokens": 32}]

    arena = cpu_arena(16, 4, {'early': (1, 0), 'later': (1, 1)})
    allocate(arena, ('later', 0), 32)
    arena.retain(('later', 0), 32)
    allocate(arena, ('early', 0), 64)
    freed = arena.retain(('early', 0), 64)
    assert freed == 2
    assert set(arena.accounting.retained) == {('early', 0)}
    assert arena.accounting.retained_pages == 4
    assert arena.evicted_prefix_tokens == 32

    unused = RetentionPolicy(1, 0.01, {'useful': (1, 0)})
    assert unused.priority(('unused', 0), 1000, 63)[0] == 0

    arena = cpu_arena(16, 8, {'a': (1, 0), 'b': (1, 1)})
    for alias in ('a', 'b', 'active'):
        allocate(arena, (alias, 0), 32)
    arena.retain(('a', 0), 32)
    arena.retain(('b', 0), 32)
    arena.accounting.pin(('active', 0))
    arena.accounting.configure_retention(
        RetentionPolicy(1, 0.01, {'a': (0.1, 2), 'b': (1, 1)}), 8)
    arena.evict_retained(2)
    assert ('a', 0) not in arena.accounting.owned
    assert ('active', 0) in arena.accounting.pinned
    assert ('b', 0) in arena.accounting.retained

    arena = cpu_arena(16, 8, {'a': (1, 0), 'b': (1, 1)})
    for key in [('a', 0), ('b', 0), ('b', 1)]:
        allocate(arena, key, 32)
        arena.retain(key, 32)
    config = dict(linear_seconds=1, pair_seconds=0.01, cap_pages=8)
    apply_retention(arena, config, {'b': (1, 1)}, {'b': [1]})
    assert set(arena.accounting.retained) == {('b', 1)}
    assert arena.accounting.retained_pages == 2
    assert arena.accounting.free_pages == 14
