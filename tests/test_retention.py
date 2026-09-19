"""Shared retention priority, capacity, and eviction tests."""


from fakes import bare_arena, fake_pipeline

from quail.backends.quail.executor.arena import KVArena
from quail.cost.retention import RetentionPolicy


def cpu_arena(pages, cap, uses):
    arena = bare_arena(KVArena.__new__(KVArena), pages)
    arena.accounting.configure_retention(RetentionPolicy(1.0, 0.01, uses), cap)
    return arena


def allocate(arena, key, tokens):
    assert arena.accounting.alloc(key, tokens) is not None
    arena._rows[key] = None
    arena._capacity_rows[key] = None


def test_retention_priority_capacity_and_eviction():
    arena = cpu_arena(16, 4, {'early': (1, 0), 'later': (1, 1)})
    allocate(arena, ('later', 0), 32)
    arena.retain(('later', 0), 32)
    allocate(arena, ('early', 0), 64)
    freed = arena.retain(('early', 0), 64)
    assert freed == 2
    assert set(arena.accounting.retained) == {('early', 0)}
    assert arena.accounting.retained_pages == 4
    assert arena.evicted_prefix_tokens == 32

    policy = RetentionPolicy(1, 0.01, {'useful': (1, 0)})
    assert policy.priority(('unused', 0), 1000, 63)[0] == 0
    assert policy.priority(('useful', 0), 100, 7)[0] > 0

    arena = cpu_arena(8, 2, {'first': (1, 0), 'last': (1, 1)})
    for alias in ('last', 'first'):
        allocate(arena, (alias, 0), 32)
        arena.retain((alias, 0), 32)
    assert set(arena.accounting.retained) == {('first', 0)}

    arena = cpu_arena(8, 2, {'a': (1, 0)})
    allocate(arena, ('a', 0), 32)
    arena.retain(('a', 0), 32)
    allocate(arena, ('a', 1), 1)
    assert arena.retain(('a', 1), 1) == 1
    assert ('a', 1) not in arena.accounting.owned

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

    for cap in (0, 1):
        arena = cpu_arena(8, cap, {'a': (1, 0)})
        allocate(arena, ('a', 0), 32)
        assert arena.retain(('a', 0), 32) == 2
        assert not arena.accounting.owned

    from quail.backends.quail.retention import apply_retention

    arena = cpu_arena(16, 8, {'a': (1, 0), 'b': (1, 1)})
    for key in [('a', 0), ('b', 0), ('b', 1)]:
        allocate(arena, key, 32)
        arena.retain(key, 32)
    config = dict(linear_seconds=1, pair_seconds=0.01, cap_pages=8)
    apply_retention(arena, config, {'b': (1, 1)}, {'b': [1]})
    assert set(arena.accounting.retained) == {('b', 1)}
    assert arena.accounting.retained_pages == 2
    assert arena.accounting.free_pages == 14


def test_filter_chains_share_retention_and_return_evicted_pages(monkeypatch):
    from types import SimpleNamespace

    from quail.backends.quail.executor import loop

    arena = cpu_arena(32, 8, {'e1': (1, 0), 'e2': (1, 1)})

    def allocate_rows(key, tokens, capacity_tokens=None, base_tokens=None,
                      sliding_tokens=None):
        pages = arena.accounting.alloc(key, tokens, capacity_tokens)
        if pages is not None:
            arena._rows[key] = None
            arena._capacity_rows[key] = None
            arena._base[key] = tokens if base_tokens is None else base_tokens
            arena._sliding_start[key] = 0
        return pages

    arena.alloc = allocate_rows
    torch = SimpleNamespace(cuda=SimpleNamespace(
        Event=lambda **kw: SimpleNamespace(record=lambda: None)))
    answers = SimpleNamespace(submit=lambda values: values,
                              result=lambda values: values)
    pipeline = fake_pipeline(
        forward_chunk=lambda chunk: [True] * len(chunk.specs))
    monkeypatch.setattr(
        loop, 'pack_chunk',
        lambda torch, arena, specs, **kw: SimpleNamespace(
            specs=specs, tokens=sum(spec['f'] + 1 for spec in specs),
            temporary_keys=(), fresh_keys=()))
    for alias, lengths in [('e2', [16, 64]), ('e1', [32, 64])]:
        result, _, _ = loop.run_filter(
            torch, arena, pipeline, answers, [[1] * length for length in lengths],
            [[2]], 80, arena_writes=True,
            arena_keys=[(alias, i) for i in range(2)], retain_survivors=True)
        assert result == {0: [1], 1: [1]}
    assert set(arena.accounting.retained) == {('e1', 1), ('e2', 1)}
    assert arena.accounting.retained_pages == 8
    assert arena.accounting.free_pages == 24


def test_retention_costs_and_future_use():
    from quail.planner.retention import schedule

    def stage(position, aliases):
        return {'written_pos': position, 'aliases': aliases, 'anchor': aliases[0],
                'semantics': 'full', 'selectivity': 0.1}

    result = schedule([(stage(0, ['a', 'b']), 'a'),
                       (stage(1, ['b', 'c']), 'c'),
                       (stage(2, ['a', 'c']), 'a')], dict(a=10., b=10., c=10.))
    assert result['initial']['a'] == [1.0, 0]
    assert result['initial']['c'] == [1.0, 1]
    assert set(result['after']['group:0']) == {'a', 'c'}
    assert result['after']['group:2'] == {}

    import pytest

    from quail.cost.retention import coefficients
    from quail.cost.sol import prefix_recompute_seconds
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    policy = RetentionPolicy(**coefficients(QWEN3_4B_FP8, H100_SXM),
                             uses={'a': (0.4, 2)})
    for length in (1, 16, 17, 1000, 24000):
        pages = -(-length // 16)
        assert policy.priority(('a', 0), length, pages)[0] == pytest.approx(
            0.4 * prefix_recompute_seconds(length, QWEN3_4B_FP8, H100_SXM) / pages)
