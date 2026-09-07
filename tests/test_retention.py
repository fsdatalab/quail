"""Shared retention priority, capacity, and eviction tests."""


from quail.executor.arena import PageArena, KVArena
from quail.executor.retention import RetentionPolicy, retention_pages


def cpu_arena(pages, cap, uses):
    arena = KVArena.__new__(KVArena)
    arena.accounting = PageArena(pages, 16)
    arena._rows = {}
    arena._capacity_rows = {}
    arena._refresh_rows = lambda *args: None
    arena.reset_stats()
    arena.accounting.configure_retention(RetentionPolicy(1.0, 0.01, uses), cap)
    return arena


def allocate(arena, key, tokens):
    assert arena.accounting.alloc(key, tokens) is not None
    arena._rows[key] = None
    arena._capacity_rows[key] = None


def test_shared_pool_replaces_an_earlier_collection():
    arena = cpu_arena(16, 4, {'early': (1, 0), 'later': (1, 1)})
    allocate(arena, ('later', 0), 32)
    arena.retain(('later', 0), 32)
    allocate(arena, ('early', 0), 64)
    freed = arena.retain(('early', 0), 64)
    assert freed == 2
    assert set(arena.accounting.retained) == {('early', 0)}
    assert arena.accounting.retained_pages == 4
    assert arena.evicted_prefix_tokens == 32


def test_longer_document_does_not_win_without_future_reuse():
    policy = RetentionPolicy(1, 0.01, {'useful': (1, 0)})
    assert policy.priority(('unused', 0), 1000, 63)[0] == 0
    assert policy.priority(('useful', 0), 100, 7)[0] > 0


def test_equal_prefixes_prefer_earlier_use():
    arena = cpu_arena(8, 2, {'first': (1, 0), 'last': (1, 1)})
    for alias in ('last', 'first'):
        allocate(arena, (alias, 0), 32)
        arena.retain((alias, 0), 32)
    assert set(arena.accounting.retained) == {('first', 0)}


def test_rejected_candidate_returns_its_pages():
    arena = cpu_arena(8, 2, {'a': (1, 0)})
    allocate(arena, ('a', 0), 32)
    arena.retain(('a', 0), 32)
    allocate(arena, ('a', 1), 1)
    assert arena.retain(('a', 1), 1) == 1
    assert ('a', 1) not in arena.accounting.owned


def test_stage_update_reorders_only_retained_prefixes():
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


def test_zero_capacity_and_oversized_prefix():
    for cap in (0, 1):
        arena = cpu_arena(8, cap, {'a': (1, 0)})
        allocate(arena, ('a', 0), 32)
        assert arena.retain(('a', 0), 32) == 2
        assert not arena.accounting.owned


def test_retention_budget_rounds_execution_reservation():
    assert retention_pages(362250, 110376, 16) == 8843
    assert retention_pages(32, 32, 16) == 0


def test_filter_chains_share_retention_and_return_evicted_pages(monkeypatch):
    from types import SimpleNamespace
    from quail.executor import loop
    from quail.executor.attention import FILTER_ATTENTION

    arena = cpu_arena(32, 8, {'e1': (1, 0), 'e2': (1, 1)})

    def allocate_rows(key, tokens, capacity_tokens=None):
        pages = arena.accounting.alloc(key, tokens, capacity_tokens)
        if pages is not None:
            arena._rows[key] = None
            arena._capacity_rows[key] = None
        return pages

    arena.alloc = allocate_rows
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: SimpleNamespace(record=lambda: None)))
    answers = SimpleNamespace(submit=lambda values: values, result=lambda values: values)
    pipeline = SimpleNamespace(attention_mode=FILTER_ATTENTION,
                               forward_chunk=lambda chunk: [True] * len(chunk['specs']))
    monkeypatch.setattr(loop, 'pack_chunk', lambda torch, arena, specs, **kw: {
        'specs': specs, 'tokens': sum(spec['f'] + 1 for spec in specs)})
    for alias, lengths in [('e2', [16, 64]), ('e1', [32, 64])]:
        result, _, _ = loop.run_filter(
            torch, arena, pipeline, answers, [[1] * length for length in lengths],
            [[2]], 80, arena_writes=True,
            arena_keys=[(alias, i) for i in range(2)], retain_survivors=True)
        assert result == {0: [1], 1: [1]}
    assert set(arena.accounting.retained) == {('e1', 1), ('e2', 1)}
    assert arena.accounting.retained_pages == 8
    assert arena.accounting.free_pages == 24


def test_expected_allocation_uses_lengths_and_selectivity():
    from quail.planner.joins import summarize_alias
    from quail.planner.retention import allocate

    lengths = {alias: summarize_alias([16, 160]) for alias in ('a', 'b')}
    credited, budgets = allocate(
        lengths, {'a': 1.0, 'b': 1.0}, lengths, {'a': (1, 0), 'b': (1, 1)},
        0, 10, 16, {'linear_seconds': 1, 'pair_seconds': 0.01})
    assert sum(row['pages'] for row in budgets.values()) == 10
    for alias in lengths:
        assert budgets[alias]['documents'] == 0.5
        assert credited[alias].resident_count == 1
        assert credited[alias].resident_total == 160
        assert credited[alias].resident_squared == 160**2


def test_schedule_keeps_future_anchors_and_ends_at_last_use():
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


def test_priority_matches_the_existing_prefix_cost():
    import pytest
    from quail.planner.retention import coefficients
    from quail.planner.sol import prefix_recompute_seconds
    from quail.specs import QWEN3_4B_FP8, H100_SXM

    policy = RetentionPolicy(**coefficients(QWEN3_4B_FP8, H100_SXM), uses={'a': (0.4, 2)})
    for length in (1, 16, 17, 1000, 24000):
        pages = -(-length // 16)
        assert policy.priority(('a', 0), length, pages)[0] == pytest.approx(
            0.4 * prefix_recompute_seconds(length, QWEN3_4B_FP8, H100_SXM) / pages)


def test_boundary_releases_dead_and_expired_but_preserves_later_alias():
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
