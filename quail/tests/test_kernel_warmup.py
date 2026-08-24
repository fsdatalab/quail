"""CPU tests for the principled kernel-warmup helpers."""

from quail.executor.loop import (TINY_WARM_TOKENS, deepgemm_m_values,
                                 filter_warmup_docs, join_warmup_jobs)


def test_fallback_m_values_include_budget_and_block_m():
    ms, src = deepgemm_m_values(2560, 110_376)
    assert src == "fallback"
    assert ms[0] == 1
    assert 64 in ms
    assert 4096 in ms
    assert 110_376 in ms
    # denser than the old 256-from-64 grid
    assert 128 in ms
    assert 256 in ms
    assert 1024 in ms


def test_fallback_m_values_respect_small_budget():
    ms, _ = deepgemm_m_values(2560, 200)
    assert ms[-1] == 200
    assert all(m <= 200 for m in ms)


def test_filter_warmup_docs_fills_one_chunk():
    docs = [[1] * 100] * 20
    got = filter_warmup_docs(docs, q_max=20, budget=500)
    assert got
    used = sum(len(d) + 20 for d in got)
    assert used <= 500
    assert used + 100 + 20 > 500


def test_filter_warmup_docs_empty_when_nothing_fits():
    assert filter_warmup_docs([[1] * 100], q_max=20, budget=50) == []
    assert filter_warmup_docs([], q_max=10, budget=1000) == []


def test_join_warmup_jobs_three_shapes():
    docs = [[7] * 50]
    qs = [[9] * 12]
    jobs = join_warmup_jobs(docs, qs, budget=8_000)
    assert len(jobs) == 3
    for prefixes, stages in jobs:
        assert prefixes
        assert len(stages) == 1
        suffixes = stages[0]
        assert suffixes
        # one prefix plus one suffix must fit the budget
        assert len(prefixes[0]) + len(suffixes[0]) <= 8_000


def test_tiny_warm_tokens_cover_gated_tails():
    assert TINY_WARM_TOKENS[0] == 64
    assert 256 in TINY_WARM_TOKENS
    assert 512 in TINY_WARM_TOKENS
