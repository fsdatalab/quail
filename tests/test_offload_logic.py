"""offload_logic must decide merges without an engine installed."""

from quail.engineext.offload_logic import (mergeable, plan_merge,
                                           split_result, synthetic_ids)


def test_mergeable_requires_trivial_chunking():
    assert mergeable(1, 1)
    assert not mergeable(2, 1)      # hybrid KV groups
    assert not mergeable(1, 4)      # CPU chunks span 4 GPU blocks


def test_plan_merge_batches_qualifying_jobs():
    jobs = [(10, 5, 5, 1), (11, 3, 3, 1), (12, 7, 7, 1)]
    merged, single = plan_merge(jobs, 1)
    assert merged == [10, 11, 12]
    assert single == []


def test_plan_merge_keeps_odd_jobs_single():
    jobs = [(10, 5, 5, 1), (11, 3, 4, 1),   # src/dst counts differ
            (12, 7, 7, 2),                  # two KV groups
            (13, 2, 2, 1)]
    merged, single = plan_merge(jobs, 1)
    assert merged == [10, 13]
    assert single == [11, 12]


def test_plan_merge_never_merges_a_lone_job():
    merged, single = plan_merge([(10, 5, 5, 1), (11, 3, 4, 1)], 1)
    assert merged == []
    assert single == [10, 11]


def test_plan_merge_respects_chunking():
    jobs = [(10, 5, 5, 1), (11, 3, 3, 1)]
    merged, single = plan_merge(jobs, 4)
    assert merged == []
    assert single == [10, 11]


def test_split_result_is_proportional_and_additive():
    times = split_result(1.0, [1, 3])
    assert times == [0.25, 0.75]
    assert sum(split_result(0.7, [2, 2, 2])) == 0.7
    assert split_result(1.0, [0, 0]) == [0.0, 0.0]


def test_synthetic_ids_never_collide_with_scheduler_ids():
    gen = synthetic_ids()
    ids = [next(gen) for _ in range(5)]
    assert ids == [-1, -2, -3, -4, -5]
    assert all(i < 0 for i in ids)
