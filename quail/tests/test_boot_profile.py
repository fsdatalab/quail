"""CPU tests for boot_profile mean/median aggregation."""

from baselines.boot_stats import aggregate, mean, median


def test_mean_median_ignore_none():
    assert mean([1.0, None, 3.0]) == 2.0
    assert median([1.0, None, 3.0]) == 2.0
    assert mean([]) is None
    assert median([None, None]) is None


def test_aggregate_quail_three_trials():
    trials = [
        dict(side="quail", trial=0,
             cold=dict(boot_s=30.0, load_model_s=10.0, arena_s=1.0,
                       pipeline_s=0.5),
             warm=dict(boot_s=0.0, load_model_s=0.0, arena_s=0.0,
                       pipeline_s=0.0)),
        dict(side="quail", trial=1,
             cold=dict(boot_s=32.0, load_model_s=11.0, arena_s=1.0,
                       pipeline_s=0.5),
             warm=dict(boot_s=0.01, load_model_s=0.0, arena_s=0.0,
                       pipeline_s=0.0)),
        dict(side="quail", trial=2,
             cold=dict(boot_s=34.0, load_model_s=12.0, arena_s=1.0,
                       pipeline_s=0.5),
             warm=dict(boot_s=0.0, load_model_s=0.0, arena_s=0.0,
                       pipeline_s=0.0)),
    ]
    agg = aggregate("quail", trials)
    assert agg["n_trials"] == 3
    assert agg["cold"]["boot_s"]["mean"] == 32.0
    assert agg["cold"]["boot_s"]["median"] == 32.0
    assert agg["cold"]["boot_s"]["trials"] == [30.0, 32.0, 34.0]
    assert agg["warm"]["boot_s"]["median"] == 0.0
