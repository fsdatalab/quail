from scripts.analyze_rebuild import (
    bootstrap_median_upper,
    percentile,
)


def test_percentile_and_bootstrap_are_deterministic():
    assert percentile([3, 1, 2], 0.5) == 2
    first = bootstrap_median_upper(
        [0.8, 0.9, 1.0, 1.1, 0.95],
        repetitions=1000,
    )
    second = bootstrap_median_upper(
        [0.8, 0.9, 1.0, 1.1, 0.95],
        repetitions=1000,
    )
    assert first == second
