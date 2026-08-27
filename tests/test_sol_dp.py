from quail.bench.sol_dp import (
    PairRelation,
    exact_live_rows,
)


def test_exact_live_rows_reduces_a_tree_to_exact_projections():
    live = exact_live_rows(
        {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        (
            PairRelation("a", "b", frozenset({(0, 0), (1, 1)})),
            PairRelation("b", "c", frozenset({(0, 0)})),
        ),
    )

    assert live == {"a": (0,), "b": (0,), "c": (0,)}


def test_exact_live_rows_handles_cycle_consistency():
    live = exact_live_rows(
        {"a": (0, 1), "b": (0, 1), "c": (0, 1)},
        (
            PairRelation("a", "b", frozenset({(0, 0), (1, 1)})),
            PairRelation("b", "c", frozenset({(0, 0), (1, 1)})),
            PairRelation("a", "c", frozenset({(0, 1), (1, 0)})),
        ),
    )

    assert live == {"a": (), "b": (), "c": ()}
