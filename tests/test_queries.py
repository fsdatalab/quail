"""CPU checks for the QUAIL-B query catalog."""

import pytest

from quail_b.queries import (
    PRIVACY_QUERIES,
    QUERIES,
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    FilterSpec,
    JoinSpec,
    QuerySpec,
    RelationSpec,
    queries,
    query_family_name,
    split_query_families,
    split_query_ids,
)


def test_catalog_has_the_33_default_queries_and_two_privacy_queries():
    assert len(QUERIES) == 33
    assert QUERY_ORDER == (
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 4)),
        *(f"FEV-{i}" for i in range(1, 11)),
        *(f"LEP-{i}" for i in range(1, 9)),
        "AGENT-1", "AGENT-2",
    )
    assert [spec.id for spec in PRIVACY_QUERIES] == ["PRIV-1", "PRIV-2"]
    assert list(queries(include_privacy=True)) == [*QUERY_ORDER, "PRIV-1", "PRIV-2"]
    for spec in QUERIES:
        assert spec.has_estimates and spec.order == "by_cost", spec.id
        assert all(name.endswith(".id") for name in spec.select), spec.id
    for spec in PRIVACY_QUERIES:
        assert not spec.has_estimates and spec.order == "as_written", spec.id


def test_spec_rejects_a_join_that_does_not_add_a_relation():
    with pytest.raises(ValueError, match="must add one relation"):
        QuerySpec(
            "X-1",
            "bad",
            (
                RelationSpec("r", "reviews", "body"),
                RelationSpec("a", "aspects", "aspect"),
                RelationSpec("b", "aspects", "aspect"),
            ),
            (
                JoinSpec("join-1", ("r", "a"), "{0} {1}"),
                JoinSpec("join-2", ("r", "a"), "{0} {1}"),
            ),
            ("r.id",),
        )


def test_filters_are_explicit_ordered_operators():
    spec = queries()["IMDB-4"]

    assert spec.relations == (
        RelationSpec("r", "reviews", "body"),
        RelationSpec("a", "aspects", "aspect"),
    )
    assert [
        (operator.id, operator.relation) for operator in spec.filters
    ] == [
        ("filter-1", "r"),
        ("filter-2", "r"),
    ]
    assert [operator.id for operator in spec.joins] == ["join-1"]
    assert isinstance(spec.operators[0], FilterSpec)
    assert isinstance(spec.operators[-1], JoinSpec)


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:9],
        QUERY_ORDER[9:17],
        QUERY_ORDER[17:25],
        QUERY_ORDER[25:33],
    )


def test_query_family_split_matches_benchmark_catalog():
    assert QUERY_FAMILY_WORKLOADS == {
        "IMDB": "imdb",
        "BIO": "biodex",
        "FEV": "fever",
        "LEP": "lepard",
        "AGENT": "agent",
    }
    assert split_query_families(QUERY_ORDER) == (
        QUERY_ORDER[0:10],
        QUERY_ORDER[10:13],
        QUERY_ORDER[13:23],
        QUERY_ORDER[23:31],
        QUERY_ORDER[31:33],
    )
    assert query_family_name(QUERY_ORDER[0:10]) == "imdb"


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))
