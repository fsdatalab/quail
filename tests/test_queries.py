"""CPU checks for the QUAIL-B query catalog."""

import pytest

from quail_b.queries import (
    PRIVACY_QUERIES,
    QUERIES,
    QUERY_FAMILY_WORKLOADS,
    QUERY_ORDER,
    AliasSpec,
    JoinSpec,
    QuerySpec,
    queries,
    query_family_name,
    split_query_families,
    split_query_ids,
)


def test_catalog_has_the_32_default_queries_and_two_privacy_queries():
    assert len(QUERIES) == 32
    assert QUERY_ORDER == (
        *(f"IMDB-{i}" for i in range(1, 11)),
        *(f"BIO-{i}" for i in range(1, 4)),
        *(f"FEV-{i}" for i in range(1, 10)),
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


def test_spec_rejects_a_join_that_does_not_add_its_alias():
    with pytest.raises(ValueError, match="must add alias"):
        QuerySpec("X-1", "bad", (AliasSpec("r", "reviews", "body"),
                                 AliasSpec("a", "aspects", "aspect")),
                  (JoinSpec("{0} {1}", ("r", "r")),), ("r.id",))


def test_parallel_query_split_matches_stock_vllm():
    assert split_query_ids(QUERY_ORDER, 4) == (
        QUERY_ORDER[0:8],
        QUERY_ORDER[8:16],
        QUERY_ORDER[16:24],
        QUERY_ORDER[24:32],
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
        QUERY_ORDER[13:22],
        QUERY_ORDER[22:30],
        QUERY_ORDER[30:32],
    )
    assert query_family_name(QUERY_ORDER[0:10]) == "imdb"


def test_query_family_rejects_mixed_or_unknown_queries():
    with pytest.raises(ValueError, match="expected one query family"):
        query_family_name(("IMDB-1", "BIO-1"))
    with pytest.raises(ValueError, match="unknown query family"):
        split_query_families(("OTHER-1",))
