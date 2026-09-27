"""Checks for the parallel QUAIL-B run layout."""

from quail.bench.quailb_parallel import _query_groups


def test_bio_queries_use_separate_groups():
    query_ids = ("IMDB-1", "IMDB-2", "BIO-1", "BIO-2", "FEV-1", "AGENT-1")

    assert _query_groups(query_ids) == [
        ("imdb", ("IMDB-1", "IMDB-2"), ""),
        ("biodex-bio-1", ("BIO-1",), "-bio-1"),
        ("biodex-bio-2", ("BIO-2",), "-bio-2"),
        ("fever", ("FEV-1",), ""),
        ("agent", ("AGENT-1",), ""),
    ]
