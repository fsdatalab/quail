"""Checks for the parallel QUAIL-B run layout."""

from datetime import datetime, timezone

from quail.bench.quailb_parallel import _merge_suites, _query_groups


def test_bio_queries_use_separate_groups():
    query_ids = ("IMDB-1", "IMDB-2", "BIO-1", "BIO-2", "FEV-1", "AGENT-1")

    assert _query_groups(query_ids) == [
        ("imdb", ("IMDB-1", "IMDB-2"), ""),
        ("biodex-bio-1", ("BIO-1",), "-bio-1"),
        ("biodex-bio-2", ("BIO-2",), "-bio-2"),
        ("fever", ("FEV-1",), ""),
        ("agent", ("AGENT-1",), ""),
    ]


def test_merged_queries_name_their_group_folder():
    def suite(family, query_id):
        return {
            "run_id": "run", "scale_factor": 0.5, "corpus_id": "c",
            "collection_id": "l", "metadata": {}, "gpu_count": 1,
            "gpu_hourly_rate_usd": 3.9492, "query_family": {"name": family},
            "queries": [{"id": query_id}],
        }

    merged = _merge_suites(
        [suite("biodex", "BIO-1"), suite("fever", "FEV-1")],
        ["biodex-bio-1", "fever"], ("BIO-1", "FEV-1"), "run",
        datetime.now(timezone.utc), 1.0, [], ())

    assert [item["directory"] for item in merged["queries"]] == [
        "biodex-bio-1/BIO-1", "fever/FEV-1"]
