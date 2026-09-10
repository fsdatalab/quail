"""Public benchmark loading without file-store objects."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import fs

import quail_b as benchmark
from quail_b import _files
from quail_b.data import GROUND_TRUTH_ROOT, PUBLISHED_CORPORA


def test_load_table_and_query(tmp_path):
    reviews = pa.table({"id": ["rv0", "rv1"], "body": ["good acting", "poor plot"]})
    assert set(PUBLISHED_CORPORA) == {0.1, 0.5, 1.0}
    for scale_factor, corpus_id in PUBLISHED_CORPORA.items():
        corpus = tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id
        corpus.mkdir(parents=True)
        pq.write_table(reviews, corpus / "reviews.parquet")
        for limit, expected in [(None, 2), (0, 0), (1, 1), (100, 2)]:
            loaded = benchmark.load_table(
                "reviews", scale_factor=scale_factor, limit=limit, root=tmp_path)
            assert loaded.equals(reviews.slice(0, expected))
    for kwargs in [
        {"name": "../reviews"}, {"name": "missing"},
        {"name": "reviews", "scale_factor": 0.7},
        *({"name": "reviews", "limit": value} for value in [-1, True, 1.5]),
    ]:
        with pytest.raises(ValueError):
            benchmark.load_table(root=tmp_path, **kwargs)
    with pytest.raises(FileNotFoundError):
        benchmark.load_table("aspects", root=tmp_path)
    assert benchmark.get_query("IMDB-1").aliases[0].table == "reviews"
    join = benchmark.get_query("IMDB-4")
    assert {alias.table for alias in join.aliases} == {"reviews", "aspects"}
    assert len(join.joins) == 1
    with pytest.raises(KeyError):
        benchmark.get_query("UNKNOWN")


def test_public_s3_and_local_reads(monkeypatch, tmp_path):
    bucket = tmp_path / "quail-bench"
    directory = bucket / "labels"
    directory.mkdir(parents=True)
    (directory / "a.json").write_text(json.dumps({"answer": True}))
    (directory / "notes.txt").write_text("not benchmark data")
    (directory / "nested").mkdir()
    pq.write_table(pa.table({"id": ["a"]}), directory / "nested" / "b.parquet")
    anonymous = []

    def s3_filesystem(**kwargs):
        anonymous.append(kwargs)
        return fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())

    monkeypatch.setattr(_files.fs, "S3FileSystem", s3_filesystem)
    expected = ["labels/a.json", "labels/nested/b.parquet"]
    for root in [None, "s3://quail-bench", bucket]:
        assert _files._list_files(root, "labels") == expected
        assert json.loads(_files._read_bytes(root, "/labels/a.json")) == {
            "answer": True}
        assert _files._list_files(root, "missing") == []
        with pytest.raises(FileNotFoundError):
            _files._read_bytes(root, "missing.json")
    assert anonymous and all(options == {"anonymous": True} for options in anonymous)
