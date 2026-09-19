"""Public benchmark loading without file-store objects."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyarrow import fs

import quail_b as benchmark
from quail_b import _files
from quail_b.data import GROUND_TRUTH_ROOT, PUBLISHED_CORPORA
from quail_b.substrait import _inspect_plan


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
    first = _inspect_plan(benchmark.get_query("IMDB-1").plan)
    assert first.relations[0].table == "reviews"
    join = _inspect_plan(benchmark.get_query("IMDB-4").plan)
    assert {relation.table for relation in join.relations} == {
        "reviews", "aspects"
    }
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
    with _files.download_cache(tmp_path / "cache"):
        for root in [None, "s3://quail-bench", bucket]:
            assert _files._list_files(root, "labels") == expected
            assert json.loads(_files._read_bytes(root, "/labels/a.json")) == {
                "answer": True}
            assert _files._list_files(root, "missing") == []
            with pytest.raises(FileNotFoundError):
                _files._read_bytes(root, "missing.json")
        (directory / "a.json").write_text(json.dumps({"answer": False}))
        assert json.loads(_files._read_bytes(None, "labels/a.json"))["answer"] is True
        for name in ("active_collection.json", "active_collection.chat.json"):
            pointer = directory / name
            for collection in ("old", "new"):
                pointer.write_text(json.dumps({"collection": collection}))
                assert json.loads(_files._read_bytes(
                    None, f"labels/{name}"))["collection"] == collection
    assert anonymous and all(options == {"anonymous": True} for options in anonymous)


def test_load_benchmark_validates_selected_inputs(tmp_path):
    from quail_b.data import DATA_SEED, SOURCE_REVISIONS, corpus_identity

    tables = {
        "reviews": pa.table({
            "id": ["r0", "r1"], "body": ["good acting", "poor ending"]}),
        "aspects": pa.table({"id": ["a0"], "aspect": ["acting"]}),
    }
    for sf, corpus_id in PUBLISHED_CORPORA.items():
        directory = tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id
        directory.mkdir(parents=True)
        manifest = corpus_identity(tables, sf, DATA_SEED, SOURCE_REVISIONS)
        manifest["corpus_id"] = corpus_id
        (directory / "manifest.json").write_text(json.dumps(manifest))
        for name, table in tables.items():
            pq.write_table(table, directory / f"{name}.parquet")
        for data_dir in (None, directory):
            loaded = benchmark.load_benchmark(
                ["IMDB-4"], scale_factor=sf, root=tmp_path,
                data_dir=data_dir, accuracy=False)
            assert loaded.corpus_id == corpus_id
            assert loaded.tables == tables
            assert loaded.queries == (benchmark.get_query("IMDB-4"),)
        filtered = benchmark.load_benchmark(
            "IMDB-1", scale_factor=sf, root=tmp_path, accuracy=False)
        assert set(filtered.tables) == {"reviews"}
        pq.write_table(tables["reviews"].slice(0, 1), directory / "reviews.parquet")
        with pytest.raises(ValueError, match="reviews does not match"):
            benchmark.load_benchmark(
                "IMDB-4", scale_factor=sf, root=tmp_path, accuracy=False)
    for ids in (["unknown"], [], ["IMDB-1", "IMDB-1"]):
        with pytest.raises(ValueError):
            benchmark.select_queries(ids)
    with pytest.raises(ValueError, match="scale factor"):
        benchmark.select_queries(scale_factor=0.2)


def test_current_benchmark_rejects_chat_reference_labels(
        monkeypatch, tmp_path):
    from quail_b import benchmark as loading
    from quail_b.data import DATA_SEED, SOURCE_REVISIONS, corpus_identity
    from quail_b.labels import GroundTruthCollection, PredicateLabels
    from quail_b.predicates import PREDICATES, predicate_payload

    spec = PREDICATES[0]
    tables = {"reviews": pa.table({"id": ["r0"], "body": ["a good movie"]})}
    corpus_id = PUBLISHED_CORPORA[0.1]
    directory = tmp_path / GROUND_TRUTH_ROOT / "corpora" / corpus_id
    directory.mkdir(parents=True)
    manifest = corpus_identity(tables, 0.1, DATA_SEED, SOURCE_REVISIONS)
    manifest["corpus_id"] = corpus_id
    (directory / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(tables["reviews"], directory / "reviews.parquet")
    payload = predicate_payload(spec)
    labels = PredicateLabels(
        spec.key, "ls_old", {"template": spec.template},
        {("r0", None): True}, {},
        predicate_payload={**payload, "prompt_format": "qwen3-chat-nonthinking-v1"})
    truth = GroundTruthCollection(
        "gt_test", corpus_id, 0.1, "qwen3-32b-fp8", {spec.key: labels})
    monkeypatch.setattr(loading, "load_ground_truth", lambda *a, **kw: truth)
    with pytest.raises(ValueError, match="different prompt format"):
        loading.load_benchmark("IMDB-1", root=tmp_path)
    labels.predicate_payload = payload
    assert loading.load_benchmark("IMDB-1", root=tmp_path).ground_truth is truth


def test_prompt_format_selects_its_own_active_collection(tmp_path):
    from quail_b.labels import _choose_collection

    root = tmp_path / GROUND_TRUTH_ROOT
    corpus = root / "corpora" / "c_test"
    corpus.mkdir(parents=True)
    for name, collection_id in (
            ("active_collection.json", "gt_raw"),
            ("active_collection.chat.json", "gt_chat")):
        (corpus / name).write_text(json.dumps({"collection_id": collection_id}))
        directory = root / "collections" / collection_id
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({
            "collection_id": collection_id, "corpus_id": "c_test",
            "scale_factor": 0.1, "status": "complete"}))
    for explicit, prompt_format, expected in (
            (None, None, "gt_raw"), (None, "chat", "gt_chat"),
            (None, "unpublished", "gt_raw"), ("gt_raw", "chat", "gt_raw")):
        _, manifest = _choose_collection(
            tmp_path, 0.1, "c_test", explicit, prompt_format)
        assert manifest["collection_id"] == expected
