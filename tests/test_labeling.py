"""CPU checks for the QUAIL-B labeling pass."""

from pathlib import Path

import quail.bench.labeling as labeling
from quail.bench.labeling import (
    _check_reused_label_set,
    _compact_label_parts,
    _corpus_identity,
    _lepard_source_answer,
    _saved_verification_sample,
)
from quail_b.data import DATA_SEED, SOURCE_REVISIONS, corpus_identity
from quail_b.predicates import MODEL_NAME, PREDICATES


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_corpus_identity_matches_the_data_module():
    rows = {"reviews": [{"id": "r0", "body": "text"}]}
    assert (corpus_identity(rows, 0.1, DATA_SEED, SOURCE_REVISIONS)
            == _corpus_identity(rows, 0.1))


def test_corpus_identity_uses_source_rows_and_order():
    rows = {
        "reviews": [{"id": "r0", "body": "a"},
                    {"id": "r1", "body": "b"}],
    }
    same = {"reviews": list(rows["reviews"])}
    reversed_rows = {"reviews": list(reversed(rows["reviews"]))}
    changed = {"reviews": [{"id": "r0", "body": "a"},
                           {"id": "r1", "body": "changed"}]}
    assert _corpus_identity(rows, 0.1) == _corpus_identity(same, 0.1)
    assert (_corpus_identity(rows, 0.1)["corpus_id"]
            != _corpus_identity(reversed_rows, 0.1)["corpus_id"])
    assert (_corpus_identity(rows, 0.1)["corpus_id"]
            != _corpus_identity(changed, 0.1)["corpus_id"])


def test_reuse_checks_the_label_sets_original_corpus(monkeypatch, tmp_path):
    import json

    spec = _spec("quailb.imdb.review.mentions_positive_aspect")
    label_set_id = "ls_old"
    table_manifest = {"rows": 2, "ordered_rows_full_hash": "abc"}
    corpus = {
        "corpus_id": "c_original",
        "corpus_full_hash": "full-original",
        "tables": {"reviews": table_manifest},
    }
    corpus_dir = tmp_path / "corpora" / corpus["corpus_id"]
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text(json.dumps(corpus))
    label_dir = (
        tmp_path / "label_sets" / spec.workload / spec.slug / label_set_id
    )
    label_dir.mkdir(parents=True)
    (label_dir / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "label_set_id": label_set_id,
        "corpus_id": corpus["corpus_id"],
        "corpus_full_hash": corpus["corpus_full_hash"],
    }))
    source_collection = {
        "collection_id": "gt_middle",
        "corpus_id": "c_middle",
        "label_sets": {spec.key: label_set_id},
    }
    target = {"tables": {"reviews": dict(table_manifest)}}
    monkeypatch.setattr(labeling, "ROOT", tmp_path)

    reused = _check_reused_label_set(
        spec, label_set_id, source_collection, target)

    assert reused["source_collection_id"] == "gt_middle"
    assert reused["source_corpus_id"] == "c_original"


def test_reuse_rejects_changed_table_in_target(monkeypatch, tmp_path):
    import json

    spec = _spec("quailb.imdb.review.mentions_positive_aspect")
    label_set_id = "ls_old"
    corpus = {
        "corpus_id": "c_original",
        "corpus_full_hash": "full-original",
        "tables": {
            "reviews": {"rows": 2, "ordered_rows_full_hash": "abc"},
        },
    }
    corpus_dir = tmp_path / "corpora" / corpus["corpus_id"]
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text(json.dumps(corpus))
    label_dir = (
        tmp_path / "label_sets" / spec.workload / spec.slug / label_set_id
    )
    label_dir.mkdir(parents=True)
    (label_dir / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "label_set_id": label_set_id,
        "corpus_id": corpus["corpus_id"],
        "corpus_full_hash": corpus["corpus_full_hash"],
    }))
    source_collection = {
        "collection_id": "gt_middle",
        "label_sets": {spec.key: label_set_id},
    }
    target = {
        "tables": {
            "reviews": {"rows": 2, "ordered_rows_full_hash": "changed"},
        },
    }
    monkeypatch.setattr(labeling, "ROOT", tmp_path)

    try:
        _check_reused_label_set(
            spec, label_set_id, source_collection, target)
    except ValueError as error:
        assert "table reviews changed" in str(error)
    else:
        raise AssertionError("changed table was reused")


def test_lepard_source_answer_uses_sampled_citation_edges():
    assert _lepard_source_answer(["p1", "p2"], ["p2", "p3"])
    assert not _lepard_source_answer(["p1", "p2"], ["p3"])


def test_saved_verification_sample_covers_completed_parts_after_resume(
        monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    spec = _spec("quailb.imdb.review.discusses_ending")
    identity = {"label_set_id": "ls_test"}
    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    parts = (tmp_path / "label_sets" / spec.workload / spec.slug
             / identity["label_set_id"] / "parts")
    parts.mkdir(parents=True)
    saved = [{
        "answer": i % 2 == 0,
        "label_source": MODEL_NAME,
        "left_id": f"rv{i}",
        "right_id": None,
    } for i in range(85)]
    # only the first of two parts is on disk, the resume case
    pq.write_table(pa.Table.from_pylist(saved),
                   parts / "part_000000_000085.parquet")
    corpus = {
        "reviews": [{"id": f"rv{i}", "body": f"review {i}"}
                    for i in range(170)]
    }

    sample = _saved_verification_sample(
        corpus, {spec.key: identity}, specs=(spec,))

    assert len(sample.rows[spec.key]) == 16
    assert sample.rows[spec.key][0] == (
        {"id": "rv0", "body": "review 0"}, None, True)


def test_compact_label_parts_keeps_every_saved_row(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    pq.write_table(pa.table({"id": ["a", "b"], "answer": [True, False]}),
                   parts_dir / "part_000000_000002.parquet")
    pq.write_table(pa.table({"id": ["c"], "answer": [True]}),
                   parts_dir / "part_000002_000003.parquet")
    # a leftover generation of boundaries, which the caller does not
    # name and compaction must therefore ignore
    pq.write_table(pa.table({"id": ["a", "b", "c"],
                             "answer": [True, False, True]}),
                   parts_dir / "part_000000_000003.parquet")
    parts = [parts_dir / "part_000000_000002.parquet",
             parts_dir / "part_000002_000003.parquet"]

    path, rows = _compact_label_parts(tmp_path, parts)
    second_path, second_rows = _compact_label_parts(tmp_path, parts)

    assert rows == second_rows == 3
    assert path == second_path == tmp_path / "labels.parquet"
    assert pq.read_table(path).to_pylist() == [
        {"id": "a", "answer": True},
        {"id": "b", "answer": False},
        {"id": "c", "answer": True},
    ]


class _FakeResult:
    def __init__(self, answer_tables):
        self.answer_tables = answer_tables


class _FakeQuery:
    """Stands in for a bound builder; answers TRUE when the text has 'yes'."""

    def __init__(self, session, name):
        self.session = session
        self.names = [name]

    def alias(self, _alias):
        return self

    def ai_filter(self, _prompt, **_kwargs):
        return self

    def ai_join(self, other, _prompt, **_kwargs):
        self.names.append(other.names[0])
        return self

    def select(self, *_columns):
        return self

    def run(self):
        import pyarrow as pa

        tables = [self.session.tables[name] for name in self.names]
        if len(tables) == 1:
            text = tables[0].column(1).to_pylist()
            table = pa.table({"l": list(range(len(text))),
                              "answer": ["yes" in value for value in text]})
            return _FakeResult({"filters": {("l", 0): table}, "joins": {}})
        left, right = (table.column(1).to_pylist() for table in tables)
        rows = [(i, j, "yes" in a and "yes" in b)
                for i, a in enumerate(left) for j, b in enumerate(right)]
        table = pa.table({"l": [r[0] for r in rows], "r": [r[1] for r in rows],
                          "answer": [r[2] for r in rows]})
        return _FakeResult({"filters": {}, "joins": {0: table}})


class _FakeSession:
    def __init__(self, _config):
        self.tables = {}

    def register(self, name, provider):
        self.tables[name] = provider.table

    def docs(self, name):
        return _FakeQuery(self, name)

    def close(self):
        pass


def test_quail_judge_maps_answers_back_to_rows(monkeypatch):
    import quail

    monkeypatch.setattr(quail, "Session", _FakeSession)
    judge = labeling.QuailJudge()
    spec = _spec("quailb.imdb.review.discusses_aspect")
    reviews = [{"id": "rv0", "body": "no"}, {"id": "rv1", "body": "yes"}]
    aspects = [{"id": "as0", "aspect": "yes the plot"},
               {"id": "as1", "aspect": "the acting"}]

    assert judge.filter(_spec("quailb.imdb.review.discusses_ending"),
                        reviews) == [False, True]
    assert judge.join(spec, reviews, aspects) == {
        (0, 0): False, (0, 1): False, (1, 0): True, (1, 1): False}
    assert judge.queries == 2
    assert judge.rows_answered == 6


def test_join_parts_keep_source_labels_over_model_answers(
        monkeypatch, tmp_path):
    import pyarrow.parquet as pq

    import quail

    monkeypatch.setattr(quail, "Session", _FakeSession)
    monkeypatch.setattr(labeling, "ROOT", tmp_path)
    judge = labeling.QuailJudge()
    spec = _spec("quailb.fever.passage.supports_claim")
    claims = [{"id": "cl0", "claim": "yes one", "label": "REFUTES",
               "evidence_wiki_url": "page0"},
              {"id": "cl1", "claim": "two", "label": "SUPPORTS",
               "evidence_wiki_url": "page1"}]
    evidence = [{"id": "page0", "text": "yes"}, {"id": "page1", "text": "no"}]
    identity = labeling.label_set_identity(spec, "c_test", "0" * 64)

    labeling._write_qwen_join_parts(
        judge, labeling.VerificationSample(), spec, claims, evidence,
        identity, "c_test", source_label=labeling._fever_source_label)

    parts = labeling._expected_parts(
        spec, identity, {"claims": claims, "evidence": evidence})
    assert len(parts) == 1 and parts[0].exists()
    rows = {(row["left_id"], row["right_id"]): (row["answer"],
                                                 row["label_source"])
            for row in pq.read_table(parts[0]).to_pylist()}
    assert rows == {
        ("cl0", "page0"): (False, "fever_annotation"),   # source wins
        ("cl0", "page1"): (False, MODEL_NAME),
        ("cl1", "page0"): (False, MODEL_NAME),
        ("cl1", "page1"): (True, "fever_annotation"),
    }


class _FakeS3:
    """Records uploads by key; lists what it holds like the real client."""

    def __init__(self):
        self.objects = {}

    def get_paginator(self, _name):
        fake = self

        class _Pages:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                contents = [{"Key": k, "Size": len(v)}
                            for k, v in fake.objects.items()
                            if k.startswith(Prefix)]
                return [{"Contents": contents}]
        return _Pages()

    def upload_file(self, path, _bucket, key):
        self.objects[key] = Path(path).read_bytes()


class _DictFiles:
    """The reader side of the round trip: a store over the fake bucket."""

    def __init__(self, objects):
        self.objects = objects

    def read_bytes(self, path):
        return self.objects[path.lstrip("/")]

    def list_files(self, path):
        prefix = path.lstrip("/").rstrip("/") + "/"
        return sorted(k for k in self.objects if k.startswith(prefix))


def _label_tree(root):
    """A complete collection with one filter predicate, compacted."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    key, label_set_id = "test.review.filter", "ls_filter"
    predicate = {"key": key, "template": "Judge {0}. TRUE or FALSE.",
                 "kind": "filter", "left_table": "reviews",
                 "left_column": "body", "right_table": None,
                 "right_column": None}
    collection_dir = root / "collections" / "gt_test"
    label_dir = root / "label_sets" / "test" / "review_filter" / label_set_id
    corpus_dir = root / "corpora" / "c_test"
    for directory in (collection_dir, label_dir / "parts", corpus_dir):
        directory.mkdir(parents=True)
    (collection_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "collection_id": "gt_test",
        "corpus_id": "c_test", "scale_factor": 0.1,
        "label_sets": {key: label_set_id},
        "summary": {"model": "qwen3-32b-fp8"}}))
    (corpus_dir / "active_collection.json").write_text(
        json.dumps({"collection_id": "gt_test"}))
    (label_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "rows": 2,
        "source_rows": {"qwen3-32b-fp8": 2}, "predicate": predicate}))
    rows = pa.Table.from_pylist([
        {"predicate_key": key, "label_set_id": label_set_id,
         "answer": True, "left_id": "r0", "right_id": None},
        {"predicate_key": key, "label_set_id": label_set_id,
         "answer": False, "left_id": "r1", "right_id": None}])
    pq.write_table(rows, label_dir / "labels.parquet")
    pq.write_table(rows, label_dir / "parts" / "part_000000_000002.parquet")
    (label_dir / "labels.parquet.tmp").write_bytes(b"half written")
    return key


def test_publish_writes_where_the_loader_reads(tmp_path):
    from quail_b.labels import load_ground_truth
    from quail_b.store import GROUND_TRUTH_ROOT

    root = tmp_path / GROUND_TRUTH_ROOT
    key = _label_tree(root)
    bucket = _FakeS3()

    uploaded = labeling.publish(root, bucket="b", client=bucket)

    assert uploaded == 4      # two manifests, the active pointer, labels
    assert all(k.startswith(GROUND_TRUTH_ROOT + "/") for k in bucket.objects)
    assert not any("/parts/" in k or k.endswith(".tmp") for k in bucket.objects)
    loaded = load_ground_truth(_DictFiles(bucket.objects), scale_factor=0.1,
                               corpus_id="c_test")
    assert loaded.collection_id == "gt_test"
    assert loaded.answer(key, "r0") is True
    assert loaded.answer(key, "r1") is False
    # a second publish finds everything in place and uploads nothing
    assert labeling.publish(root, bucket="b", client=bucket) == 0
