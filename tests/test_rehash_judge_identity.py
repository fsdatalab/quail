"""CPU checks for the one-shot judge-identity rehash migration."""

from quail.bench.judge_pass import (
    MODEL_NAME,
    PREDICATES,
    _label_dir_by_id,
    judgment_identity,
    label_set_identity,
)
from migrations.rehash_judge_identity import _rehash_label_dir


def _spec(key):
    return next(spec for spec in PREDICATES if spec.key == key)


def test_rehash_label_dir_rebuilds_parts_from_the_compacted_file(
        tmp_path, monkeypatch):
    """The old parts directory holds two generations of files, so the
    rehash has to read labels.parquet and rebuild the parts itself."""
    import pyarrow.parquet as pq

    from quail.bench import judge_pass

    monkeypatch.setattr(judge_pass, "VOLUME_ROOT", tmp_path)
    spec = _spec("quailb.imdb.review.discusses_ending")
    old_identity = label_set_identity(spec, "c_test", "f" * 64)
    new_identity = dict(old_identity, label_set_id="ls_new")
    rows = {"reviews": [{"id": f"rv{i}"} for i in range(170)]}

    old_dir = _label_dir_by_id(spec, old_identity["label_set_id"])
    (old_dir / "parts").mkdir(parents=True)
    labels = [{
        "judgment_id": f"jd_{i}", "example_id": f"ex_{i}",
        "example_full_hash": f"{i:064d}",
        "label_set_id": old_identity["label_set_id"],
        "predicate_key": spec.key,
        "predicate_version": old_identity["predicate_version"],
        "answer": i % 2 == 0, "label_source": MODEL_NAME,
        "left_role": spec.left_role, "left_table": spec.left_table,
        "left_id": f"rv{i}", "left_content_sha256": f"{i:064x}",
        "right_role": None, "right_table": None, "right_id": None,
        "right_content_sha256": None, "selected_token_id": 1,
    } for i in range(170)]
    judge_pass._atomic_parquet(old_dir / "labels.parquet", labels)
    # a leftover generation, under boundaries nothing writes any more
    judge_pass._atomic_parquet(
        old_dir / "parts" / "part_000000_000170.parquet", labels)

    stats = _rehash_label_dir(spec, old_dir, new_identity, rows, 170)

    assert stats["rows"] == 170
    assert stats["true_rows"] == 85
    new_dir = _label_dir_by_id(spec, "ls_new")
    written = sorted(p.name for p in (new_dir / "parts").glob("*.parquet"))
    assert written[0] == "part_000000_000085.parquet"
    assert written[-1] == "part_000085_000170.parquet"
    assert len(written) == 2
    rebuilt = (pq.read_table(new_dir / "parts" / written[0]).to_pylist()
               + pq.read_table(new_dir / "parts" / written[1]).to_pylist())
    assert [r["left_id"] for r in rebuilt] == [r["left_id"] for r in labels]
    assert [r["answer"] for r in rebuilt] == [r["answer"] for r in labels]
    assert {r["label_set_id"] for r in rebuilt} == {"ls_new"}
    assert rebuilt[7]["judgment_id"] == judgment_identity(
        "ls_new", labels[7]["example_full_hash"])
    compact = pq.read_table(new_dir / "labels.parquet").to_pylist()
    assert [r["judgment_id"] for r in compact] == [
        r["judgment_id"] for r in rebuilt]
