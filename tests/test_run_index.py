import gzip
import json

from scripts.build_run_index import load_runs, markdown


def test_run_index_reads_immutable_directory(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "run-1",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "git_commit": "abcdef012345",
        "phase": "custom-smoke",
        "config": {
            "n_docs": 8,
            "n_filters": 2,
            "k": 1,
            "document_tokens": 300,
        },
    }))
    with gzip.open(run / "result.json.gz", "wt") as handle:
        json.dump({
            "wall_ns": 1000,
            "steps": 3,
            "accuracy": 1.0,
            "ground_truth_used_by_runtime": False,
        }, handle)
    rows = load_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["wall_ns"] == 1000
    assert rows[0]["ground_truth_used_by_runtime"] is False
    rendered = markdown(rows)
    assert "run-1" in rendered
    assert "100.00%" in rendered
