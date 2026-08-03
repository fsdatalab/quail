import gzip
import json

from scripts.build_batch_catalog import build_catalog, load_kernel_rows


def test_build_batch_catalog_from_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "run",
        "phase": "cascade-kernel",
    }))
    with gzip.open(run / "result.json.gz", "wt") as handle:
        json.dump({
            "groups": 8,
            "k": 2,
            "prefix_tokens": 304,
            "tail_tokens": 32,
            "cascade_ms": 0.1,
            "standard_ms": 0.08,
            "speedup": 0.8,
            "max_absolute_difference": 0.001,
        }, handle)
    rows = load_kernel_rows(tmp_path)
    catalog = build_catalog(rows)
    assert rows[0]["cascade_ns"] == 100_000
    assert catalog["kv_cache_dtype"] == "fp8"
    assert catalog["validation"]["status"] == "needs-held-out-runs"
