"""CPU checks for the pass measurements and the fitted table."""

import json
import math

import numpy as np
import pytest

from calibrate import passes
from quail.cost.budgets import chunk_budget
from quail.cost.predict import PassTable
from quail.specs.h100_sxm import H100_SXM
from quail.specs.qwen3_4b import QWEN3_4B_FP8

MODEL, DEVICE = QWEN3_4B_FP8, H100_SXM


def _record(words, num_documents, ratio):
    tokens = num_documents * (words * 1.3 + 51)
    metrics = {"filter": {"chunks": 1, "fresh_tokens": tokens,
                          "wall_s": 1.0, "gpu_s": 0.5}}
    record = passes.record_from_metrics(MODEL, DEVICE, words, num_documents, metrics)
    return {**record, "wall_seconds": record["sol_seconds"] * ratio}


def test_the_grid_leaves_out_points_that_are_too_large():
    points = passes.grid()
    assert (2400, 4096) not in points
    assert (20, 1) in points
    assert all(words * num <= passes.MAX_WORDS for words, num in points)


def test_synthetic_documents_have_the_requested_words_and_differ():
    table = passes.synthetic_table(30, 3)
    texts = table.column("text").to_pylist()
    assert len(set(texts)) == 3
    assert all(len(text.split()) == 32 for text in texts)


def test_a_record_prices_the_pass_in_ideal_seconds():
    record = _record(300, 64, 2.0)
    assert record["chunk_tokens"] == chunk_budget(MODEL, DEVICE)
    assert record["wall_seconds"] == pytest.approx(2.0 * record["sol_seconds"])
    assert record["tokens"] == record["fresh_tokens"]


def test_a_run_without_forward_passes_is_refused():
    with pytest.raises(ValueError, match="forward pass"):
        passes.record_from_metrics(MODEL, DEVICE, 10, 1, {"scan": {"chunks": 0}})


def test_the_fit_recovers_a_ratio_that_falls_with_pass_size(tmp_path):
    pytest.importorskip("sklearn")
    from calibrate import fit_passes

    def truth(tokens):
        return 1.2 + 6.0 / math.sqrt(tokens / 100)

    records = []
    for words, num_documents in passes.grid():
        record = _record(words, num_documents, 1.0)
        record["wall_seconds"] = record["sol_seconds"] * truth(record["tokens"])
        records.append(record)
    path = tmp_path / "m.jsonl"
    passes.save(path, records)
    loaded = fit_passes.load(path)
    error = fit_passes.cross_validate(loaded)
    assert error["forest_median"] < 0.1
    assert error["forest_median"] < error["constant_median"]
    table = fit_passes.fit_table(loaded, "test")
    assert isinstance(table, PassTable)
    assert np.all(table.ratios > 1.0)
    assert json.loads(json.dumps(table.to_dict()))["model"] == "qwen3-4b-fp8"
