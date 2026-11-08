"""CPU checks for predicted pass times built from measured ratios."""

import json

import numpy as np
import pytest

from quail.cost import predict
from quail.cost.budgets import chunk_budget
from quail.cost.predict import PassTable, predict_seconds, split_passes
from quail.cost.sol import speed_of_light
from quail.cost.work import scan
from quail.specs.h100_sxm import H100_SXM
from quail.specs.qwen3_4b import QWEN3_4B_FP8

CHUNK = chunk_budget(QWEN3_4B_FP8, H100_SXM)


def _table(tokens, ratios, model="qwen3-4b-fp8", device="h100-sxm"):
    return PassTable(
        model=model, device=device, tokens=np.asarray(tokens, dtype=float),
        pairs_per_token=np.array([0.0]), kv_read_per_token=np.array([0.0]),
        ratios=np.asarray(ratios, dtype=float).reshape(len(tokens), 1, 1),
        source="test")


def test_a_flat_table_scales_the_speed_of_light_by_its_ratio():
    work = scan(300, 50) * (3 * CHUNK / 350)
    ideal = speed_of_light(work, QWEN3_4B_FP8, H100_SXM, CHUNK).seconds
    table = _table([1, 1e9], [2.0, 2.0])
    seconds = predict_seconds(table, work, QWEN3_4B_FP8, H100_SXM, CHUNK)
    assert seconds == pytest.approx(2.0 * ideal, rel=1e-9)


def test_the_ratio_changes_linearly_in_the_log_of_the_tokens():
    table = _table([100, 10_000], [4.0, 2.0])
    middle = scan(900, 100)  # 1,000 tokens
    assert table.ratio(middle) == pytest.approx(3.0)


def test_a_pass_outside_the_grid_takes_the_nearest_measured_ratio():
    table = _table([1_000, 10_000], [4.0, 2.0])
    assert table.ratio(scan(10, 10)) == pytest.approx(4.0)
    assert table.ratio(scan(50_000, 50_000)) == pytest.approx(2.0)


def test_small_passes_cost_more_per_ideal_second_than_large_ones():
    table = _table([100, 100_000], [5.0, 1.5])
    small = predict_seconds(table, scan(200, 50), QWEN3_4B_FP8, H100_SXM, CHUNK)
    ideal = speed_of_light(scan(200, 50), QWEN3_4B_FP8, H100_SXM, CHUNK).seconds
    assert small / ideal > 3.0


def test_work_splits_into_full_passes_and_one_partial_pass():
    work = scan(500, 100) * (2.5 * CHUNK / 600)
    (full, full_count), (partial, partial_count) = split_passes(work, CHUNK)
    assert (full_count, partial_count) == (2.0, 1.0)
    assert full.tokens == pytest.approx(CHUNK)
    assert partial.tokens == pytest.approx(0.5 * CHUNK)
    total = full * full_count + partial * partial_count
    assert total.pairs == pytest.approx(work.total().pairs)
    assert total.kv_written == pytest.approx(work.total().kv_written)


def test_work_without_tokens_has_no_passes():
    assert split_passes(scan(0, 0) * 0, CHUNK) == []


def test_a_saved_table_loads_and_a_missing_one_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "TABLE_DIR", tmp_path)
    predict.load_table.cache_clear()
    assert predict.load_table("qwen3-4b-fp8", "h100-sxm") is None
    table = _table([100, 10_000], [4.0, 2.0])
    (tmp_path / "qwen3-4b-fp8__h100-sxm.json").write_text(
        json.dumps(table.to_dict()))
    predict.load_table.cache_clear()
    loaded = predict.load_table("qwen3-4b-fp8", "h100-sxm")
    assert loaded.ratio(scan(900, 100)) == pytest.approx(3.0)
    predict.load_table.cache_clear()


def test_a_table_measured_on_another_device_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "TABLE_DIR", tmp_path)
    predict.load_table.cache_clear()
    other = _table([100, 10_000], [4.0, 2.0], device="l40s")
    (tmp_path / "qwen3-4b-fp8__h100-sxm.json").write_text(
        json.dumps(other.to_dict()))
    with pytest.raises(ValueError, match="l40s"):
        predict.load_table("qwen3-4b-fp8", "h100-sxm")
    predict.load_table.cache_clear()


@pytest.mark.parametrize("ratios", [[0.0, 2.0], [float("nan"), 2.0], [-1.0, 2.0]])
def test_ratios_must_be_positive_and_finite(ratios):
    with pytest.raises(ValueError, match="positive and finite"):
        _table([100, 10_000], ratios)


def test_explain_shows_predicted_time_only_when_a_table_exists(tmp_path, monkeypatch):
    import pyarrow as pa

    import quail
    from quail.planner.plan import EngineConfig

    monkeypatch.setattr(predict, "TABLE_DIR", tmp_path)
    predict.load_table.cache_clear()
    sql = ("SELECT r.id FROM reviews r "
           "WHERE AI_FILTER(PROMPT('Good? {0}', r.body))")

    def explained():
        with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                           tokenizer=str.split) as session:
            session.register("reviews", quail.DocumentProvider.from_table(
                pa.table({"id": ["a", "b"], "body": ["good film", "bad film"]}),
                id_col="id"))
            return session.sql(sql).explain()

    assert "pred. time" not in explained()
    (tmp_path / "qwen3-4b-fp8__h100-sxm.json").write_text(
        json.dumps(_table([1, 1e9], [2.0, 2.0]).to_dict()))
    predict.load_table.cache_clear()
    assert "pred. time" in explained()
    predict.load_table.cache_clear()
