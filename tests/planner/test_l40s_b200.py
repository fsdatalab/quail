"""L40S and B200 plans, KV budgets, and $/query without a GPU."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import letter_tokens

import quail
from bench.quailb import queries, register_tables, run_suite
from quail.cost import budgets
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import (
    B200,
    DEVICES,
    DIFFUSION_GEMMA_26B_FP8,
    H100_SXM,
    L40S,
    modal_gpu_type,
)


def test_modal_gpu_type_appends_count_for_eight_l40s():
    assert modal_gpu_type("l40s") == "L40S"
    assert modal_gpu_type("l40s", 8) == "L40S:8"


def _standin_filter_and_join(tmp_path):
    reviews = pa.table({
        "id": [f"r{i}" for i in range(4)],
        "body": [f"review text {i}" for i in range(4)],
    })
    claims = pa.table({
        "id": [f"c{i}" for i in range(4)],
        "claim": [f"claim {i}" for i in range(4)],
        "label": ["SUPPORTS", "REFUTES", "SUPPORTS", "REFUTES"],
        "evidence_wiki_url": ["e0", "e1", "e0", "e1"],
    })
    evidence = pa.table({
        "id": ["e0", "e1"],
        "text": ["Wikipedia passage 0", "Wikipedia passage 1"],
    })
    pq.write_table(reviews, tmp_path / "reviews.parquet")
    pq.write_table(claims, tmp_path / "claims.parquet")
    pq.write_table(evidence, tmp_path / "evidence.parquet")


def test_diffusion_gemma_refuses_l40s_at_default_chunk(tmp_path):
    assert budgets.minimum_weight_gpus(DIFFUSION_GEMMA_26B_FP8, L40S) == 1
    assert budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, L40S) <= (
        DIFFUSION_GEMMA_26B_FP8.chunk_cap_tokens)
    with pytest.raises(ValueError, match="one chunk's sliding KV"):
        budgets.arena_pages(DIFFUSION_GEMMA_26B_FP8, L40S)
    assert budgets.arena_tokens(DIFFUSION_GEMMA_26B_FP8, B200) > (
        budgets.arena_tokens(DIFFUSION_GEMMA_26B_FP8, H100_SXM))

    _standin_filter_and_join(tmp_path)
    with quail.Session(
        EngineConfig(model="diffusion-gemma-26b-a4b-fp8", device="l40s"),
        tokenizer=letter_tokens,
    ) as sess:
        register_tables(sess, tmp_path)
        plan = queries(sess)["IMDB-1"][1]().plan()
    assert isinstance(plan, Refusal)
    assert plan.constraint == "flash_attention_4_unsupported"
    assert plan.needed > plan.available
    assert "l40s" in plan.reasons[0]


def test_filter_and_join_plan_on_l40s_and_b200(tmp_path):
    _standin_filter_and_join(tmp_path)
    for device in ("l40s", "b200"):
        with quail.Session(
            EngineConfig(model="qwen3-4b-fp8", device=device),
            tokenizer=letter_tokens,
        ) as sess:
            register_tables(sess, tmp_path)
            listed = queries(sess)
            for query_id in ("IMDB-1", "FEV-1"):
                plan = listed[query_id][1]().plan()
                assert not isinstance(plan, Refusal), f"{device} {query_id}"


@pytest.mark.parametrize("device", ("l40s", "b200"))
def test_hybrid_filter_plans_with_fp32_convolution_cost(tmp_path, device):
    _standin_filter_and_join(tmp_path)
    with quail.Session(
        EngineConfig(model="qwen3.5-4b-fp8", device=device),
        tokenizer=letter_tokens,
    ) as sess:
        register_tables(sess, tmp_path)
        plan = queries(sess)["IMDB-1"][1]().plan()
    if device == "l40s":
        assert isinstance(plan, Refusal)
        assert plan.constraint == "flash_attention_4_unsupported"
    else:
        assert not isinstance(plan, Refusal)


def test_run_suite_uses_the_device_hourly_price(tmp_path, monkeypatch):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {"queries": []}

    class FakeSession:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("bench.quailb.benchmark.run", fake_run)
    monkeypatch.setattr(
        "bench.quailb.benchmark.report", lambda *a, **k: None)
    monkeypatch.setattr("bench.quailb.quail.Session", FakeSession)

    for device in ("l40s", "b200", "h100-sxm"):
        captured.clear()
        run_suite(
            only=["IMDB-1"],
            config=EngineConfig(model="qwen3-4b-fp8", device=device),
            output_dir=tmp_path / device,
        )
        assert captured["gpu_hourly_rate_usd"] == DEVICES[device].usd_per_hour
