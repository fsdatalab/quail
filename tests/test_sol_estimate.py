"""CPU checks for the public speed of light estimate."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

import quail
from quail.catalog import DocumentProvider
from quail.planner.plan import EngineConfig
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_USD_PER_HOUR

FILTER = "Judge the review.\n\n{0}\nAnswer TRUE or FALSE."
JOIN = "Judge the pair.\n\n{0}\nAspect: {1}\nAnswer TRUE or FALSE."


def _session(tmp_path):
    pq.write_table(pa.table({
        "id": ["r0", "r1", "r2"],
        "body": ["good film", "bad film", "good acting"],
    }), tmp_path / "reviews.parquet")
    pq.write_table(pa.table({
        "id": ["a0", "a1"],
        "aspect": ["acting", "ending"],
    }), tmp_path / "aspects.parquet")
    sess = quail.Session(
        EngineConfig(
            gpus=1,
            model="qwen3-4b-fp8",
            backend="quail",
            device="h100-sxm",
        ),
        tokenizer=str.split,
    )
    sess.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    sess.register("aspects", DocumentProvider.from_parquet(
        str(tmp_path / "aspects.parquet"), id_col="id"))
    return sess


def _answer(prompt, assignment):
    if len(prompt.args) == 1:
        return assignment["r"] != 1
    return (assignment["r"], assignment["a"]) == (0, 0)


def test_distinct_prefix_estimates_for_filters_and_joins(tmp_path):
    sess = _session(tmp_path)
    try:
        query = (sess.docs("reviews").alias("r")
                 .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
                 .select("r.id"))
        filters = query.logical.operators().filters
        question = filters["r"][0].prompt.tail_tokens
        preamble = filters["r"][0].prompt.preamble_tokens
        distinct = quail.speed_of_light_estimate(query, _answer)
        per_document = quail.speed_of_light_estimate(
            query, _answer, credit_shared_prefixes=False)
    finally:
        sess.close()

    # "good film" and "good acting" share the preamble and one document token.
    assert per_document.fresh_tokens == 3 * (preamble + 2 + question)
    assert distinct.fresh_tokens == per_document.fresh_tokens - preamble - 1
    assert distinct.filter_stages[0]["evaluated"] == 3
    assert distinct.filter_stages[0]["passed"] == 2
    assert distinct.post_filter_counts == {"r": 2}
    assert distinct.join_stages == ()
    assert distinct.seconds > 0
    assert distinct.usd_per_query == (
        distinct.seconds * H100_USD_PER_HOUR / 3600)
    assert distinct.model == "qwen3-4b-fp8"
    assert distinct.device == "h100-sxm"

    sess = _session(tmp_path)
    try:
        query = (sess.docs("reviews").alias("r")
                 .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
                 .ai_join(
                     sess.docs("aspects").alias("a"),
                     quail.prompt(JOIN, quail.col("r.body"),
                                  quail.col("a.aspect")))
                 .select("r.id", "a.id"))
        estimate = quail.speed_of_light_estimate(query, _answer)
    finally:
        sess.close()

    assert len(estimate.join_stages) == 1
    stage = estimate.join_stages[0]
    assert stage["evaluated_pairs"] == 2 * 2
    assert stage["passing_pairs"] == 1
    assert stage["anchor"] in {"r", "a"}
    assert set(estimate.relation_order) == {"r", "a"}
    assert estimate.join_pair_evaluations == 4
    assert estimate.input_document_rows == 5
    assert estimate.search["dp_final_records"] >= 1
    assert estimate.assumptions()["persistent_kv_capacity"] == "unlimited"
    record = json.loads(json.dumps(estimate.as_dict()))
    assert record["sol_s"] == estimate.seconds
    assert record["join_stages"][0]["template"] == stage["template"]


def test_canvas_rows_count_once_per_evaluation(tmp_path):
    sess = _session(tmp_path)
    try:
        query = (sess.docs("reviews").alias("r")
                 .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
                 .ai_join(
                     sess.docs("aspects").alias("a"),
                     quail.prompt(JOIN, quail.col("r.body"),
                                  quail.col("a.aspect")))
                 .select("r.id", "a.id"))
        decoder = quail.speed_of_light_estimate(query, _answer)
        canvas = quail.speed_of_light_estimate(
            query, _answer,
            model=replace(DIFFUSION_GEMMA_26B_FP8, canvas_tokens=8))
    finally:
        sess.close()

    # three filter evaluations and four join pairs each carry the
    # eight canvas rows; the anchor frames carry none
    assert canvas.fresh_tokens == decoder.fresh_tokens + 8 * (3 + 4)
    assert canvas.join_pair_evaluations == 4
