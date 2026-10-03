"""CPU checks for the public speed of light estimate."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
        EngineConfig(gpus=1, model="qwen3-4b-fp8", device="h100-sxm"),
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
    with _session(tmp_path) as sess:
        filtered = (sess.docs("reviews").alias("r")
                    .ai_filter(quail.prompt(FILTER, quail.col("r.body"))))
        query = filtered.select("r.id")
        filters = query.logical.operators().filters
        question = filters["r"][0].prompt.tail_tokens
        preamble = filters["r"][0].prompt.preamble_tokens
        distinct = quail.speed_of_light_estimate(query, _answer)
        per_document = quail.speed_of_light_estimate(
            query, _answer, credit_shared_prefixes=False)
        query = (filtered
                 .ai_join(sess.docs("aspects").alias("a"),
                          quail.prompt(JOIN, quail.col("r.body"),
                                       quail.col("a.aspect")))
                 .select("r.id", "a.id"))
        estimate = quail.speed_of_light_estimate(query, _answer)
        canvas = quail.speed_of_light_estimate(
            query, _answer,
            model=replace(DIFFUSION_GEMMA_26B_FP8, canvas_tokens=8))

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
    # three filter evaluations and four join pairs each carry the
    # eight canvas rows; the anchor frames carry none
    assert canvas.fresh_tokens == estimate.fresh_tokens + 8 * (3 + 4)


def test_planner_and_estimate_price_hybrid_queries_and_prefix_reuse(tmp_path):
    pq.write_table(pa.table({
        "id": ["r0", "r1"], "body": ["shared " * 1030 + s for s in ("a", "b")],
    }), tmp_path / "reviews.parquet")
    pq.write_table(pa.table({
        "id": ["q0", "q1"], "body": ["one two", "three four"],
    }), tmp_path / "questions.parquet")
    with quail.Session(config=quail.EngineConfig(
            model="diffusion-gemma-26b-a4b-fp8", device="h100-sxm"),
            tokenizer=str.split) as session:
        for name in ("reviews", "questions"):
            session.register(name, quail.DocumentProvider.from_parquet(
                str(tmp_path / f"{name}.parquet"), id_col="id"))
        query = (session.docs("reviews").alias("r")
                 .ai_filter(quail.prompt("First: {0}", quail.col("r.body")),
                            selectivity=1.0)
                 .ai_filter(quail.prompt("Second: {0}", quail.col("r.body")),
                            selectivity=1.0)
                 .ai_join(session.docs("questions").alias("q"),
                          quail.prompt("Compare {0} and {1}", quail.col("r.body"),
                                       quail.col("q.body")), selectivity=1.0)
                 .select("r.id", "q.id"))
        exact = quail.speed_of_light_estimate(
            query, lambda *_: True, credit_shared_prefixes=False)
        shared = quail.speed_of_light_estimate(query, lambda *_: True)
        planned = query.plan()
        pre = query.logical.operators().filters["r"][0].prompt.preamble_tokens
    assert exact.join_stages[0]["anchor"] == "r"
    assert exact.seconds == pytest.approx(planned.estimated_seconds)
    assert exact.work.sliding_pairs < exact.work.pairs
    assert exact.work.sliding_kv_read < exact.work.kv_read
    assert exact.work.pairs - shared.work.pairs == sum(range(1, pre + 1031))
    assert exact.work.sliding_pairs - shared.work.sliding_pairs == sum(
        min(position, 1024) for position in range(1, pre + 1031))
    assert shared.as_dict()["sliding_pairs"] == shared.work.sliding_pairs


CLASSIFY = "Judge the review's tone.\n\n{0}"
JOINED_CLASSIFY = "How does the review in DOCUMENT {0} treat DOCUMENT {1}?"


def _label(prompt, assignment):
    return "good" if assignment["r"] == 0 else "bad"


def _classification(query):
    ((call, _),) = query.logical.operators().labels.calls
    return call


def test_classifications_and_label_filters_price_like_filters(tmp_path):
    with _session(tmp_path) as sess:
        classified = (sess.docs("reviews").alias("r")
                      .ai_classify(quail.prompt(CLASSIFY, quail.col("r.body")),
                                   ["good", "bad"], name="tone"))
        query = classified.select("r.id", "tone")
        call = _classification(query)
        tail, preamble = call.prompt.tail_tokens, call.prompt.preamble_tokens
        distinct = quail.speed_of_light_estimate(query, _answer, label=_label)
        per_document = quail.speed_of_light_estimate(
            query, _answer, label=_label, credit_shared_prefixes=False)
        with pytest.raises(ValueError, match="label="):
            quail.speed_of_light_estimate(query, _answer)

        # an AI.IF filter first, then the classification over its
        # survivors, then the label filter, then the join over the
        # label's survivors
        query = (sess.docs("reviews").alias("r")
                 .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
                 .ai_classify(quail.prompt(CLASSIFY, quail.col("r.body")),
                              ["good", "bad"], name="tone")
                 .label_in("tone", ["good"])
                 .ai_join(sess.docs("aspects").alias("a"),
                          quail.prompt(JOIN, quail.col("r.body"),
                                       quail.col("a.aspect")))
                 .select("r.id", "a.id", "tone"))
        filters = query.logical.operators().filters
        question = filters["r"][0].prompt.tail_tokens
        estimate = quail.speed_of_light_estimate(
            query, _answer, label=_label, credit_shared_prefixes=False)

    # one question per document: the reference tail over each document
    assert per_document.fresh_tokens == 3 * (preamble + 2 + tail)
    assert distinct.fresh_tokens == per_document.fresh_tokens - preamble - 1
    assert [stage["operator"] for stage in distinct.filter_stages] == [
        "AI.CLASSIFY"]
    assert distinct.filter_stages[0]["name"] == "tone"
    assert distinct.filter_stages[0]["evaluated"] == 3
    assert distinct.classification_evaluations == 3
    assert distinct.filter_evaluations == 0
    assert distinct.post_filter_counts == {"r": 3}
    assert distinct.as_dict()["classification_evaluations"] == 3

    stages = estimate.filter_stages
    assert [stage["operator"] for stage in stages] == [
        "AI.IF", "AI.CLASSIFY", "IN"]
    assert [(stage["evaluated"], stage["passed"]) for stage in stages] == [
        (3, 2), (2, 2), (2, 1)]
    assert stages[2]["accepted"] == ["good"]
    assert stages[2]["fresh_tokens"] == 0
    # the filter scans every document; the classification asks its tail
    # of the two survivors over their resident prefixes
    assert stages[0]["fresh_tokens"] == 3 * (preamble + 2 + question)
    assert stages[1]["fresh_tokens"] == 2 * tail
    assert estimate.post_filter_counts == {"r": 1, "a": 2}
    # the join pairs only the label's survivor with every aspect
    assert estimate.join_pair_evaluations == 1 * 2
    assert estimate.join_stages[0]["classifications"] == []
    assert estimate.fresh_tokens == (
        stages[0]["fresh_tokens"] + stages[1]["fresh_tokens"]
        + estimate.join_stages[0]["fresh_tokens"])


def test_a_classification_of_joined_rows_is_priced_per_kept_pair(tmp_path):
    with _session(tmp_path) as sess:
        join = quail.prompt(JOIN, quail.col("r.body"), quail.col("a.aspect"))
        joined = (sess.docs("reviews").alias("r")
                  .ai_join(sess.docs("aspects").alias("a"), join))
        join_only = quail.speed_of_light_estimate(
            joined.select("r.id", "a.id"), _answer)
        query = (sess.docs("reviews").alias("r")
                 .ai_join(sess.docs("aspects").alias("a"), join)
                 .ai_classify(quail.prompt(JOINED_CLASSIFY, quail.col("r.body"),
                                           quail.col("a.aspect")),
                              ["praises", "pans"], name="stance")
                 .select("r.id", "a.id", "stance"))
        call = _classification(query)
        estimate = quail.speed_of_light_estimate(query, _answer, label=_label)

    (stage,) = estimate.join_stages
    anchor = stage["anchor"]
    partner = "a" if anchor == "r" else "r"
    assert join_only.join_stages[0]["anchor"] == anchor
    parts = {alias: (len(label), len(note))
             for alias, label, note in call.prompt.label_token_ids}
    # the one kept pair (r0, a0): the anchor note over the anchor's
    # resident prefix, then the partner label, the partner document,
    # and the classification tail
    partner_tokens = {"r": 2, "a": 1}[partner]
    pair_tokens = parts[partner][0] + partner_tokens + call.prompt.tail_tokens
    (classification,) = stage["classifications"]
    assert classification["name"] == "stance"
    assert classification["pairs"] == 1
    assert classification["fresh_tokens"] == parts[anchor][1] + pair_tokens
    assert estimate.classification_evaluations == 1
    assert estimate.fresh_tokens == (
        join_only.fresh_tokens + classification["fresh_tokens"])
    assert stage["fresh_tokens"] == (
        join_only.join_stages[0]["fresh_tokens"] + classification["fresh_tokens"])


def test_unpriced_operators_are_refused(tmp_path):
    def keep(tables):
        return tables["r"]

    with _session(tmp_path) as sess:
        filtered = (sess.docs("reviews").alias("r")
                    .ai_filter(quail.prompt(FILTER, quail.col("r.body"))))
        with pytest.raises(NotImplementedError, match="quail.apply"):
            quail.speed_of_light_estimate(
                filtered.apply(keep, quail.col("r.id")).select("r.id"),
                _answer)
        limited = (sess.docs("reviews").alias("r")
                   .ai_filter(quail.prompt(FILTER, quail.col("r.body")))
                   .limit(1).select("r.id"))
        with pytest.raises(NotImplementedError, match="LIMIT"):
            quail.speed_of_light_estimate(limited, _answer)
        scored = sess.sql(
            "SELECT r.id, AI.SCORE(PROMPT('Rate {0}', r.body)) AS score "
            "FROM reviews AS r")
        with pytest.raises(NotImplementedError, match="'score' column"):
            quail.speed_of_light_estimate(scored, _answer)
        compared = sess.sql(
            "SELECT r.id FROM reviews AS r "
            "WHERE AI.SCORE(PROMPT('Rate {0}', r.body)) > 0.5")
        with pytest.raises(NotImplementedError, match="Compare filter"):
            quail.speed_of_light_estimate(compared, _answer)
        exists = (sess.docs("reviews").alias("r")
                  .ai_join(sess.docs("aspects").alias("a"),
                           quail.prompt(JOIN, quail.col("r.body"),
                                        quail.col("a.aspect")),
                           semantics="exists")
                  .select("r.id"))
        with pytest.raises(NotImplementedError, match="'exists' joins"):
            quail.speed_of_light_estimate(exists, _answer)
