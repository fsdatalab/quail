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


def _quoted(text):
    return "'" + text.replace("'", "''") + "'"


def _answer(prompt, assignment):
    if len(prompt.args) == 1:
        return assignment["r"] != 1
    return (assignment["r"], assignment["a"]) == (0, 0)


def test_distinct_prefix_estimates_for_filters_and_joins(tmp_path):
    with _session(tmp_path) as sess:
        query = sess.sql(
            f"SELECT r.id FROM reviews r "
            f"WHERE AI_FILTER(PROMPT({_quoted(FILTER)}, r.body))")
        filters = query.logical.operators().filters
        lowered = quail.lower_prompt(filters["r"][0].prompt, sess.tokenizer)
        question, preamble = lowered.tail_tokens, lowered.preamble_tokens
        distinct = quail.speed_of_light_estimate(query, _answer)
        per_document = quail.speed_of_light_estimate(
            query, _answer, credit_shared_prefixes=False)
        query = sess.sql(
            f"SELECT r.id, a.id FROM reviews r JOIN aspects a "
            f"ON AI_FILTER(PROMPT({_quoted(JOIN)}, r.body, a.aspect)) "
            f"WHERE AI_FILTER(PROMPT({_quoted(FILTER)}, r.body))")
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
        query = session.sql("""
            SELECT r.id, q.id FROM reviews r
            JOIN questions q
              ON AI_FILTER(PROMPT('Compare {0} and {1}', r.body, q.body),
                           {'selectivity': 1.0})
            WHERE AI_FILTER(PROMPT('First: {0}', r.body), {'selectivity': 1.0})
              AND AI_FILTER(PROMPT('Second: {0}', r.body), {'selectivity': 1.0})
        """)
        exact = quail.speed_of_light_estimate(
            query, lambda *_: True, credit_shared_prefixes=False)
        shared = quail.speed_of_light_estimate(query, lambda *_: True)
        planned = query.plan()
        pre = quail.lower_prompt(query.logical.operators().filters["r"][0].prompt,
                                 session.tokenizer).preamble_tokens
    assert exact.join_stages[0]["anchor"] == "r"
    # the plan credits the shared prefix as the estimate does, and prices
    # it by component as latency does; the two differ only in packing the
    # filters' and the join's passes
    assert shared.latency.seconds < exact.latency.seconds
    assert planned.estimated_seconds == pytest.approx(shared.latency.seconds,
                                                      rel=0.05)
    assert exact.work.sliding_pairs < exact.work.pairs
    assert exact.work.sliding_kv_read < exact.work.kv_read
    assert exact.work.pairs - shared.work.pairs == sum(range(1, pre + 1031))
    assert exact.work.sliding_pairs - shared.work.sliding_pairs == sum(
        min(position, 1024) for position in range(1, pre + 1031))
    assert shared.as_dict()["sliding_pairs"] == shared.work.sliding_pairs


CLASSIFY = "Judge the review's tone.\n\n{0}"


def _label(prompt, assignment):
    return "good" if assignment["r"] == 0 else "bad"


def _classification(query):
    ((call, _),) = query.logical.operators().labels.calls
    return call


def test_classifications_and_label_filters_price_like_filters(tmp_path):
    with _session(tmp_path) as sess:
        tone = (f"AI.CLASSIFY(PROMPT({_quoted(CLASSIFY)}, r.body), "
                "['good', 'bad']) AS tone")
        query = sess.sql(f"SELECT r.id, {tone} FROM reviews r")
        call = _classification(query)
        lowered = quail.lower_prompt(call.prompt, sess.tokenizer)
        tail, preamble = lowered.tail_tokens, lowered.preamble_tokens
        distinct = quail.speed_of_light_estimate(query, _answer, label=_label)
        per_document = quail.speed_of_light_estimate(
            query, _answer, label=_label, credit_shared_prefixes=False)
        with pytest.raises(ValueError, match="label="):
            quail.speed_of_light_estimate(query, _answer)

        # an AI.IF filter first, then the classification over its
        # survivors, then the label filter, then the join over the
        # label's survivors
        query = sess.sql(
            f"SELECT r.id, a.id, {tone} FROM reviews r JOIN aspects a "
            f"ON AI_FILTER(PROMPT({_quoted(JOIN)}, r.body, a.aspect)) "
            f"WHERE AI_FILTER(PROMPT({_quoted(FILTER)}, r.body)) "
            "AND tone IN ('good')")
        filters = query.logical.operators().filters
        question = quail.lower_prompt(filters["r"][0].prompt,
                                      sess.tokenizer).tail_tokens
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

    # one readout row per classified document
    assert distinct.filter_stages[0]["answer_rows"] == 3
    assert distinct.latency.component("readout").seconds > 0

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








def test_map_answers_decode_one_pass_per_token_and_share_their_document(tmp_path):
    # tokens each answer writes, by map and row
    written = {"plot": {0: 5, 1: 2, 2: 1}, "names": {0: 3, 1: 4, 2: 1}}
    with _session(tmp_path) as sess:
        query = sess.sql(
            "SELECT r.id, AI.MAP(r.body, 'Retell the plot.') AS plot, "
            "AI.MAP(r.body, 'List the names.') AS names FROM reviews r")
        maps = {node.call.prompt: node.name
                for node in query.logical.operators().maps}
        lowered = {name: quail.lower_prompt(prompt, sess.tokenizer)
                   for prompt, name in maps.items()}

        def estimate(lengths, shared):
            def answer(prompt, assignment):
                return " ".join(["word"] * lengths[maps[prompt]][assignment["r"]])
            return quail.speed_of_light_estimate(
                query, answer, credit_shared_prefixes=shared)

        one_token = {name: {row: 1 for row in range(3)} for name in written}
        estimates = {shared: (estimate(written, shared), estimate(one_token, shared))
                     for shared in (True, False)}
    # the longest answer writes 5 tokens: its first from the prompt, then
    # 4 fed back, one pass each
    assert estimates[True][0].latency.passes == 4
    assert sum(stage["answer_rows"]
               for stage in estimates[True][0].filter_stages) == 16
    # each fed token reads its document, its map's tail, and the answer
    # before it; the maps of one review read the review once per pass
    # with the shared prefix credit, once per answer without it
    head = lowered["plot"].preamble_tokens
    expected = {True: 0.0, False: 0.0}
    for row in range(3):
        document = head + 2
        fed = {name: written[name][row] - 1 for name in written}
        answer_reads = sum(count * lowered[name].tail_tokens + count * (count - 1) / 2
                  for name, count in fed.items())
        expected[True] += document * max(fed.values()) + answer_reads
        expected[False] += document * sum(fed.values()) + answer_reads
    for shared, (full, base) in estimates.items():
        assert full.work.kv_read - base.work.kv_read == expected[shared]


