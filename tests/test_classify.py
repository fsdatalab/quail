"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch
from test_score import _finish

import quail
from quail.backends.quail.executor import loop
from quail.backends.quail.executor.classify import (
    QuailClassifier,
    classify_inputs,
)
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.bench.quailb import run_output
from quail.bench.substrait import read_plan
from quail.catalog import DocumentProvider
from quail.execution.labels import best_label, label_scores, label_trie
from quail.execution.reranker import RerankerBatch
from quail.logical import ColumnRef, CompileError, bind_classify_prompt, label_text
from quail.physical import (
    AiClassify,
    ClassifySpec,
    LabelFilter,
    decode_graph,
    encode_graph,
)
from quail.planner.plan import EngineConfig, Refusal
from quail_b import rendering
from quail_b.prompts import AGENT_OUTCOME, AGENT_OUTCOME_DESCRIPTIONS
from quail_b.prompts import AGENT_OUTCOME_LABELS as OUTCOMES
from quail_b.queries import get_query

# refund request, refund status, shipping, one token per word
LABELS = ("refund request", "refund status", "shipping")
IDS = ((1, 2), (1, 3), (4,))


def _bytes(text):
    return list(text.encode())


def test_label_scores_sum_each_labels_tokens_and_ties_go_first():
    trie = label_trie(IDS)
    assert trie == {(): [1, 4], (1,): [2, 3]}
    with pytest.raises(ValueError, match="no tokens"):
        label_trie([(1,), ()])
    prefixes, targets = [(), (1,)], [1, 2, 3, 4]
    unread = 1e-9    # tokens the trie never reads after that prefix
    logprobs = np.log([[0.6, unread, unread, 0.4],
                       [unread, 0.3, 0.7, unread]])
    scores = label_scores(IDS, prefixes, targets, logprobs)
    assert np.allclose(np.exp(scores), [0.18, 0.42, 0.4])
    assert best_label(scores) == 1
    assert best_label([-1.0, -2.0, -1.0]) == 0


def test_classify_prompt_is_quail_b_text_with_labels_after_it():
    ref = ColumnRef("t", "agent_traces", "trace")
    bound = bind_classify_prompt(AGENT_OUTCOME, (ref,), OUTCOMES,
                                 AGENT_OUTCOME_DESCRIPTIONS)
    document = "ran the tests; two failed"
    text = bound.preamble + document + bound.tail.replace("{0}", "")
    assert text == rendering.render_classify_prompt(
        AGENT_OUTCOME, document, OUTCOMES, AGENT_OUTCOME_DESCRIPTIONS)
    assert label_text("resolved") == rendering.LABEL_PREFIX + "resolved"
    tokenized = bind_classify_prompt(AGENT_OUTCOME, (ref,), OUTCOMES,
                                     AGENT_OUTCOME_DESCRIPTIONS, _bytes)
    assert list(tokenized.tail_token_ids) == _bytes(
        bound.tail.replace("{0}", ""))


def test_label_readout_is_log_softmax_over_the_whole_vocabulary():
    torch = pytest.importorskip("torch")
    head = torch.randn(300, 8, dtype=torch.bfloat16)
    hidden = torch.randn(70, 8, dtype=torch.bfloat16)
    readout = AsyncLabelLogprobs(torch, torch.nn.functional, head, [5, 17, 299])
    got = readout.logprobs(hidden)
    logits = torch.nn.functional.linear(hidden, head).float()
    expected = torch.log_softmax(logits, dim=1)[:, [5, 17, 299]]
    assert torch.allclose(got, expected, atol=1e-5)
    assert readout.dtype.shape == (3,)


def test_classifier_reads_every_trie_node_after_each_document(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS)
    documents = {"d": [[10, 11], [12], [13, 14, 15]]}
    prefixes, frame, suffixes, nodes, targets = classify_inputs(
        spec, documents, [0, 2])
    assert [list(prefix) for prefix in prefixes] == [
        [90, 10, 11], [90, 13, 14, 15]]
    assert frame == [91, 92]
    assert (nodes, suffixes) == ([(), (1,)], [[93], [93, 1]])
    assert targets == [1, 2, 3, 4]

    # document i prefers label i: its label tokens get log p = -1,
    # every other token -5; the frame entry's row is never read
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            for suffix in entry["suffixes"]:
                prefix = tuple(suffix[1:])
                wanted = IDS[document]
                rows.append([
                    -1.0 if prefix + (token,) == tuple(wanted[:len(prefix) + 1])
                    else -5.0 for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    readout = SimpleNamespace(
        targets=np.asarray(targets), dtype=np.dtype((np.float32, (4,))),
        submit=lambda rows: rows, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1], [2]], documents)
    assert list(batch.scores) == list(LABELS)
    # every row packs its prefix, the two-token frame, and two suffixes
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2 + 4) + 3 * (2 + 3)


@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": [7, 9], "body": ["where is my refund", "love it"],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"), tokenizer=_bytes)
    value.register("documents",
                   DocumentProvider.from_parquet(str(path), id_col="id"))
    yield value
    value.close()


def _topic(session, *, accepted=("refund", "shipping")):
    query = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
    if accepted:
        query = query.label_in("topic", accepted, selectivity=0.5)
    return query.select("d.id", "topic")


class _Labels:
    def __init__(self, labels):
        self.labels = labels

    def score(self, spec, rows, documents):
        assert isinstance(spec, ClassifySpec)
        values = np.asarray([self.labels[row[0]] for row in rows], dtype=object)
        return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0)


def test_classify_plans_filters_and_returns_labels(session):
    query = _topic(session)
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (test,) = [n for n in plan.nodes if isinstance(n, LabelFilter)]
    assert classify.spec.labels == ("refund", "shipping", "praise")
    assert classify.spec.label_token_ids[0] == tuple(_bytes(" refund"))
    assert (test.score_name, test.accepted) == ("topic", ("refund", "shipping"))
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph

    result = _finish(query, session, _Labels(["refund", "praise"]))
    rows = result.collect()
    assert rows.column("topic").to_pylist() == ["refund"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["refund", "praise"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True, False]

    # a projected label with no filter classifies every document
    plain = _topic(session, accepted=())
    result = _finish(plain, session, _Labels(["praise", "shipping"]))
    assert result.collect().column("topic").to_pylist() == [
        "praise", "shipping"]


def test_classify_refusals_and_builder_errors(session):
    query = _topic(session)
    with pytest.raises(CompileError, match="does not have"):
        session.docs("documents").alias("d").ai_classify(
            quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
            name="x").label_in("x", ["c"])
    with pytest.raises(CompileError, match="differ ignoring case"):
        session.docs("documents").alias("d").ai_classify(
            quail.prompt("{0}", quail.col("d.body")), ["a", "A"], name="x")
    mixed = session.docs("documents").alias("d").ai_filter(
        quail.prompt("Is {0} short?", quail.col("d.body"))).ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x").select("d.id", "x")
    assert "AI.IF" in mixed.plan().reasons[0]
    assert not isinstance(query.plan(), Refusal)

    big = quail.Session(EngineConfig(model="qwen3-32b-fp8", device="h100-sxm"),
                        tokenizer=_bytes)
    big.register("documents", session.catalog.get("documents"))
    refused = _topic(big).plan()
    assert isinstance(refused, Refusal) and "full output head" in refused.reasons[0]
    big.close()

    vllm = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                      backend="stock_vllm"), tokenizer=_bytes)
    vllm.register("documents", session.catalog.get("documents"))
    assert "not implemented on" in _topic(vllm).plan().reasons[0]
    vllm.close()


def test_bench_reads_classify_plans_and_reports_labels_by_operator():
    plan = read_plan(get_query("IMDB-14").plan)
    (sentiment, complaint) = plan.classifies
    (critical,) = plan.label_filters
    assert (sentiment.output, complaint.output) == ("sentiment", "complaint")
    assert critical.accepted == ("negative", "mixed")
    assert plan.select == ("r.id", "r.sentiment", "r.complaint")

    tables = {"reviews": pa.table({"id": ["a", "b", "c"]})}
    result = SimpleNamespace(
        answer_tables={
            "filters": {("r", 0): pa.table({
                "r": [0, 1, 2], "answer": [True, False, True]})},
            "joins": {},
            "classifies": {
                "sentiment": pa.table({
                    "r": pa.array([0, 1, 2], pa.int32()),
                    "sentiment": ["negative", "positive", "mixed"]}),
                "complaint": pa.table({
                    "r": pa.array([0, 2], pa.int32()),
                    "complaint": ["poor acting", "too long"]})},
        },
        report={"wall_s": 1.0},
        collect=lambda: pa.table({
            "d": ["a", "c"], "s": ["negative", "mixed"],
            "c": ["poor acting", "too long"]}))
    output = run_output(result, plan, tables)
    assert output.filter_answers == {}
    assert output.rows.column_names == ["r", "sentiment", "complaint"]
    assert output.classify_answers[complaint.id].to_pydict() == {
        "r": ["a", "c"], "label": ["poor acting", "too long"]}
    assert output.classify_answers[sentiment.id].column("r").to_pylist() == [
        "a", "b", "c"]
