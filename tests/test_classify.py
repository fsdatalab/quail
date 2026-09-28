"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch
from test_planner import _token_store
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
from quail.execution.labels import (
    best_label,
    label_chain_scores,
    label_scores,
    label_trie,
)
from quail.execution.reranker import RerankerBatch
from quail.logical import ColumnRef, CompileError, bind_classify_prompt, label_text
from quail.physical import (
    AiClassify,
    ClassifySpec,
    LabelFilter,
    decode_graph,
    encode_graph,
)
from quail.planner.classify import suffix_lengths
from quail.planner.physical_optimizer import PlanningContext
from quail.planner.physical_rules import PrefixSharing
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import H100_SXM, QWEN3_4B_FP8
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


def test_label_chain_scores_read_one_row_per_label_token():
    targets = [1, 2, 3, 4]
    nan = float("nan")
    # chain i's row r scores token r of label i; rows past a label are NaN
    logprobs = np.log([
        [[0.6, nan, nan, nan], [nan, 0.3, nan, nan]],
        [[0.6, nan, nan, nan], [nan, nan, 0.7, nan]],
        [[nan, nan, nan, 0.4], [nan, nan, nan, nan]],
    ])
    scores = label_chain_scores(IDS, targets, logprobs)
    assert np.allclose(np.exp(scores), [0.18, 0.42, 0.4])
    assert suffix_lengths("label_chains", IDS) == [2, 2, 1]
    assert suffix_lengths("trie_nodes", IDS) == [1, 2]


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


def test_multi_row_label_readout_pads_each_suffix_to_its_rows():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("pinned host copies need a GPU")
    head = torch.randn(300, 8, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(5, 8, dtype=torch.bfloat16, device="cuda")
    readout = AsyncLabelLogprobs(torch, torch.nn.functional, head, [5, 17],
                                 rows=3)
    assert readout.dtype.shape == (3, 2)
    got = readout.result(readout.submit(hidden, rows_per_answer=[2, 3]))
    flat = readout.logprobs(hidden).cpu().numpy()
    assert got.shape == (2, 3, 2)
    assert np.allclose(got[0, :2], flat[:2]) and np.isnan(got[0, 2]).all()
    assert np.allclose(got[1], flat[2:])
    with pytest.raises(ValueError, match="rows to read"):
        readout.submit(hidden, rows_per_answer=[4, 1])


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
        targets=np.asarray(targets), rows=1,
        dtype=np.dtype((np.float32, (4,))),
        submit=lambda rows: rows, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1], [2]], documents)
    assert list(batch.scores) == list(LABELS)
    # every row packs its prefix, the two-token frame, and two suffixes
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2 + 4) + 3 * (2 + 3)


def test_classifier_reads_every_row_of_one_chain_per_label(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="label_chains")
    documents = {"d": [[10, 11], [12]]}
    prefixes, frame, suffixes, nodes, targets = classify_inputs(
        spec, documents, [0, 1])
    assert frame == [91, 92]
    # the cue's last token, then every label token but its last
    assert (nodes, suffixes) == (None, [[93, 1], [93, 1], [93]])
    assert targets == [1, 2, 3, 4]

    # document i prefers label i: after the cue and the first r tokens
    # of a label, that label's token r gets log p = -1, others -5
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            wanted = IDS[document]
            for suffix in entry["suffixes"]:
                count = len(suffix) if entry.get("read_all_rows") else 1
                for row in range(count):
                    seen = tuple(suffix[1:row + 1])
                    rows.append([
                        -1.0 if seen + (token,) == tuple(wanted[:row + 1])
                        else -5.0 for token in targets])
        return np.asarray(rows, dtype=np.float32)

    def submit(rows, rows_per_answer=None):
        rows_per_answer = rows_per_answer or [1] * len(rows)
        padded = np.full((len(rows_per_answer), 2, 4), np.nan, np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            padded[answer, :count] = rows[start:start + count]
            start += count
        return padded

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    readout = SimpleNamespace(
        targets=np.asarray(targets), rows=2,
        dtype=np.dtype((np.float32, (2, 4))),
        submit=submit, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1]], documents)
    assert list(batch.scores) == list(LABELS[:2])
    assert batch.label_tokens == 2 * (2 + 2 + 1)
    # every row packs its prefix, the two-token frame, and three chains
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2) + 2 * (2 + 5)

    # one-token labels read one row per chain
    single = ClassifySpec(
        name="tone", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((2,), (4,)), scoring="label_chains")
    readout = SimpleNamespace(
        targets=np.asarray([2, 4]), rows=1,
        dtype=np.dtype((np.float32, (2,))),
        submit=lambda rows, rows_per_answer=None: rows,
        result=lambda rows: rows)
    state["label_readout"] = readout

    def forward_single(chunk):
        return np.asarray([
            [-1.0, -5.0] if entry["key"][2] == 0 else [-5.0, -1.0]
            for entry in chunk.specs for _ in entry["suffixes"]],
            dtype=np.float32)

    state["pipeline"] = fake_pipeline(forward_chunk=forward_single)
    batch = QuailClassifier(state).classify(single, [[0], [1]], documents)
    assert list(batch.scores) == ["a", "b"]


def test_classifier_borrows_shared_prefix_pages(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, share_prefixes=True)
    # after the head, document 1 shares two 16-token pages with document 0
    documents = {"d": [[10] * 40, [10] * 32 + [11] * 8]}
    packed = []

    def recording_pack(torch, arena, specs, **kw):
        packed.extend((spec["key"][2], spec.get("start", 0))
                      for spec in specs if spec["prefix"] is not None)
        return fake_pack(torch, arena, specs, **kw)

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            wanted = IDS[entry["key"][2]]
            for suffix in entry["suffixes"]:
                prefix = tuple(suffix[1:])
                rows.append([
                    -1.0 if prefix + (token,) == tuple(wanted[:len(prefix) + 1])
                    else -5.0 for token in [1, 2, 3, 4]])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(loop, "pack_chunk", recording_pack)
    readout = SimpleNamespace(
        targets=np.asarray([1, 2, 3, 4]), rows=1,
        dtype=np.dtype((np.float32, (4,))),
        submit=lambda rows: rows, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1]], documents)
    assert list(batch.scores) == list(LABELS[:2])
    assert packed == [(0, 0), (1, 32)]
    assert batch.borrowed_tokens == 32 and batch.cached_tokens == 32
    assert batch.fresh_tokens == (41 + 41 - 32) + 2 * (2 + 3)


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
        return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0,
                             label_tokens=3 * len(rows))


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
    assert result.report["node_metrics"]["ai-classify:0"]["label_tokens"] == 6

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

    # the plan setting picks the scoring rule; label chains by default
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring == "label_chains"
    assert plan.settings["label_scoring"] == "label_chains"
    for rule in ("trie_nodes", "label_chains", "next_rule"):
        ruled = quail.Session(EngineConfig(
            model="qwen3-4b-fp8", device="h100-sxm", label_scoring=rule),
            tokenizer=_bytes)
        ruled.register("documents", session.catalog.get("documents"))
        plan = _topic(ruled).plan()
        if rule == "next_rule":
            assert isinstance(plan, Refusal) and "unknown" in plan.reasons[0]
        else:
            (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
            assert classify.spec.scoring == rule
            assert plan.settings["label_scoring"] == rule
        ruled.close()

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


def test_prefix_sharing_rule_fires_for_a_classification(session, tmp_path):
    plan = _topic(session).plan()
    shared = " ".join(str(i) for i in range(40))
    store = _token_store(tmp_path / "shared.arrow",
                         [shared, shared + " 99 98", "7 7 7"])
    plain = _token_store(tmp_path / "plain.arrow", ["1 2 3", "4 5 6"])

    def context(lengths):
        return PlanningContext(
            model=QWEN3_4B_FP8, device=H100_SXM, gpu_count=1,
            document_tokens={"d": lengths}, backend="quail")

    graph = PrefixSharing().rewrite(plan.graph, context(store.lengths))
    (classify,) = [n for n in graph.nodes if isinstance(n, AiClassify)]
    assert classify.spec.share_prefixes
    assert classify.explain_fields()["share_prefixes"]
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(graph, codecs), codecs) == graph
    assert PrefixSharing().rewrite(plan.graph, context(plain.lengths)) is None


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
