"""AI.CLASSIFY: label scoring, prompt text, planning, execution, and results."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import cpu_arena, fake_pack, fake_pipeline, fake_torch, letter_tokens
from test_score import _finish

import quail
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.executor.classify import (
    QuailClassifier,
    document_prefixes,
    label_requests,
)
from quail.backends.quail.executor.readout import AsyncLabelLogprobs, answer_rows
from quail.backends.quail.executor.state import LoadedModelState, QueryExecutionState
from quail.bench.quailb import run_output
from quail.bench.substrait import read_plan
from quail.catalog import DocumentProvider
from quail.execution.labels import (
    GreedyDecoder,
    best_label,
)
from quail.execution.pipelines import build_pipelines
from quail.execution.reranker import RerankerBatch
from quail.labels import label_trie
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    LogicalPlan,
    Scan,
    bind_classify_prompt,
)
from quail.logical.prompts import join_anchor_note
from quail.physical import (
    AiClassify,
    AiJoin,
    ClassifySpec,
    Filter,
    InList,
    RequestExecution,
    decode_graph,
    encode_graph,
)
from quail.planner.logical_rules import push_down_projection
from quail.planner.plan import EngineConfig, Refusal
from quail.specs import H100_SXM, QWEN3_4B_FP8
from quail_b.prompts import AGENT_PROGRESS, AGENT_PROGRESS_DESCRIPTIONS
from quail_b.prompts import AGENT_PROGRESS_LABELS as STAGES
from quail_b.queries import get_query

# one token per word: refund=1, request=2, status=3, shipping=4
LABELS = ("refund request", "refund status", "shipping")
IDS = ((1, 2), (1, 3), (4,))


def _bytes(text):
    return list(text.encode())


def test_label_trie_holds_each_proper_prefix_and_ties_go_first():
    trie = label_trie(IDS)
    assert trie == {(): [1, 4], (1,): [2, 3]}
    with pytest.raises(ValueError, match="no tokens"):
        label_trie([(1,), ()])
    assert best_label([-1.0, -0.5, -2.0]) == 1
    assert best_label([-1.0, -2.0, -1.0]) == 0


def test_greedy_decoder_follows_the_likeliest_child_to_a_label():
    targets = [1, 2, 3, 4]
    decoder = GreedyDecoder(IDS, targets, documents=2)
    assert decoder.rounds == 2
    # round 0 feeds the cue, request 0, for both documents
    assert [decoder.requests(doc) for doc in range(2)] == [[0], [0]]
    assert decoder.tokens == 2
    # document 0 follows token 1, document 1 takes the one-token label 4
    decoder.update(0, np.array([-1.0, -9.0, -9.0, -3.0]))
    decoder.update(1, np.array([-2.0, -9.0, -9.0, -0.5]))
    assert decoder.label.tolist() == [-1, 2]
    # round 1 feeds only token 1, request 1 plus its target column 0
    assert decoder.requests(1) is None and decoder.requests(0) == [1]
    assert decoder.tokens == 2 + 1
    # after token 1, token 3 beats token 2: "refund status"
    decoder.update(0, np.array([-9.0, -5.0, -1.0, -9.0]))
    assert decoder.label.tolist() == [1, 2]
    # a tie goes to the earlier label's token
    tied = GreedyDecoder(IDS, targets, documents=1)
    tied.requests(0)
    tied.update(0, np.array([-1.0, -9.0, -9.0, -1.0]))
    assert tied.node[0] == (1,)
    with pytest.raises(ValueError, match="prefix"):
        GreedyDecoder(((1, 2), (1,)), [1, 2], documents=1)


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
    # a many-row answer's rows land at (answer, row within it)
    answers, rows = answer_rows([2, 1, 3])
    assert answers.tolist() == [0, 0, 1, 2, 2, 2]
    assert rows.tolist() == [0, 1, 0, 0, 1, 2]


def test_classifier_reads_the_letters_at_the_cue_row(monkeypatch):
    from dataclasses import replace

    single = ClassifySpec(
        name="tone", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((2,), (4,)), scoring="letters")
    documents = {"d": [[10, 11], [12]]}
    prefixes = document_prefixes(single, documents, [0, 1])
    assert [list(prefix) for prefix in prefixes] == [[90, 10, 11], [90, 12]]
    requests = label_requests(single)
    assert requests.frame == [91, 92] and requests.suffixes == [[93]]
    assert requests.targets == [2, 4]
    with pytest.raises(ValueError, match="unknown"):
        label_requests(replace(single, scoring="trie_paths"))
    readout = SimpleNamespace(
        targets=np.asarray([2, 4]), rows=1,
        dtype=np.dtype((np.float32, (2,))),
        submit=lambda rows, rows_per_answer=None: rows,
        result=lambda rows: rows)

    def forward_single(chunk):
        return np.asarray([
            [-1.0, -5.0] if entry["key"][2] == 0 else [-5.0, -1.0]
            for entry in chunk.specs for _ in entry["suffixes"]],
            dtype=np.float32)

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward_single),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(),
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=object(),
    )
    batch = QuailClassifier(state).classify(single, [[0], [1]], documents)
    assert list(batch.scores) == ["a", "b"]
    assert batch.suffix_tokens == 2 * 1
    # every row packs its prefix, the two-token frame, and the cue
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2) + 2 * (2 + 1)



def test_classifier_decodes_one_token_per_round(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="trie_decode")
    requests = label_requests(spec)
    assert requests.rounds == 2 and requests.score is None
    assert requests.suffixes == [[93], [1], [2], [3], [4]]
    assert not requests.read_all_rows
    documents = {"d": [[10, 11], [12], [13]]}
    targets = [1, 2, 3, 4]
    # document 0 wants "refund request", 1 "shipping" outright, 2
    # "refund status": the wanted label's next token gets -1, another
    # label's first token -2 for token 1 and -3 for token 4, else -5
    wanted = {0: IDS[0], 1: IDS[2], 2: IDS[1]}

    def logprob(document, seen, token):
        if seen + (token,) == tuple(wanted[document][:len(seen) + 1]):
            return -1.0
        if not seen and token in (1, 4):
            return -2.0 if token == 1 else -3.0
        return -5.0

    packed = []
    path = {}

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            for suffix in entry["suffixes"]:
                packed.append((document, list(suffix), entry["f"],
                               entry.get("write_suffix_tokens", 0)))
                # the cue starts the path; a fed token extends it
                path[document] = (() if suffix[-1] == 93
                                  else path[document] + tuple(suffix))
                rows.append([logprob(document, path[document], token)
                             for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    readout = SimpleNamespace(
        targets=np.asarray(targets), rows=1,
        dtype=np.dtype((np.float32, (4,))),
        submit=lambda rows, rows_per_answer=None: rows,
        result=lambda rows: rows)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(),
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=object(),
    )
    arena = state.loaded_model.arena
    reserved = {}
    activate = arena.activate

    def recording_activate(key, tokens, capacity_tokens=None, **kw):
        reserved[key[2]] = capacity_tokens
        return activate(key, tokens, capacity_tokens=capacity_tokens, **kw)

    monkeypatch.setattr(arena, "activate", recording_activate)
    sent = []
    batch = QuailClassifier(state).classify(
        spec, [[0], [1], [2]], documents,
        on_answers=lambda rows, labels: sent.append((rows.tolist(), labels)))
    assert list(batch.scores) == ["refund request", "shipping", "refund status"]
    # each label streams once, as the chunk that decides it is read
    assert sent == [([1], ["shipping"]),
                    ([0, 2], ["refund request", "refund status"])]
    # each document reserves its prefix, the two-token frame, and one
    # kept token per round: the cue and the first label token
    assert reserved == {0: 3 + 2 + 2, 1: 2 + 2 + 2, 2: 2 + 2 + 2}
    # round 0 packs the frame and cue after each document's prefix (3
    # tokens for document 0, 2 for the others) and keeps all three in KV;
    # document 1 decodes its one-token label there. Round 1 feeds only
    # token 1, after the kept frame and cue, and keeps it
    assert sorted(packed) == sorted([
        (0, [91, 92, 93], 3, 3), (1, [91, 92, 93], 2, 3),
        (2, [91, 92, 93], 2, 3), (0, [1], 3 + 3, 1), (2, [1], 2 + 3, 1)])
    assert batch.suffix_tokens == 3 * 1 + 2 * 1


@pytest.fixture()
def session(tmp_path):
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({
        "id": [7, 9], "body": ["where is my refund", "love it"],
    }), path)
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
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
        self.calls = []

    def score(self, spec, rows, documents):
        assert isinstance(spec, ClassifySpec)
        self.calls.append((spec.name, rows.tolist()))
        values = np.asarray([
            spec.labels[0] if spec.name == "kind" else self.labels[row[0]]
            for row in rows], dtype=object)
        probabilities = np.asarray([
            [np.nan] * len(spec.labels) if value is None
            else [0.8 if label == value else 0.1 for label in spec.labels]
            for value in values]) if spec.probabilities else None
        return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0,
                             suffix_tokens=3 * len(rows),
                             probabilities=probabilities)


def test_classify_plans_filters_and_returns_labels(session):
    query = _topic(session)
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (test,) = [n for n in plan.nodes if isinstance(n, Filter)]
    assert classify.spec.labels == ("refund", "shipping", "praise")
    assert classify.spec.label_token_ids[0] == tuple(_bytes(" refund"))
    assert test.predicate == InList("topic", ("refund", "shipping"))
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph

    result = _finish(query, session, _Labels(["refund", "praise"]))
    rows = result.collect()
    assert rows.column("topic").to_pylist() == ["refund"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["refund", "praise"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True, False]
    assert result.report["node_metrics"]["ai-classify:0"]["suffix_tokens"] == 6

    # a projected label with no filter classifies every document
    plain = _topic(session, accepted=())
    result = _finish(plain, session, _Labels(["praise", "shipping"]))
    assert result.collect().column("topic").to_pylist() == [
        "praise", "shipping"]

    # a document whose decoded answer names no label leaves the run
    result = _finish(plain, session, _Labels(["praise", None]))
    assert result.collect().column("topic").to_pylist() == ["praise"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["praise"]
    assert result.report["node_metrics"]["ai-classify:0"]["output_rows"] == 1

    # with probabilities, a map of each label's probability beside the
    # label, from a rule that scores every label
    probable = session.sql(
        "SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
        "ARRAY['refund', 'shipping', 'praise'], {'probabilities': true}) "
        "AS topic FROM documents d")
    built = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic", probabilities=True)
        .select("d.id", "topic"))
    assert probable.logical == built.logical
    plan = probable.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.probabilities
    assert classify.spec.scoring != "trie_decode"
    assert plan.nodes[-1].columns == ("d.id", "topic", "topic_probabilities")
    rows = _finish(probable, session,
                   _Labels(["praise", "shipping"])).collect()
    assert rows.column("topic_probabilities").to_pylist() == [
        [("refund", 0.1), ("shipping", 0.1), ("praise", 0.8)],
        [("refund", 0.1), ("shipping", 0.8), ("praise", 0.1)]]


def _chain(session, *, filters=1, later_filters=()):
    query = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
    query = query.label_in("topic", ["refund", "shipping"], selectivity=0.5)
    if filters > 1:
        query = query.label_in("topic", ["refund"], selectivity=0.5)
    query = query.ai_classify(
        quail.prompt("What kind of {0}?", quail.col("d.body")),
        ["complaint", "question"], name="kind")
    for name, accepted in later_filters:
        query = query.label_in(name, accepted, selectivity=0.5)
    return query.select("d.id", "topic", "kind")


def test_sql_classifies_and_tests_labels(session):
    call = ("AI.CLASSIFY(PROMPT('What is {0} about?', d.body), "
            "ARRAY['refund', 'shipping', 'praise'])")
    query = session.sql(
        f"SELECT d.id, {call} AS topic FROM documents d "
        f"WHERE {call} IN ('refund', 'shipping')")
    built = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
        .label_in("topic", ["refund", "shipping"]).select("d.id", "topic"))
    assert query.logical == built.logical
    (test,) = [n for n in query.plan().nodes if isinstance(n, Filter)]
    assert test.predicate == InList("topic", ("refund", "shipping"))
    # a filter on the label alone, with a selectivity option
    filtered = session.sql(
        f"SELECT d.id FROM documents d WHERE {call[:-1]}, "
        f"{{'selectivity': 0.5}}) IN ('praise')")
    (predicate,) = filtered.logical.operators().label_filters["d"]
    assert predicate.selectivity == 0.5
    assert predicate.values == ("praise",)
    assert "d" not in filtered.logical.operators().filters
    # equality is membership in one label, on either side
    for form in (f"{call} = 'refund'", f"'refund' = {call}"):
        equal = session.sql(f"SELECT d.id FROM documents d WHERE {form}")
        (predicate,) = equal.logical.operators().label_filters["d"]
        assert predicate.values == ("refund",)
    for bad in (f"SELECT d.id, {call} FROM documents d",
                f"SELECT d.id FROM documents d WHERE {call} < 'refund'",
                f"SELECT d.id FROM documents d WHERE {call} IN (1)"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_frontends_place_classifications_on_their_table(session):
    # a classification sits above its table's AI.IF filters, below the
    # first filter on its label; a later classification sits above
    # that filter
    query = _chain(session, later_filters=(("kind", ("complaint",)),))
    query.plan()
    plan = query.logical
    kinds = [type(node).__name__ for node in plan.walk()]
    assert kinds == ["Scan", "SemanticClassify", "Filter",
                     "SemanticClassify", "Filter", "Project"]
    topic, kind = plan.operators().classifies
    assert (topic.name, topic.alias, kind.name) == ("topic", "d", "kind")
    assert plan.root.input.condition == quail.logical.InList(
        ColumnRef("d", "documents", "kind"), ("complaint",))
    assert plan.operators().labels.tests == {topic.call: [0], kind.call: [1]}
    # projection pushdown reaches the scan through the classifications
    assert push_down_projection(plan.root) is plan.root
    (scan,) = [node for node in plan.walk() if isinstance(node, Scan)]
    assert scan.columns == ("id", "body")
    mixed = (session.docs("documents").alias("d")
             .ai_filter(quail.prompt("Is {0} short?", quail.col("d.body")))
             .ai_classify(quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
                          name="x")
             .label_in("x", ["a"]).select("d.id"))
    assert [type(node).__name__ for node in mixed.logical.walk()] == [
        "Scan", "SemanticFilter", "SemanticClassify", "Filter", "Project"]
    operators = mixed.logical.operators()
    assert [type(p.expression).__name__
            for p in operators.filters["d"]] == ["ModelCall"]
    assert [test.position for test in operators.label_filters["d"]] == [1]
    # SQL names a classification only tested after its filter position
    hidden = session.sql(
        "SELECT d.id FROM documents d WHERE AI_FILTER(PROMPT('Is {0} short?', "
        "d.body)) AND AI.CLASSIFY(PROMPT('{0}', d.body), ARRAY['a', 'b']) "
        "IN ('a')")
    (node,) = hidden.logical.operators().classifies
    assert node.name == "__label_d_1"
    assert [type(node).__name__ for node in hidden.logical.walk()] == [
        "Scan", "SemanticFilter", "SemanticClassify", "Filter", "Project"]
    # a classification the query neither returns nor tests is not planned
    unused = (session.docs("documents").alias("d")
              .ai_filter(quail.prompt("Is {0} short?", quail.col("d.body")))
              .ai_classify(quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
                           name="x").select("d.id"))
    assert unused.logical.operators().classifies == ()
    text = _topic(session).explain()
    assert "SemanticClassify: topic" in text
    assert ("Filter: d.topic IN ['refund', 'shipping'] (selectivity=50%)"
            in text.split("SemanticClassify")[0])
    assert "Project: d.id, topic" in text


def _corpus(session, tmp_path, name, column, rows, words):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table({
        "id": [f"{name}{i}" for i in range(rows)],
        column: [" ".join(f"w{i}x{j}" for j in range(words)) for i in range(rows)],
    }), path)
    session.register(name, DocumentProvider.from_parquet(str(path), id_col="id"))


def test_rules_place_and_score_classifications(session, tmp_path):
    from quail.logical import classified_above_joins
    from quail.planner.logical_rules import (
        ClassifyPlacement,
        built_in_logical_rules,
        lift_classifications,
        push_down_projection,
    )
    from quail.planner.physical_rules import built_in_physical_rules
    from quail.planner.statistics import undecided

    assert [rule.name for rule in built_in_logical_rules()] == [
        "projection_pushdown", "filter_pushdown", "classify_placement",
        "filter_order", "join_order"]
    assert [rule.name for rule in built_in_physical_rules()] == [
        "kv_retention", "label_scoring", "prefix_sharing", "tree_attention"]
    plan = _topic(session).plan()
    # label_scoring chose every classification's scoring rule
    assert all(node.spec.scoring for node in plan.nodes
               if isinstance(node, AiClassify))
    assert plan.settings["classify_placement"] == "before joins"

    # sixty long reviews and three aspects
    _corpus(session, tmp_path, "reviews", "body", 60, 120)
    _corpus(session, tmp_path, "aspects", "aspect", 3, 4)
    labels = [f"label number {i}" for i in range(12)]
    r, a = quail.col("r.body"), quail.col("a.aspect")
    join = quail.prompt("Does {0} mention {1}?", r, a)

    def classified(filtered=False):
        query = session.docs("reviews").alias("r")
        if filtered:
            query = query.ai_filter(quail.prompt("Is {0} long?", r),
                                    selectivity=0.5)
        return query.ai_classify(quail.prompt("Topic of {0}?", r), labels,
                                 name="topic")

    # a join keeping one pair in a hundred: classifying the matched
    # documents after it beats classifying all sixty before it
    after = (classified().label_in("topic", labels[:11], selectivity=0.95)
             .ai_join(session.docs("aspects").alias("a"), join, selectivity=0.01)
             .select("r.id", "a.id", "topic"))
    written = after.logical.root
    plan = after.plan()
    assert plan.settings["classify_placement"] == "after joins"
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:a", "ai_join:r", "ai-classify:0", "filter:r:0",
        "recombine", "project"]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.inputs[0].source.node_id == "ai_join:r"
    assert classify.spec.scoring == "trie_tree"
    assert classify.spec.expected_inputs == pytest.approx(1.8)
    assert round(plan.estimated_seconds, 3) == 0.071
    assert plan.settings["search_seconds"] == plan.estimated_seconds
    # the query's logical plan is the one the rules left: the
    # classification sits above the join, which carries its stage
    lifted = lift_classifications(written)
    assert undecided(after.logical.root) == lift_classifications(
        push_down_projection(written))
    staged = after.logical.root.input.input.input
    assert (staged.exec_idx, staged.exec_anchor) == (0, "r")
    assert [type(node).__name__ for node in LogicalPlan(lifted).walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticClassify",
        "Filter", "Project"]
    assert classified_above_joins(lifted) == {"r"}
    assert classified_above_joins(written) == frozenset()
    assert lift_classifications(lifted) is None
    assert lift_classifications(_topic(session).logical.root) is None

    # the rule moves the classifications only when the lifted plan
    # prices lower, and never onto a refused plan
    def cost(above, below):
        return lambda plan, context: (
            above if classified_above_joins(plan.root) else below)

    assert ClassifyPlacement(cost(1.0, 2.0)).rewrite(written, None) == lifted
    assert ClassifyPlacement(cost(2.0, 1.0)).rewrite(written, None) is None
    assert ClassifyPlacement(cost(1.0, 1.0)).rewrite(written, None) is None
    assert ClassifyPlacement(cost(None, 2.0)).rewrite(written, None) is None

    # a filtered partner on a forced anchor stays classified before the
    # join, over the filter's survivors with their KV resident
    before = (classified(filtered=True)
              .label_in("topic", labels[:6], selectivity=0.5)
              .ai_join(session.docs("aspects").alias("a"), join,
                       selectivity=0.3, anchor="a")
              .select("r.id", "a.id", "topic"))
    plan = before.plan()
    assert plan.settings["classify_placement"] == "before joins"
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:a", "ai_filter:r", "ai-classify:0", "filter:r:1",
        "ai_join:a", "project"]
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert (classify.spec.scoring, classify.spec.expected_inputs) == (
        "trie_tree", 30.0)
    assert round(plan.estimated_seconds, 3) == 0.413


def test_sql_classifies_the_rows_a_join_keeps(session, tmp_path):
    path = tmp_path / "aspects.parquet"
    pq.write_table(pa.table({"id": [1, 2], "aspect": ["price", "size"]}),
                   path)
    session.register("aspects",
                     DocumentProvider.from_parquet(str(path), id_col="id"))
    stance = ("AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say about "
              "DOCUMENT {1}?', d.body, a.aspect), ARRAY['praise', 'complaint'])")
    query = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        f"JOIN aspects a ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', "
        f"d.body, a.aspect))")
    (call,) = [column.expression for column in query.logical.root.columns
               if isinstance(column, Alias)]
    assert call.kind == "label" and call.aliases() == ("d", "a")
    assert call.prompt.frame == join_anchor_note(0)
    assert call.prompt.tail.startswith("\n\nAnswer with exactly one")
    # the classification sits above the join of the two tables
    assert [type(node).__name__ for node in query.logical.walk()] == [
        "Scan", "Scan", "Join", "SemanticJoin", "SemanticClassify", "Project"]
    assert query.logical.root.input.call is call
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    (join,) = [n for n in plan.nodes if isinstance(n, AiJoin)]
    assert classify.spec.partner is not None
    assert classify.spec.anchor == join.anchor
    assert classify.inputs[0].source.node_id == join.node_id
    # More than one independent label consumer ends the join pipeline.
    other = stance.replace("What does DOCUMENT", "How does DOCUMENT")
    refused = session.sql(
        f"SELECT d.id, a.id, {stance} AS stance, {other} AS other "
        "FROM documents d JOIN aspects a "
        "ON AI_FILTER(PROMPT('Does {0} mention {1}?', d.body, a.aspect))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "joined_classify_pipeline"
    assert "standalone pair classification" in refused.reasons[0]
    distributed = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=2),
        tokenizer=letter_tokens)
    for name in ("documents", "aspects"):
        distributed.register(name, session.catalog.get(name))
    refused = distributed.sql(
        f"SELECT d.id, a.id, {stance} AS stance FROM documents d "
        "JOIN aspects a ON AI_FILTER(PROMPT('Does {0} mention {1}?', "
        "d.body, a.aspect))").plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "joined_classify_pipeline"
    distributed.close()
    # a joined row's label is not tested in WHERE
    with pytest.raises(CompileError, match="one table|filter on a label"):
        session.sql(
            f"SELECT d.id, a.id FROM documents d JOIN aspects a "
            f"ON AI_FILTER(PROMPT('Does {{0}} mention {{1}}?', d.body, "
            f"a.aspect)) "
            f"WHERE {stance} = 'praise'")


def test_planner_prices_the_rules_and_takes_the_cheapest():
    from quail.cost import budgets
    from quail.planner.classify import ClassifyRefusedError, _Table

    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    capacity = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    long_labels = tuple(tuple(range(100 + 5 * i, 104 + 5 * i))
                        for i in range(30))
    letters = tuple((200 + i,) for i in range(30))
    table = _Table(alias="d", mean=200.0, longest=300, budget=chunk,
                   chunk=chunk, backend_name="quail",
                   model=QWEN3_4B_FP8, device=H100_SXM, tokenizer=_bytes,
                   capacity=capacity, lengths=(200,) * 1000, tree=True)
    # thirty four-token labels sharing nothing, on resident documents
    # (no decode): the packed trie streams 91 tokens a document, the
    # letters one after a frame 30 tokens longer
    scoring, chosen = table.choose(1000, 20, 30, long_labels, True,
                                   lettered=(20, 60, letters))
    assert scoring == "letters" and chosen.suffix_tokens == 1000
    scoring, packed = table.choose(1000, 20, 30, long_labels, True)
    assert scoring == "trie_tree"
    assert packed.suffix_tokens == 1000 * len(label_trie(long_labels))
    assert chosen.seconds < packed.seconds
    # one-token labels: the trie reads the cue row too, after the
    # shorter prompt that names the labels
    ones = tuple((300 + i,) for i in range(3))
    scoring, _ = table.choose(1000, 20, 30, ones, True,
                              lettered=(20, 60, letters[:3]))
    assert scoring == "trie_tree"
    # fresh documents: a greedy decode feeds the cue and then one token
    # a round, kept in KV, four tokens over four rounds, and costs least
    scoring, decoded = table.choose(1000, 20, 30, long_labels, False,
                                    lettered=(20, 60, letters))
    assert scoring == "trie_decode"
    assert decoded.suffix_tokens == 1000 * 4
    fresh_trie = table.simulate("trie_tree", 1000, 20, 30, long_labels, False)
    assert decoded.rounds == 4 and decoded.seconds < fresh_trie.seconds
    # a decode cannot end at a label that is another label's prefix
    prefixed = long_labels[:-1] + (long_labels[0][:2],)
    assert table.choose(1000, 20, 30, prefixed, False,
                        lettered=(20, 60, letters))[0] == "letters"
    assert table.choose(1000, 20, 30, prefixed, False)[0] == "trie_tree"
    # without tree attention or a lettered prompt, the decode alone;
    # with neither and no decode, nothing can run
    unified = _Table(**{**table.__dict__, "tree": False})
    assert unified.choose(1000, 20, 30, long_labels, False)[0] == "trie_decode"
    with pytest.raises(ClassifyRefusedError, match="no label scoring rule"):
        unified.choose(1000, 20, 30, prefixed, True)


def test_chained_classifications_share_one_pipeline(session):
    query = _chain(session)
    plan = query.plan()
    classifies = [n for n in plan.nodes if isinstance(n, AiClassify)]
    topic, kind = classifies
    assert [node.spec.name for node in classifies] == ["topic", "kind"]
    assert build_pipelines(plan.graph)[topic.node_id].node_ids == (
        topic.node_id, "filter:d:0", kind.node_id)
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    assert kind.spec.estimated_seconds < topic.spec.estimated_seconds

    scorer = _Labels(["refund", "praise"])
    result = _finish(query, session, scorer)
    assert result.collect().to_pydict() == {
        "d.id": [7], "topic": ["refund"], "kind": ["complaint"]}
    labels = result.answer_tables["classifies"]
    assert labels["topic"].column("topic").to_pylist() == ["refund", "praise"]
    assert labels["kind"].select(["d", "kind"]).to_pydict() == {
        "d": [0], "kind": ["complaint"]}
    assert scorer.calls == [("topic", [[0], [1]]), ("kind", [[0]])]

    split = _chain(session, filters=2)
    assert build_pipelines(split.plan().graph)[topic.node_id].node_ids == (
        topic.node_id, "filter:d:0", "filter:d:1", kind.node_id)

    interleaved = _chain(session, later_filters=(
        ("kind", ("complaint",)), ("topic", ("refund",))))
    assert build_pipelines(interleaved.plan().graph)[topic.node_id].node_ids == (
        topic.node_id, "filter:d:0", kind.node_id, "filter:d:1", "filter:d:2")
    scorer = _Labels(["shipping", "praise"])
    result = _finish(interleaved, session, scorer)
    assert result.collect().num_rows == 0
    assert scorer.calls == [("topic", [[0], [1]]), ("kind", [[0]])]


def test_classify_refusals_and_builder_errors(session):
    query = _topic(session)
    # a label the classification cannot return matches no row; the
    # filter is accepted and plans as a test nothing passes
    none = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x").label_in("x", ["c"]).select("d.id"))
    (test,) = [n for n in none.plan().nodes if isinstance(n, Filter)]
    assert test.predicate == InList("x", ("c",))
    with pytest.raises(CompileError, match="differ ignoring case"):
        session.docs("documents").alias("d").ai_classify(
            quail.prompt("{0}", quail.col("d.body")), ["a", "A"], name="x")
    mixed = session.docs("documents").alias("d").ai_filter(
        quail.prompt("Is {0} short?", quail.col("d.body"))).ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x").select("d.id", "x")
    # beside AI.IF the classification is a step of the general plan
    assert [type(node).__name__ for node in mixed.plan().nodes] == [
        "Scan", "AiFilter", "AiClassify", "Project"]
    assert not isinstance(query.plan(), Refusal)

    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring in ("letters", "trie_tree", "trie_decode")
    # Qwen3 32B keeps its separate output head, so it classifies too
    big = quail.Session(EngineConfig(model="qwen3-32b-fp8", device="h100-sxm"),
                        tokenizer=_bytes)
    big.register("documents", session.catalog.get("documents"))
    assert not isinstance(_topic(big).plan(), Refusal)
    big.close()

    # stock vLLM classifies too, with one decode request per document
    vllm = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                      backend="stock_vllm"), tokenizer=_bytes)
    vllm.register("documents", session.catalog.get("documents"))
    request = next(node for node in _topic(vllm).plan().nodes
                   if isinstance(node, RequestExecution))
    assert [spec.output for spec in request.classifies] == ["topic"]
    # it decodes the label as text, so it has no label probabilities
    probable = vllm.docs("documents").alias("d").ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x", probabilities=True).select("d.id", "x")
    assert probable.plan().constraint == "classify_probabilities_need_quail_backend"
    vllm.close()


def test_bench_reads_classify_plans_and_reports_labels_by_operator():
    plan = read_plan(get_query("IMDB-14").plan)
    (sentiment, complaint) = plan.classifies
    (critical,) = plan.in_lists
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


def test_planner_reads_letters_on_a_diffusion_model(tmp_path):
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"id": [1, 2], "body": ["ab", "cd"]}), path)
    session = quail.Session(
        EngineConfig(model=DIFFUSION_GEMMA_26B_FP8.name, device="h100-sxm"),
        tokenizer=_bytes)
    session.register("documents", DocumentProvider.from_parquet(path, id_col="id"))
    # the reply opens after the turn suffix, so a letter is scored
    # without a leading space; one read beats 48 denoising steps
    plan = _topic(session).plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring == "letters"
    assert [list(ids) for ids in classify.spec.label_token_ids] == [
        _bytes(letter) for letter in "ABC"[:len(classify.spec.labels)]]
    assert classify.spec.estimated_seconds > 0
    chained = _chain(session).plan()
    classifies = [n for n in chained.nodes if isinstance(n, AiClassify)]
    assert len(classifies) == 2
    assert build_pipelines(chained.graph)[classifies[0].node_id].node_ids == (
        classifies[0].node_id, "filter:d:0", classifies[1].node_id)
    session.close()


def test_a_document_without_a_label_fails_every_filter_on_it(session):
    # document 7's answer named no label: it leaves the run before the
    # filter, as under the request backends
    result = _finish(_topic(session), session, _Labels([None, "refund"]))
    assert result.collect().column("topic").to_pylist() == ["refund"]
    (answers,) = result.answer_tables["filters"].values()
    assert answers.column("answer").to_pylist() == [True]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["refund"]
    assert labels.column("d").to_pylist() == [1]


def test_trie_chains_cover_every_node_once_and_gather_ancestors():
    from quail.execution.labels import tree_scores, trie_chains

    # labels a=[1,2,3], b=[1,2,4], c=[1,5,6], d=[7,8]: the rows are the
    # nodes with children, (), (1,), (1,2), (1,5), and (7,)
    labels = ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8))
    chains = trie_chains(labels)
    assert [nodes for nodes, _, _ in chains] == [
        [(), (1,), (1, 2)], [(1, 5)], [(7,)]]
    assert [start for _, start, _ in chains] == [0, 2, 1]
    # (1, 5) reads the cue row and the (1,) row; (7,) reads the cue row
    assert [gathers for _, _, gathers in chains] == [[], [(0, 2)], [(0, 1)]]
    nodes = [node for chain, _, _ in chains for node in chain]
    assert sorted(nodes) == sorted(label_trie(labels))
    # a label whose node is not on the first chain reads the right row
    targets = [1, 2, 3, 4, 5, 6, 7, 8]
    logprobs = np.full((len(nodes), len(targets)), -9.0)
    logprobs[0, 0] = -1.0     # () -> 1
    logprobs[0, 6] = -2.0     # () -> 7
    logprobs[1, 1] = -0.5     # (1,) -> 2
    logprobs[1, 4] = -0.25    # (1,) -> 5
    logprobs[2, 2] = -3.0     # (1, 2) -> 3
    logprobs[2, 3] = -0.1     # (1, 2) -> 4
    logprobs[3, 5] = -0.5     # (1, 5) -> 6
    logprobs[4, 7] = -0.3     # (7,) -> 8
    scores = tree_scores(labels, chains, targets, logprobs)
    assert scores.tolist() == [-4.5, -1.6, -1.75, -2.3]
    assert best_label(scores) == 1
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=1, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 93)), labels=("a", "b", "c", "d"),
        label_token_ids=labels, scoring="trie_tree")
    requests = label_requests(spec)
    # the cue, then each node's last token in chain order
    assert requests.suffixes == [[93, 1, 2, 5, 7]]
    assert requests.chains == chains


def test_pack_chunk_packs_chains_as_segments_reading_ancestors(monkeypatch):
    from fakes import cpu_staging
    from test_sliding_kv import plain_arena

    from quail.execution.labels import trie_chains

    torch = cpu_staging(monkeypatch)
    arena = plain_arena()
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=16, base_tokens=4)
    chains = trie_chains(((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8)))
    group = dict(key=key, prefix=None, f=4, suffixes=[[93, 1, 2, 5, 7]],
                 read_all_rows=True, chains=chains)
    chunk = chunk_mod.pack_chunk(torch, arena, [group], attention_mode="tree")
    # rows: cue, (1,), (1,2) | (1,5) | (7,): positions past the frame
    assert chunk.positions.tolist() == [4, 5, 6, 6, 5]
    assert chunk.meta["cu_a"].tolist() == [0, 3, 4, 5]
    reads = chunk.meta["reads"]
    assert reads["rows"].tolist() == [0, 1, 2, 3, 4]
    nodes = reads["nodes"]
    assert nodes["rows"].tolist() == [3, 4]
    assert nodes["key_rows"].tolist() == [0, 1, 0]
    assert nodes["cu_q"].tolist() == [0, 1, 2]
    assert nodes["cu_k"].tolist() == [0, 2, 3]
    assert nodes["b_index"].tolist() == [3, 4]
    assert chunk.final_indices.tolist() == [0, 1, 2, 3, 4]
    assert chunk.rows_per_answer == (5,)
    with pytest.raises(ValueError, match="tree attention"):
        chunk_mod.pack_chunk(torch, arena, [group], attention_mode="unified")


def test_merge_partial_equals_attention_over_the_union_of_keys():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.attention import merge_partial

    torch.manual_seed(0)
    heads, dim, keys = 2, 4, 6
    q = torch.randn(3, heads, dim)
    k = torch.randn(keys, heads, dim)
    v = torch.randn(keys, heads, dim)

    def attend(rows, key_slice):
        scores = torch.einsum("rhd,khd->hrk", q[rows], k[key_slice])
        weights = scores.softmax(-1)
        out = torch.einsum("hrk,khd->rhd", weights, v[key_slice])
        return out, scores.logsumexp(-1)

    out_b, lse_b = attend(slice(0, 3), slice(0, 4))
    out_c, lse_c = attend(slice(1, 3), slice(4, 6))
    whole, _ = attend(slice(1, 3), slice(0, 6))
    merge_partial(out_b, lse_b, torch.tensor([1, 2]), out_c, lse_c)
    assert torch.allclose(out_b[1:], whole, atol=1e-5)
    assert torch.allclose(out_b[0], attend(slice(0, 1), slice(0, 4))[0][0])


def test_classifier_scores_the_packed_trie(monkeypatch):
    from quail.execution.labels import trie_chains

    labels = ((1, 2, 3), (1, 2, 4), (1, 5, 6), (7, 8))
    chains = trie_chains(labels)
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 93)), labels=("a", "b", "c", "d"),
        label_token_ids=labels, scoring="trie_tree")
    targets = label_requests(spec).targets
    nodes = [node for chain, _, _ in chains for node in chain]

    # document 0 prefers label b, document 1 label d
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            wanted = labels[1 if document == 0 else 3]
            if entry.get("chains") is None:
                rows.append([0.0] * len(targets))     # the frame entry
                continue
            assert entry["chains"] == chains
            for node in nodes:
                rows.append([
                    -1.0 if node + (token,) == wanted[:len(node) + 1] else -5.0
                    for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    def submit(rows, rows_per_answer=None):
        # one record per answer: the frame entry's one row, a suffix's five
        out = np.full((len(rows_per_answer), len(nodes), len(targets)), np.nan)
        row = 0
        for index, count in enumerate(rows_per_answer):
            out[index, :count] = rows[row:row + count]
            row += count
        return out

    readout = SimpleNamespace(
        targets=np.asarray(targets), rows=len(nodes),
        dtype=np.dtype((np.float32, (len(nodes), len(targets)))),
        submit=submit, result=lambda rows: rows)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(),
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=object(),
    )
    batch = QuailClassifier(state).classify(
        spec, [[0], [1]], {"d": [[10, 11], [12]]})
    assert list(batch.scores) == ["b", "d"]
    # the trie's five rows once per document
    assert batch.suffix_tokens == 2 * 5


def test_sql_category_forms_options_and_label_tables(session):
    from quail.logical.prompts import CLASSIFY_INSTRUCTION

    def call_of(sql):
        query = session.sql(sql)
        (column,) = [c for c in query.logical.root.columns if isinstance(c, Alias)]
        return column.expression

    plain = call_of("SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', "
                    "d.body), ARRAY['refund', 'praise']) AS topic "
                    "FROM documents d")
    named = call_of("SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', "
                    "d.body), categories => ARRAY['refund', 'praise']) AS topic "
                    "FROM documents d")
    assert named == plain
    pairs = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY[('refund', 'money "
                    "back'), ('praise', NULL)]) AS topic FROM documents d")
    objects = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY[{'label': 'refund',"
                      " 'description': 'money back'}, {'label': 'praise'}]) "
                      "AS topic FROM documents d")
    assert pairs == objects
    assert pairs.labels == ("refund", "praise")
    assert pairs.descriptions == ("money back", "")
    assert "- refund: money back" in pairs.prompt.tail
    # a bare column asks the default instruction; a task description
    # follows it
    assert pairs.prompt.tail.startswith("{0}\n\n" + CLASSIFY_INSTRUCTION)
    tasked = call_of("SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                     "{'task_description': 'Pick the request kind.'}) AS topic "
                     "FROM documents d")
    assert "Pick the request kind." in tasked.prompt.tail
    # a registered label table, in ordinal order, then by label text
    session.register("kinds", DocumentProvider.from_table(pa.table({
        "label": ["praise", "refund"], "description": [None, "money back"],
        "ordinal": [2, 1]}), id_col="label"))
    table = call_of("SELECT d.id, AI.CLASSIFY(d.body, kinds) AS topic "
                    "FROM documents d")
    assert table.labels == ("refund", "praise")
    assert table.descriptions == ("money back", "")
    session.register("plain_kinds", DocumentProvider.from_table(pa.table({
        "label": ["refund", "praise"]}), id_col="label"))
    assert call_of("SELECT d.id, AI.CLASSIFY(d.body, plain_kinds) AS topic "
                   "FROM documents d").labels == ("praise", "refund")
    built = (session.docs("documents").alias("d")
             .ai_classify(quail.prompt("{0}", quail.col("d.body")), "kinds",
                          name="topic").select("d.id", "topic"))
    (column,) = [c for c in built.logical.root.columns if isinstance(c, Alias)]
    assert column.expression == table
    long_text = " ".join(["word"] * 26)
    for bad in (f"SELECT d.id, AI.CLASSIFY(d.body, ARRAY[('refund', '{long_text}'),"
                f" ('praise', NULL)]) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'examples': 'x'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'layout': 'labels_first'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'probabilities': 'yes'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY[1, 2]) AS topic "
                "FROM documents d"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_explain_names_the_classification_rule_and_the_filter(session):
    text = _topic(session).explain()
    assert "AiClassify: topic over d" in text
    # three short labels: the packed trie's 19 rows cost less than the
    # lettered prompt's longer frame
    assert "rule=trie_tree, labels=3" in text
    assert "Filter: topic IN ['refund', 'shipping']" in text


def test_choice_letters_are_one_token_each_and_capped():
    from quail.logical.prompts import MAX_LETTERS, choice_letters

    assert choice_letters(30) == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcd")
    assert choice_letters(60)[52:56] == ("AA", "AB", "AC", "AD")
    # every single letter maps to one shared id, and a two-letter
    # answer is three bytes
    same = lambda text: [1] if len(text) == 2 else _bytes(text)  # noqa: E731
    assert choice_letters(60, same) == ("A",)    # one token id for all
    assert len(choice_letters(60, letter_tokens)) == 60
    assert choice_letters(60, _bytes) == ()
    # without a prefix a single letter is one byte
    assert len(choice_letters(60, _bytes, prefix="")) == 52
    with pytest.raises(CompileError, match="at most"):
        choice_letters(MAX_LETTERS + 1)


def test_classify_prompt_letters_its_categories():
    from quail.logical.prompts import LETTERS_INSTRUCTION, answer_prefix

    ref = ColumnRef("t", "agent_traces", "trace")
    bound = bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                 AGENT_PROGRESS_DESCRIPTIONS)
    lettered = bound.lettered
    assert lettered.letters == tuple("ABCDE") and bound.letters == ()
    assert lettered.preamble == bound.preamble
    assert LETTERS_INSTRUCTION in lettered.tail
    for letter, label, description in zip(
            "ABCDE", STAGES, AGENT_PROGRESS_DESCRIPTIONS):
        assert f"\n- {letter}: {label} ({description})" in lettered.tail
        assert f"\n- {label}: {description}" in bound.tail
    assert lettered.lettered is None
    # a label follows the answer cue after a space; a reply that opens
    # a model turn starts without one
    assert bound.label_prefix == " " and answer_prefix(("", "")) == " "
    turned = bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                  turn=("<user>", "<model>"))
    assert turned.label_prefix == "" and turned.lettered.tail.endswith("<model>")
    # no lettered prompt when the tokenizer has no one-token letters
    assert bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                tokenizer=_bytes).lettered is None
    tokenized = bind_classify_prompt(AGENT_PROGRESS, (ref,), STAGES,
                                     tokenizer=letter_tokens)
    assert list(tokenized.lettered.tail_token_ids) == _bytes(
        tokenized.lettered.tail.replace("{0}", ""))


def test_classifier_reads_the_letter_at_the_first_canvas_row(monkeypatch):
    from quail.specs.base import CANVAS_SEED, AnswerCanvas

    settings = AnswerCanvas(rows=4, turn_close_id=6, pad_id=0)
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((1,), (2,)), scoring="letters")
    vocab = 8
    wanted = {0: 1, 1: 2}      # document index -> the letter its first row favors
    packed = {}

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            row = entry["key"][2]
            packed[row] = list(entry["canvas"])
            # the frame and the cue pack as one entry before the canvas
            assert [list(s) for s in entry["suffixes"]] == [[91, 92, 93]]
            assert entry["read_all_rows"]
            for position in range(settings.rows):
                rows.append([-1.0 if position == 0 and token == wanted[row]
                             else -5.0 for token in (1, 2)])
        return np.asarray(rows, dtype=np.float32)

    def submit(rows, rows_per_answer=None):
        padded = np.full((len(rows_per_answer), 4, 2), np.nan, np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            padded[answer, :count] = rows[start:start + count]
            start += count
        return padded

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    readout = SimpleNamespace(targets=np.asarray([1, 2]), rows=4,
                              dtype=np.dtype((np.float32, (4, 2))),
                              submit=submit, result=lambda rows: rows)
    state = QueryExecutionState(
        loaded_model=LoadedModelState(
            arena=cpu_arena(64),
            pipeline=fake_pipeline(forward_chunk=forward, canvas_ids=(7,),
                                   tree_attention=False),
            model_spec=SimpleNamespace(name="tiny", vocab=vocab,
                                       answer_canvas=settings),
            label_readout=readout,
            input_staging=SimpleNamespace(fixed_tokens=set()),
            model=object(),
        ),
        torch=fake_torch(),
        chunk_tokens=64,
        answer_rows=object(),
        async_answers=object(),
    )
    documents = {"d": [[9]] * 4 + [[10, 11], [12]]}
    batch = QuailClassifier(state).classify(spec, [[4], [5]], documents)
    assert list(batch.scores) == ["a", "b"]
    # each canvas: a random token from the document's row, the turn
    # close, then padding
    for index, row in enumerate((4, 5)):
        noise = np.random.default_rng((CANVAS_SEED, row, 0)).integers(0, vocab)
        assert packed[index] == [noise, 6, 0, 0]
    # the cue and the canvas after the prompt and the frame, once
    assert batch.suffix_tokens == 2 * (1 + 4)
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2) + 2 * (2 + 5)
