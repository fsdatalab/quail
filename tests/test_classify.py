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
    document_prefixes,
    label_requests,
)
from quail.backends.quail.executor.readout import AsyncLabelLogprobs
from quail.bench.quailb import run_output
from quail.bench.substrait import read_plan
from quail.catalog import DocumentProvider
from quail.execution.labels import (
    GreedyDecoder,
    best_label,
    label_path_scores,
    label_trie,
    trie_paths,
)
from quail.execution.reranker import RerankerBatch
from quail.logical import (
    Alias,
    ColumnRef,
    CompileError,
    bind_classify_prompt,
    label_text,
)
from quail.physical import (
    AiClassify,
    ClassifySpec,
    ClassifyStage,
    LabelFilter,
    RequestExecution,
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


def test_label_trie_holds_each_proper_prefix_and_ties_go_first():
    trie = label_trie(IDS)
    assert trie == {(): [1, 4], (1,): [2, 3]}
    with pytest.raises(ValueError, match="no tokens"):
        label_trie([(1,), ()])
    assert best_label([-1.0, -0.5, -2.0]) == 1
    assert best_label([-1.0, -2.0, -1.0]) == 0


def test_trie_paths_cover_every_proper_prefix():
    targets = [1, 2, 3, 4]
    nan = float("nan")
    # a denoising step sends the cue and the whole canvas
    assert suffix_lengths("canvas", IDS, 16) == [17]
    # a decode sends the cue, then the cue and one token, for as many
    # rounds as the mean label length rounded up
    assert suffix_lengths("trie_decode", IDS) == [1, 2]


    # one chain over the deepest proper prefix (1,) reads both rows
    assert trie_paths(IDS) == [(1,)]
    assert trie_paths([(7,), (8,)]) == [()]
    # (1, 4)'s prefix (1,) is on the (1, 2) path, so it needs no chain
    assert trie_paths([(1, 2, 3), (1, 2), (1, 4), (5, 6)]) == [(5,), (1, 2)]
    assert suffix_lengths("trie_paths", IDS) == [2]
    with pytest.raises(ValueError, match="unknown"):
        suffix_lengths("next_rule", IDS)
    # its row 0 holds every first token, its row 1 every token after 1
    path_logprobs = np.log([[[0.6, nan, nan, 0.4], [nan, 0.3, 0.7, nan]]])
    scores = label_path_scores(IDS, [(1,)], targets, path_logprobs)
    assert np.allclose(np.exp(scores), [0.18, 0.42, 0.4])


def test_greedy_decoder_follows_the_likeliest_child_to_a_label():
    targets = [1, 2, 3, 4]
    decoder = GreedyDecoder(IDS, targets, documents=2)
    assert decoder.nodes == [(), (1,)] and decoder.rounds == 2
    # round 0 reads the row after the cue for both documents
    assert [decoder.requests(doc) for doc in range(2)] == [[0], [0]]
    assert decoder.tokens == 2
    # document 0 follows token 1, document 1 takes the one-token label 4
    decoder.update(0, np.array([-1.0, -9.0, -9.0, -3.0]))
    decoder.update(1, np.array([-2.0, -9.0, -9.0, -0.5]))
    assert decoder.label.tolist() == [-1, 2]
    assert decoder.requests(1) is None and decoder.requests(0) == [1]
    assert decoder.tokens == 2 + 2
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


def test_classifier_reads_every_row_of_each_trie_path(monkeypatch):
    # one two-row chain over the deepest proper prefix (1,) scores all
    # three labels
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="trie_paths")
    documents = {"d": [[10, 11], [12]]}
    prefixes = document_prefixes(spec, documents, [0, 1])
    assert [list(prefix) for prefix in prefixes] == [[90, 10, 11], [90, 12]]
    requests = label_requests(spec)
    assert requests.frame == [91, 92] and requests.read_all_rows
    assert requests.suffixes == [[93, 1]]
    targets = requests.targets
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
    readout_chains = SimpleNamespace(
        targets=np.asarray(targets), rows=2,
        dtype=np.dtype((np.float32, (2, 4))),
        submit=submit, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout_chains,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1]], documents)
    assert list(batch.scores) == list(LABELS[:2])
    assert batch.label_tokens == 2 * 2
    # every row packs its prefix, the two-token frame, and the chain
    assert batch.fresh_tokens + batch.cached_tokens == (3 + 2) + 2 * (2 + 2)

    # one-token labels read the cue's one row
    single = ClassifySpec(
        name="tone", aliases=("d",), query_template="", arguments=(),
        expected_inputs=2, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((2,), (4,)), scoring="trie_paths")
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
    assert batch.label_tokens == 2 * 1


def test_classifier_decodes_one_token_per_round(monkeypatch):
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="trie_decode")
    requests = label_requests(spec)
    assert requests.rounds == 2 and requests.score is None
    assert requests.suffixes == [[93], [93, 1]] and not requests.read_all_rows
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

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            for suffix in entry["suffixes"]:
                packed.append((document, list(suffix)))
                seen = tuple(suffix[1:])
                rows.append([logprob(document, seen, token) for token in targets])
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    readout = SimpleNamespace(
        targets=np.asarray(targets), rows=1,
        dtype=np.dtype((np.float32, (4,))),
        submit=lambda rows, rows_per_answer=None: rows,
        result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1], [2]], documents)
    assert list(batch.scores) == ["refund request", "shipping", "refund status"]
    # document 1 decoded its one-token label in round 0 and sent
    # nothing more; the others sent the cue and token 1 in round 1
    assert sorted(packed) == sorted([
        (0, [91, 92]), (1, [91, 92]), (2, [91, 92]),
        (0, [93]), (1, [93]), (2, [93]), (0, [93, 1]), (2, [93, 1])])
    assert batch.label_tokens == 3 * 1 + 2 * 2


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

    # every row of the one trie-path chain [93, 1] is read
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            wanted = IDS[entry["key"][2]]
            for suffix in entry["suffixes"]:
                for row in range(len(suffix)):
                    seen = tuple(suffix[1:row + 1])
                    rows.append([
                        -1.0 if seen + (token,) == tuple(wanted[:row + 1])
                        else -5.0 for token in [1, 2, 3, 4]])
        return np.asarray(rows, dtype=np.float32)

    def submit(rows, rows_per_answer=None):
        return rows.reshape(-1, 2, 4)

    monkeypatch.setattr(loop, "pack_chunk", recording_pack)
    readout = SimpleNamespace(
        targets=np.asarray([1, 2, 3, 4]), rows=2,
        dtype=np.dtype((np.float32, (2, 4))),
        submit=submit, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1]], documents)
    assert list(batch.scores) == list(LABELS[:2])
    assert packed == [(0, 0), (1, 32)]
    assert batch.borrowed_tokens == 32 and batch.cached_tokens == 32
    assert batch.fresh_tokens == (41 + 41 - 32) + 2 * (2 + 2)


def test_classifier_runs_chained_stages_on_resident_documents(monkeypatch):
    # stage 0 labels LABELS with the trie-path rule; the gate lets only
    # "refund request" through to stage 1, which labels ("a", "b")
    second = ClassifySpec(
        name="kind", aliases=("d",), query_template="", arguments=(),
        expected_inputs=1, estimated_seconds=0.0,
        prompt_token_parts=((90,), (94, 95)), labels=("a", "b"),
        label_token_ids=((5,), (6,)), scoring="trie_paths")
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=LABELS,
        label_token_ids=IDS, scoring="trie_paths",
        stages=(ClassifyStage(spec=second, accepted=(LABELS[0],)),))
    documents = {"d": [[10, 11], [12], [13, 14, 15]]}
    packed = []

    def recording_pack(torch, arena, specs, **kw):
        packed.extend((entry["key"][2], entry["prefix"] is not None,
                       [list(suffix) for suffix in entry["suffixes"]])
                      for entry in specs)
        return fake_pack(torch, arena, specs, **kw)

    # document i prefers label i at stage 0 (one chain over (1,), two
    # rows); at stage 1 every document prefers "b"
    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][2]
            for suffix in entry["suffixes"]:
                count = len(suffix) if entry.get("read_all_rows") else 1
                for row in range(count):
                    if suffix[0] == 93:
                        seen = tuple(suffix[1:row + 1])
                        wanted = IDS[document]
                        rows.append([
                            -1.0 if seen + (t,) == tuple(wanted[:row + 1])
                            else -5.0 for t in targets])
                    else:
                        rows.append([-1.0 if t == 6 else -5.0
                                     for t in targets])
        return np.asarray(rows, dtype=np.float32)

    targets = [1, 2, 3, 4, 5, 6]

    def submit(rows, rows_per_answer=None):
        rows_per_answer = rows_per_answer or [1] * len(rows)
        padded = np.full((len(rows_per_answer), 2, 6), np.nan, np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            padded[answer, :count] = rows[start:start + count]
            start += count
        return padded

    monkeypatch.setattr(loop, "pack_chunk", recording_pack)
    readout = SimpleNamespace(
        targets=np.asarray(targets), rows=2,
        dtype=np.dtype((np.float32, (2, 6))),
        submit=submit, result=lambda rows: rows)
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(spec, [[0], [1], [2]], documents)
    assert list(batch.scores) == list(LABELS)
    assert list(batch.later["kind"]) == ["b", None, None]
    # document 0 packs its prefix once: stage 1 streams frame and chain
    assert [(doc, fresh) for doc, fresh, _ in packed].count((0, True)) == 1
    assert ([suffixes for doc, _, suffixes in packed if doc == 0]
            == [[[91, 92]], [[93, 1]], [[94]], [[95]]])
    # stage 0: prefix, 2-token frame, one 2-token chain per document;
    # stage 1: document 0's 1-token frame and 1-token chain
    assert batch.fresh_tokens == (3 + 2 + 4) + 3 * (2 + 2) + (1 + 1)
    assert batch.label_tokens == 3 * 2 + 1


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
        later = {}
        for stage in spec.stages:
            # a later stage labels the rows the gate accepts with its
            # first label
            later[stage.spec.name] = np.asarray([
                stage.spec.labels[0]
                if stage.accepted is None or label in stage.accepted else None
                for label in values], dtype=object)
        return RerankerBatch(values, fresh_tokens=len(rows), cached_tokens=0,
                             label_tokens=3 * len(rows), later=later)


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

    # a document whose decoded answer names no label leaves the run
    result = _finish(plain, session, _Labels(["praise", None]))
    assert result.collect().column("topic").to_pylist() == ["praise"]
    labels = result.answer_tables["classifies"]["topic"]
    assert labels.column("topic").to_pylist() == ["praise"]
    assert result.report["node_metrics"]["ai-classify:0"]["output_rows"] == 1


def _chain(session, *, filters=1):
    query = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", "shipping", "praise"], name="topic")
    query = query.label_in("topic", ["refund", "shipping"], selectivity=0.5)
    if filters > 1:
        query = query.label_in("topic", ["refund"], selectivity=0.5)
    return query.ai_classify(
        quail.prompt("What kind of {0}?", quail.col("d.body")),
        ["complaint", "question"], name="kind").select("d.id", "topic", "kind")


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
    (test,) = [n for n in query.plan().nodes if isinstance(n, LabelFilter)]
    assert (test.score_name, test.accepted) == ("topic", ("refund", "shipping"))
    # a label filter alone, with a selectivity option
    filtered = session.sql(
        f"SELECT d.id FROM documents d WHERE {call[:-1]}, "
        f"{{'selectivity': 0.5}}) IN ('praise')")
    (predicate,) = filtered.logical.operators().filters["d"]
    assert predicate.selectivity == 0.5
    assert predicate.expression.accepted == ("praise",)
    # equality is membership in one label, on either side
    for form in (f"{call} = 'refund'", f"'refund' = {call}"):
        equal = session.sql(f"SELECT d.id FROM documents d WHERE {form}")
        (predicate,) = equal.logical.operators().filters["d"]
        assert predicate.expression.accepted == ("refund",)
    for bad in (f"SELECT d.id, {call} FROM documents d",
                f"SELECT d.id FROM documents d WHERE {call} < 'refund'",
                f"SELECT d.id FROM documents d WHERE {call} IN (1)"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_planner_prices_the_exhaustive_rules_by_their_label_tokens():
    from quail.cost import budgets
    from quail.planner.classify import _Table

    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    capacity = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    long_labels = tuple(tuple(range(100 + 5 * i, 104 + 5 * i))
                        for i in range(30))
    table = _Table(alias="d", mean=200.0, longest=300, budget=chunk,
                   chunk=chunk, scoring=None, backend_name="quail",
                   model=QWEN3_4B_FP8, device=H100_SXM, tokenizer=_bytes,
                   capacity=capacity, lengths=(200,) * 1000)
    # thirty four-token labels sharing nothing: a greedy decode sends
    # the cue and then one token a round, ten tokens over four rounds,
    # against 150 for the paths and 91 for the packed trie
    scoring, decoded = table.choose(1000, 20, 30, long_labels, False)
    assert scoring == "trie_decode"
    assert decoded.label_tokens == 1000 * (1 + 2 + 3 + 4)
    assert decoded.rounds == 4
    forced = _Table(**{**table.__dict__, "scoring": "trie_paths"})
    scoring, simulated = forced.choose(1000, 20, 30, long_labels, False)
    assert scoring == "trie_paths"
    assert simulated.label_tokens == 1000 * sum(
        1 + len(path) for path in trie_paths(long_labels))
    assert simulated.rounds == 1 and decoded.seconds < simulated.seconds
    # under tree attention the packed trie computes each node once
    # and costs less than the paths
    tree = _Table(**{**table.__dict__, "tree": True, "scoring": "trie_tree"})
    scoring, packed = tree.choose(1000, 20, 30, long_labels, False)
    assert scoring == "trie_tree"
    assert packed.label_tokens == 1000 * len(label_trie(long_labels))
    assert packed.seconds < simulated.seconds
    # a decode cannot end at a label that is another label's prefix,
    # and the documents of a chained stage are not decoded
    prefixed = long_labels[:-1] + (long_labels[0][:2],)
    assert table.choose(1000, 20, 30, prefixed, False)[0] == "trie_paths"
    assert table.choose(1000, 20, 30, long_labels, True)[0] == "trie_paths"


def test_simulation_sends_a_decode_one_chain_per_round():
    from quail.planner.classify import simulate

    # three documents of 100 tokens, a chunk of 250: two chunks for
    # the documents' first round, then the second round of each
    # document two chunks after the one that launched it
    prefixes = [100, 100, 100]
    chains = [[3, 2], [3, 2], [3, 2]]
    result = simulate(prefixes, 10, chains, chunk=250, capacity=10_000,
                      model=QWEN3_4B_FP8, device=H100_SXM, one_per_round=True)
    assert result.rounds == 2 and result.passes == 4
    assert result.label_tokens == 3 * 3 + 3 * 2
    assert result.work.tokens == 3 * 110 + 3 * 3 + 3 * 2
    assert result.seconds > 0
    # one round sends both chains at once, in fewer chunks
    once = simulate(prefixes, 10, chains, chunk=250, capacity=10_000,
                    model=QWEN3_4B_FP8, device=H100_SXM)
    assert once.rounds == 1 and once.passes == 2
    assert once.label_tokens == result.label_tokens
    # a full arena admits documents as earlier ones finish
    tight = simulate(prefixes, 10, chains, chunk=250, capacity=120,
                     model=QWEN3_4B_FP8, device=H100_SXM, one_per_round=True)
    assert tight.passes > result.passes
    assert tight.seconds > result.seconds


def test_simulation_credits_the_prefix_tokens_a_document_borrows():
    from quail.planner.classify import simulate

    prefixes = [100, 100, 100]
    chains = [[3], [3], [3]]
    scratch = simulate(prefixes, 10, chains, chunk=250, capacity=10_000,
                       model=QWEN3_4B_FP8, device=H100_SXM)
    # the second and third documents borrow 80 of their 100 tokens
    # from the first: they compute only the rest, the frame, and the
    # chain, and read the borrowed KV
    borrowed = simulate(prefixes, 10, chains, chunk=250, capacity=10_000,
                        model=QWEN3_4B_FP8, device=H100_SXM,
                        shared=[0, 80, 80])
    assert borrowed.work.tokens == scratch.work.tokens - 2 * 80
    assert borrowed.work.kv_read == scratch.work.kv_read + 2 * 80
    assert borrowed.work.pairs < scratch.work.pairs
    assert borrowed.seconds < scratch.seconds
    assert borrowed.label_tokens == scratch.label_tokens


def test_chained_classifications_share_one_node(session):
    query = _chain(session)
    plan = query.plan()
    classifies = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert len(classifies) == 1
    (node,) = classifies
    (stage,) = node.spec.stages
    assert stage.spec.name == "kind" and stage.accepted == ("refund", "shipping")
    assert node.explain_fields()["stages"] == [
        {"accepted": ["refund", "shipping"], "output": "kind"}]
    codecs = session.registry.codecs
    assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
    # the later stage streams only its frame and suffixes
    assert stage.spec.estimated_seconds < node.spec.estimated_seconds

    result = _finish(query, session, _Labels(["refund", "praise"]))
    assert result.collect().to_pydict() == {
        "d.id": [7], "topic": ["refund"], "kind": ["complaint"]}
    labels = result.answer_tables["classifies"]
    assert labels["topic"].column("topic").to_pylist() == ["refund", "praise"]
    assert labels["kind"].to_pydict() == {"d": [0], "kind": ["complaint"]}

    # a second filter between two classifications breaks the chain
    split = _chain(session, filters=2)
    assert sum(isinstance(n, AiClassify) for n in split.plan().nodes) == 2


def test_classify_refusals_and_builder_errors(session):
    query = _topic(session)
    # a label the classification cannot return matches no row; the
    # filter is accepted and plans as a test nothing passes
    none = (session.docs("documents").alias("d").ai_classify(
        quail.prompt("{0}", quail.col("d.body")), ["a", "b"],
        name="x").label_in("x", ["c"]).select("d.id"))
    (test,) = [n for n in none.plan().nodes if isinstance(n, LabelFilter)]
    assert test.accepted == ("c",)
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

    # the plan setting picks the scoring rule; the cost model by default
    plan = query.plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring in ("trie_paths", "trie_tree", "trie_decode")
    assert plan.settings["label_scoring"] == "cost model"
    for rule in ("trie_paths", "trie_tree", "trie_decode", "next_rule"):
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

    # Qwen3 32B keeps its separate output head, so it classifies too
    big = quail.Session(EngineConfig(model="qwen3-32b-fp8", device="h100-sxm"),
                        tokenizer=_bytes)
    big.register("documents", session.catalog.get("documents"))
    assert not isinstance(_topic(big).plan(), Refusal)
    big.close()

    # stock vLLM classifies too, with one request per label-trie node
    vllm = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                      backend="stock_vllm"), tokenizer=_bytes)
    vllm.register("documents", session.catalog.get("documents"))
    request = next(node for node in _topic(vllm).plan().nodes
                   if isinstance(node, RequestExecution))
    assert [spec.output for spec in request.classifies] == ["topic"]
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
    # the shared documents borrow their prefix, so the estimate falls
    (planned,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert 0 < classify.spec.estimated_seconds < planned.spec.estimated_seconds
    assert classify.spec.expected_inputs == planned.spec.expected_inputs
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


def test_classifier_decodes_each_document_with_denoising_steps(monkeypatch):
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.denoise import CANVAS_SEED
    from quail.specs import Denoising

    # an eight-token vocabulary; 0 stops the answer and 7 is a special
    # token
    spelling = {1: "a", 2: "b", 3: "\n", 4: "thought", 5: "x", 6: " "}
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: "".join(
        spelling.get(i, "" if skip_special_tokens else "<7>") for i in ids))
    settings = Denoising(
        canvas_rows=4, max_steps=5, t_min=0.4, t_max=0.8, entropy_bound=0.1,
        confidence_threshold=0.005, stability_threshold=1, logit_softcap=30.0,
        stop_token_ids=(0,))
    spec = ClassifySpec(
        name="topic", aliases=("d",), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((90,), (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((1,), (2,)), scoring="canvas")
    # document 0 answers "thought", a special token, then "a"; document 1
    # answers "b" only from its second step on, document 2 never
    # settles and answers "x"
    answers = {0: [[4, 3, 7, 1]] * 5,
               1: [[5, 5, 5, 5]] + [[6, 2, 0, 5]] * 4,
               2: [[5, 5, 0, 5], [5, 6, 0, 5]] * 3}
    vocab = 8
    head = torch.eye(vocab, dtype=torch.bfloat16)
    packed = {}
    seen = {}

    def forward(chunk):
        rows = []
        for index, entry in enumerate(chunk.specs):
            document = entry["key"][2]
            packed.setdefault(document, []).append(list(entry["canvas"]))
            # the first step writes the frame before the cue
            suffixes = [list(suffix) for suffix in entry["suffixes"]]
            first = len(packed[document]) == 1
            assert suffixes == ([[91, 92, 93]] if first else [[93]])
            assert entry["read_all_rows"]
            # the step's self-conditioning input is the soft embedding
            # the step before wrote, zero at the first step
            conditioning = chunk.meta["canvas"]["conditioning"][
                4 * index:4 * index + 4].float()
            seen.setdefault(document, []).append(conditioning)
            wanted = answers[document][len(packed[document]) - 1]
            for token in wanted:
                row = torch.full((vocab,), -1000.0)
                row[token] = 1000.0
                rows.append(row)
        return torch.stack(rows)

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    monkeypatch.setattr(torch.cuda, "Event", lambda **kw: SimpleNamespace(
        record=lambda: None, elapsed_time=lambda other: 2.0))
    pipeline = fake_pipeline(forward_chunk=forward, canvas_ids=(7,),
                             tree_attention=False,
                             normalizer=torch.tensor(2.0, dtype=torch.bfloat16))
    state = {"torch": torch, "arena": cpu_arena(64), "pipeline": pipeline,
             "chunk_tokens": 64, "model": SimpleNamespace(quail_full_head=head),
             "model_spec": SimpleNamespace(name="tiny", vocab=vocab,
                                           denoising=settings),
             "tokenizer": tokenizer,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    # rows 4, 5, and 6 of the table
    documents = {"d": [[9]] * 4 + [[10, 11], [12], [13]]}
    batch = QuailClassifier(state).classify(spec, [[4], [5], [6]], documents)
    assert list(batch.scores) == ["a", "b", None]
    # a canvas is done once its argmax tokens repeat with every row
    # certain, or after max_steps steps
    assert [len(packed[d]) for d in range(3)] == [2, 3, 5]
    # the first canvas is drawn from the document's row; every row of
    # a certain step is kept, so the next canvas is its argmax tokens
    first = np.random.default_rng((CANVAS_SEED, 5)).integers(0, vocab, 4)
    assert packed[1][0] == first.tolist()
    assert packed[1][1:] == [[5, 5, 5, 5], [6, 2, 0, 5]]
    assert packed[0][1] == [4, 3, 7, 1]
    # the first step reads zero conditioning; a later step reads the
    # step before's probabilities times the embedding and normalizer
    assert not seen[1][0].any()
    for step, tokens in enumerate(answers[1][:2]):
        expected = 2.0 * torch.eye(vocab)[tokens]
        assert torch.allclose(seen[1][step + 1], expected)
    # the cue and the canvas each step, after the prompt and the frame
    steps = 2 + 3 + 5
    assert batch.label_tokens == steps * (1 + 4)
    assert batch.fresh_tokens == (3 + 2 + 2) + 3 * 2 + steps * 5
    assert batch.cached_tokens == 0
    assert batch.later == {}


def test_pack_chunk_reads_every_canvas_row_when_asked():
    from fakes import cpu_staging

    torch = cpu_staging(pytest.MonkeyPatch())
    arena = cpu_arena(64)
    canvas = (90, 91, 92)
    groups = [dict(key=("d", 0), prefix=[1, 2], f=2, suffixes=[[10]],
                   read_all_rows=True),
              dict(key=("d", 1), prefix=[4], f=1, suffixes=[[12]],
                   read_all_rows=True)]
    chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas)
    assert chunk.input_ids.tolist() == [1, 2, 10, *canvas, 4, 12, *canvas]
    assert chunk.final_indices.tolist() == [3, 4, 5, 8, 9, 10]
    assert chunk.rows_per_answer == (3, 3)
    # a group's own canvas replaces the chunk's, every row read, and
    # its rows take their self-conditioning input from its rows of the
    # conditioning tensor
    conditioning = torch.arange(24, dtype=torch.float32).view(12, 2)
    groups = [dict(key=("d", 0), prefix=[1, 2], f=2, suffixes=[[10]],
                   read_all_rows=True, canvas=[80, 81], conditioning=4),
              dict(key=("d", 1), prefix=[4], f=1, suffixes=[[12]],
                   read_all_rows=True, canvas=[82, 83], conditioning=0)]
    chunk = loop.pack_chunk(torch, arena, groups, attention_mode="unified",
                            canvas=canvas, conditioning=conditioning)
    assert chunk.input_ids.tolist() == [1, 2, 10, 80, 81, 4, 12, 82, 83]
    assert chunk.final_indices.tolist() == [3, 4, 7, 8]
    assert chunk.rows_per_answer == (2, 2)
    assert chunk.meta["canvas"]["rows"].tolist() == [3, 4, 7, 8]
    assert chunk.meta["canvas"]["conditioning_rows"].tolist() == [4, 5, 0, 1]
    assert chunk.meta["canvas"]["conditioning"].tolist() == [
        [8, 9], [10, 11], [0, 1], [2, 3]]
    with pytest.raises(ValueError, match="self-conditioning"):
        loop.pack_chunk(torch, arena, [groups[0], dict(groups[1],
                                                       conditioning=None)],
                        attention_mode="unified", conditioning=conditioning)


def test_planner_decodes_labels_on_a_diffusion_model(tmp_path):
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"id": [1, 2], "body": ["ab", "cd"]}), path)
    session = quail.Session(
        EngineConfig(model=DIFFUSION_GEMMA_26B_FP8.name, device="h100-sxm"),
        tokenizer=_bytes)
    session.register("documents", DocumentProvider.from_parquet(path, id_col="id"))
    # the labels are their own tokens; the model decodes its answer
    plan = _topic(session).plan()
    (classify,) = [n for n in plan.nodes if isinstance(n, AiClassify)]
    assert classify.spec.scoring == "canvas"
    assert [list(ids) for ids in classify.spec.label_token_ids] == [
        _bytes(label_text(label)) for label in classify.spec.labels]
    assert classify.spec.estimated_seconds > 0
    assert plan.settings["label_scoring"] == "cost model"
    # a chain's classifications stay separate nodes
    chained = _chain(session).plan()
    classifies = [n for n in chained.nodes if isinstance(n, AiClassify)]
    assert len(classifies) == 2 and not any(n.spec.stages for n in classifies)
    # a label longer than the canvas does not fit it
    long_label = "x" * 20
    refused = session.docs("documents").alias("d").ai_classify(
        quail.prompt("What is {0} about?", quail.col("d.body")),
        ["refund", long_label], name="topic").select("d.id", "topic").plan()
    assert isinstance(refused, Refusal) and "16-row" in refused.reasons[0]
    forced = quail.Session(
        EngineConfig(model=DIFFUSION_GEMMA_26B_FP8.name, device="h100-sxm",
                     label_scoring="trie_paths"), tokenizer=_bytes)
    forced.register("documents", DocumentProvider.from_parquet(path, id_col="id"))
    refused = _topic(forced).plan()
    assert isinstance(refused, Refusal) and "canvas" in refused.reasons[0]
    forced.close()
    session.close()


def test_planner_prices_the_canvas_rule_as_rounds_of_denoising_steps():
    from quail.cost import budgets
    from quail.planner.classify import _Table, simulate
    from quail.specs import DIFFUSION_GEMMA_26B_FP8

    model = DIFFUSION_GEMMA_26B_FP8
    chunk = budgets.chunk_budget(model, H100_SXM)
    capacity = budgets.arena_tokens(model, H100_SXM, chunk)
    table = _Table(alias="d", mean=200.0, longest=300, budget=chunk,
                   chunk=chunk, scoring=None, backend_name="quail",
                   model=model, device=H100_SXM, tokenizer=_bytes,
                   capacity=capacity, lengths=(200,) * 100)
    scoring, simulated = table.choose(100, 20, 30, ((1, 2), (3,)), False)
    assert scoring == "canvas"
    # every document sends the cue and the 16-row canvas at each of the
    # 48 steps, and computes its prompt once
    steps = model.denoising.max_steps
    assert simulated.label_tokens == 100 * steps * 17
    assert simulated.work.tokens == 100 * (20 + 200 + 30) + 100 * steps * 17
    # every document fits the first chunk, and each later chunk runs
    # every document's next step
    assert simulated.passes == steps
    once = simulate([220] * 100, 30, [[17]] * 100, chunk, capacity, model,
                    H100_SXM, canvas_rows=16)
    assert once.passes == 1 and simulated.seconds > once.seconds

    # a round's answers are read while the next chunk runs: document 0's
    # second round waits a chunk that document 1's first round fills
    rounds = simulate([100, 100], 10, [[17], [17]], chunk=130,
                      capacity=10_000, model=model, device=H100_SXM,
                      canvas_rows=16, rounds=2)
    assert rounds.passes == 4 and rounds.label_tokens == 4 * 17
    assert rounds.work.tokens == 2 * 110 + 4 * 17
    # with nothing else to pack, the next round follows in the next chunk
    alone = simulate([100], 10, [[17]], chunk=130, capacity=10_000,
                     model=model, device=H100_SXM, canvas_rows=16, rounds=3)
    assert alone.passes == 3
    # the arena holds one document: the second starts after the first's
    # last round
    tight = simulate([100, 100], 10, [[17], [17]], chunk=1000, capacity=150,
                     model=model, device=H100_SXM, canvas_rows=16, rounds=2)
    assert tight.passes == 4


def test_a_document_without_a_label_fails_every_label_filter(session):
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
    assert requests.chains == chains and requests.read_all_rows


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
    chunk = loop.pack_chunk(torch, arena, [group], attention_mode="tree")
    # rows: cue, (1,), (1,2) | (1,5) | (6,): positions past the frame
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
        loop.pack_chunk(torch, arena, [group], attention_mode="unified")


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

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
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
    state = {"torch": fake_torch(), "arena": cpu_arena(64),
             "pipeline": fake_pipeline(forward_chunk=forward),
             "chunk_tokens": 64, "label_readout": readout,
             "input_staging": SimpleNamespace(fixed_tokens=set())}
    batch = QuailClassifier(state).classify(
        spec, [[0], [1]], {"d": [[10, 11], [12]]})
    assert list(batch.scores) == ["b", "d"]
    # the trie's five rows once per document
    assert batch.label_tokens == 2 * 5


def test_planner_prefers_the_packed_trie_when_labels_share_prefixes():
    from quail.cost import budgets
    from quail.planner.classify import _Table

    chunk = budgets.chunk_budget(QWEN3_4B_FP8, H100_SXM)
    capacity = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, chunk)
    table = _Table(alias="d", mean=200.0, longest=300, budget=chunk,
                   chunk=chunk, scoring=None, backend_name="quail",
                   model=QWEN3_4B_FP8, device=H100_SXM, tokenizer=_bytes,
                   capacity=capacity, lengths=(200,) * 500, tree=True)
    # twenty labels branching under one shared token: 22 trie rows
    # against 20 chains of three tokens; a greedy decode's six tokens
    # over three rounds cost least of all
    shared = tuple((7, i, 1) for i in range(20))
    scoring, decoded = table.choose(500, 20, 30, shared, False)
    assert scoring == "trie_decode" and decoded.rounds == 3
    # among the one-round rules the packed trie wins under tree
    # attention, the chains without it
    resident = _Table(**{**table.__dict__})
    scoring, simulated = resident.choose(500, 20, 30, shared, True)
    assert scoring == "trie_tree"
    assert simulated.label_tokens == 500 * len(label_trie(shared))
    unified = _Table(**{**table.__dict__, "tree": False})
    assert unified.choose(500, 20, 30, shared, True)[0] == "trie_paths"
    # one-token labels: every rule reads the cue row once; the tie
    # keeps the chains
    scoring, _ = table.choose(500, 20, 30, ((1,), (2,)), False)
    assert scoring == "trie_paths"


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
                     "{'task_description': 'Pick the request kind.', "
                     "'output_mode': 'single'}) AS topic FROM documents d")
    assert "Pick the request kind." in tasked.prompt.tail
    # labels first: the question and categories go before the document
    first = call_of("SELECT d.id, AI.CLASSIFY(PROMPT('What is {0} about?', "
                    "d.body), ARRAY['refund', 'praise'], {'layout': "
                    "'labels_first'}) AS topic FROM documents d")
    assert first != plain
    assert first.prompt.preamble.startswith(CLASSIFY_INSTRUCTION)
    assert "- refund" in first.prompt.preamble
    assert first.prompt.tail == "{0}\nANSWER:"
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
                "{'output_mode': 'multi'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'examples': 'x'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'praise'], "
                "{'layout': 'sideways'}) AS topic FROM documents d",
                "SELECT d.id, AI.CLASSIFY(d.body, ARRAY[1, 2]) AS topic "
                "FROM documents d"):
        with pytest.raises(CompileError):
            session.sql(bad)


def test_explain_names_the_classification_rule_and_label_filter(session):
    text = _topic(session).explain()
    assert "AiClassify: topic over d" in text
    # the byte tokenizer gives every label the same leading space, so
    # the packed trie wins
    assert "rule=trie_tree, labels=3" in text
    assert "LabelFilter: topic in ['refund', 'shipping']" in text


def test_label_readout_projects_only_the_targets_when_unnormalized():
    torch = pytest.importorskip("torch")
    head = torch.randn(300, 8, dtype=torch.bfloat16)
    hidden = torch.randn(7, 8, dtype=torch.bfloat16)
    readout = AsyncLabelLogprobs(torch, torch.nn.functional, head, [5, 17, 299],
                                 normalize=False)
    got = readout.logprobs(hidden)
    expected = torch.nn.functional.linear(hidden, head).float()[:, [5, 17, 299]]
    assert torch.allclose(got, expected, atol=1e-5)
    # the same ranking as the normalized readout, row by row
    normalized = AsyncLabelLogprobs(torch, torch.nn.functional, head,
                                    [5, 17, 299]).logprobs(hidden)
    assert torch.equal(got.argmax(1), normalized.argmax(1))
