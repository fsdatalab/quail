"""End-to-end Session tests with a fake executor."""

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers.fakes import register_claims_evidence
from helpers.sessions import CONFIG, TRUTH, _run, fake_tok, make_executor

import quail
from demos import quickstart, quickstart_modal
from quail.api import execute as execution
from quail.catalog.pairs import (
    allowed_members,
    estimate_pair_fraction,
    members_by_partner,
    pair_table,
    partner_map,
)
from quail.execution.result import (
    QueryResult,
    build_result_declaration,
    count_rows,
    document_index_table,
)
from quail.physical import AiJoin
from quail.planner.plan import EngineConfig


def test_session_upgrades_and_reuses_offsets_without_invalidating_old_tokens():
    calls = []

    def encode(text):
        calls.append(text)
        return [ord(char) for char in text]

    encode.token_bytes = lambda: {ord(char): char.encode() for char in "café東京"}
    with quail.Session(CONFIG, tokenizer=encode) as session:
        session.register("docs", quail.DocumentProvider.from_table(
            pa.table({"id": [1, 2], "text": ["café", "東京"]}), id_col="id"))
        previous = session.inputs.tokenize("docs", "text").tokens
        assert not previous.has_offsets
        calls.clear()
        upgraded = session.inputs.tokenize("docs", "text", with_offsets=True).tokens
        repeated = session.inputs.tokenize("docs", "text", with_offsets=True).tokens
        assert upgraded is repeated and upgraded.has_offsets
        assert not calls
        assert list(previous[0]) == list(upgraded[0]) == encode("café")
        assert upgraded[1].character_offsets.tolist() == [[0, 1], [1, 2]]


def test_offset_tokenization_uses_gigatoken_without_transformers(monkeypatch):
    from gigatoken import Tokenizer as Gigatoken
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    texts = ["café", "東京"]
    native = Tokenizer(models.BPE())
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    expected = native.encode_batch(texts)
    fast = Gigatoken(native)
    monkeypatch.setitem(sys.modules, "gigatoken",
                        SimpleNamespace(Tokenizer=lambda source: fast))
    monkeypatch.setitem(sys.modules, "transformers", None)
    with quail.Session(CONFIG) as session:
        session.register("docs", quail.DocumentProvider.from_table(
            pa.table({"id": [1, 2], "text": texts}), id_col="id"))
        scan = session.inputs.tokenize_async("docs", "text", with_offsets=True).result()
        for index, encoding in enumerate(expected):
            assert list(scan.tokens[index]) == encoding.ids
            assert scan.tokens[index].character_offsets.tolist() == [
                list(pair) for pair in encoding.offsets]



@pytest.fixture()
def sess(tmp_path):
    s = quail.Session(CONFIG, tokenizer=fake_tok)
    pq.write_table(pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
    }), tmp_path / "r.parquet")
    s.register("reviews", quail.DocumentProvider.from_parquet(
        str(tmp_path / "r.parquet"), id_col="id"))
    yield s
    s.close()


def test_session_requires_model_and_device_and_loads_gigatoken(monkeypatch):
    with pytest.raises(TypeError, match="'model' and 'device'"):
        EngineConfig()
    with pytest.raises(TypeError, match="'device'"):
        EngineConfig(model="qwen3-4b-fp8")
    with pytest.raises(TypeError, match="required positional argument: 'config'"):
        quail.Session()
    assert (CONFIG.gpus, CONFIG.backend) == (1, "quail")

    sources = []
    resolutions = []

    def resolve(model_name, revision, *, include_weights):
        resolutions.append((model_name, revision, include_weights))
        return "/snapshots/pinned-model"

    monkeypatch.setattr("quail.api.session.resolve_model_path", resolve)

    class Tokenizer:
        def __init__(self, source):
            sources.append(source)

        def encode(self, text):
            assert text in ("", "hello")
            return np.array([1, 2], dtype=np.uint32)[:len(text)]

        def decode(self, ids):
            return b"hello"[:len(ids)]

    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=Tokenizer))
    with quail.Session(CONFIG) as session:
        token_ids = session.tokenizer("hello")
        piece = session.tokenizer.decode([1])
        # the session loads one encoder for the caller and warms one on
        # the tokenization thread; the thread keeps its own instance
        session.inputs.warm_tokenizer().result()
    assert sources == ["/snapshots/pinned-model"] * 2
    assert resolutions == [(session.model.hf_name, session.model.revision, False)]
    assert token_ids == [1, 2]
    assert all(type(token_id) is int for token_id in token_ids)
    assert piece == b"h"


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_explain_analyze_shows_measured_rows_beside_estimates(
        sess, monkeypatch):
    query = sess.sql(FILTER_SQL + " LIMIT 1")
    execution_calls = []

    def executor(request):
        execution_calls.append(request)
        return make_executor(TRUTH)(request)

    original = execution.execute_query
    # Query.run binds execute_query at import, so patch it there
    monkeypatch.setattr(
        "quail.api.query.execute_query",
        lambda q, physical_executor=executor, plan=None: original(
            q, physical_executor=physical_executor, plan=plan))
    planned = query.explain()
    text = query.explain(analyze=True)

    assert len(execution_calls) == 1
    assert "   rows   " not in planned and "run:" not in planned
    header = next(line for line in text.splitlines() if "est. rows" in line)
    assert header.split() == ["est.", "rows", "rows", "est.", "pass", "pass",
                              "est.", "time", "time", "fresh", "tokens"]
    assert re.search(
        r"AiFilter: r\s+3\s+4\s+[\d.]+ ms\s+(?:<1|[\d.]+) ms\s+60$",
        text, re.M)
    # Both predicates have equal estimated cost and pass four of six rows.
    first = re.search(r"(?:1st: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                      r"\{0\}\\n\\nq\1:'\)\s+6\s+6\s+50%\s+66\.7%$",
                      text, re.M)
    second = re.search(r"(?:2nd: )?predicate ([12])  PROMPT\('DOCUMENT:\\n"
                       r"\{0\}\\n\\nq\1:'\)\s+3\s+4\s+50%\s+50%$",
                       text, re.M)
    assert first and second and first[1] != second[1]
    assert re.search(r"Scan reviews as r\s+6\s+6\s+<1 ms$", text, re.M)
    assert re.search(r"Project: r\.id\s+1\.5\s+2\s+<1 ms$", text, re.M)
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", text, re.M)
    assert "run:" in text
    assert "query time   1 s (model startup excluded)" in text
    assert "startup      500 ms" in text
    assert "throughput   6 documents/second over 6 input documents" in text
    assert "tokens       1,234 fresh" in text
    assert "GPU cost     $0.0011 per query (1 GPU at $3.9492/hour" in text

    # the executed result renders the same measured table on its own
    result = _run(sess.sql(FILTER_SQL + " LIMIT 1"), make_executor(TRUTH))
    explained = result.explain()
    assert re.search(r"Limit: 1\s+1\s+1\s+<1 ms$", explained, re.M)
    assert re.search(rf"(?:1st: )?predicate {first[1]}\s+6\s+6\s+50%\s+66\.7%$",
                     explained, re.M)
    stages = [s for s in result.report["stages"] if s["op"] == "filter"]
    assert [s["written_pos"] for s in stages] == [int(first[1]) - 1,
                                                int(second[1]) - 1]


def test_query_rows_observers_saved_reports_and_streamed_results(sess):
    res = _run(sess.sql(FILTER_SQL), make_executor(TRUTH))
    assert res.columns == ["r.id"]
    assert sorted(res.to_rows()) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6, abs=1e-3)
    assert stages[1]["evaluated"] == 4, "stage 2 only saw stage-1 survivors"
    assert res.report["wall_s"] == res.report["model_wall_s"] == 1.0
    assert res.report["finish_s"] >= 0
    assert res.report["input_ready_s"] >= res.report["planning_s"]
    assert "gpu_s" not in res.report

    def timed(request):
        response = make_executor(TRUTH)(request)
        response.metrics["gpu_s"] = 0.75
        return response

    limited = _run(sess.sql(FILTER_SQL + " LIMIT 1"), timed)
    assert limited.count() == 1
    assert limited.report["gpu_s"] == 0.75

    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)
    session = quail.Session(CONFIG, tokenizer=fake_tok, registry=registry)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": ["r0", "r1"], "review": ["first", "second"]}), id_col="id"))
    result = _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor({"r": {"q:": [1, 0]}}))
    assert result.observer(NodeTypes)["types"] == [
        "quail.scan", "quail.ai_filter", "quail.project", "quail.limit"]
    assert result.observer(NodeTypes) is result.observer("test.node_types")
    with pytest.raises(KeyError):
        result.observer("example.missing")

    # saved reports can be loaded separately from their result tables
    restored = QueryResult.from_table(result.collect(), report=dict(result.report))
    assert restored.plan is None
    restored.attach_executed_plan(registry.codecs)
    assert [node.node_id for node in restored.plan.topological_nodes()] == [
        node.node_id for node in result.plan.topological_nodes()]
    assert restored.node_metrics == result.node_metrics
    assert restored.explain() == result.explain()
    # a node's extra counters are saved beside the standard fields and
    # return to its extension metrics when the report is loaded
    report = dict(result.report)
    report["node_metrics"] = {
        node_id: {**metrics, "borrowed_prefix_tokens": 3}
        for node_id, metrics in report["node_metrics"].items()}
    counted = QueryResult.from_table(result.collect(), report=report)
    counted.attach_executed_plan(registry.codecs)
    assert all(metrics.extension == {"borrowed_prefix_tokens": 3}
               for metrics in counted.node_metrics.values())
    assert {node_id: metrics.wall_s
            for node_id, metrics in counted.node_metrics.items()} == {
        node_id: metrics.wall_s
        for node_id, metrics in result.node_metrics.items()}

    # results are assembled from survivor indices and join tables
    true_join_tables = [
        document_index_table({"r1": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1], "a1": [0, 0]}, "join_answers"),
        document_index_table({"r2": [0, 1, 1], "a2": [0, 0, 1]}, "join_answers"),
    ]
    survivors = {alias: pa.array(rows, type=pa.int32()) for alias, rows in
                 {"r1": [0, 1], "r2": [0, 1], "a1": [0], "a2": [0, 1]}.items()}
    declaration, _ = build_result_declaration(true_join_tables, survivors, "r1")
    assert count_rows(declaration) == 6

    relation = document_index_table({"r": list(range(5))}, "filter_survivors")
    assert relation.schema.metadata[b"quail.kind"] == b"filter_survivors"
    survivors = {"r": pa.array(range(5), type=pa.int32())}
    declaration, index_schema = build_result_declaration([], survivors, "r")
    schema = pa.schema([pa.field("r.id", pa.string(), nullable=False)],
                       metadata={b"quail.kind": b"query_result"})
    result = QueryResult(
        columns=["r.id"], declaration=declaration,
        document_index_schema=index_schema, output_schema=schema,
        projection=[("r", pa.array([f"r{i}" for i in range(5)]))],
        report={}, survivor_indices=survivors, true_join_tables={})

    reader = result.execute_stream(batch_rows=2)
    batches = list(reader)
    assert isinstance(reader, pa.RecordBatchReader)
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.Table.from_batches(batches).column("r.id").to_pylist() == [
        "r0", "r1", "r2", "r3", "r4"]
    assert result.count() == 5
    assert result.collect(limit=3).column("r.id").to_pylist() == ["r0", "r1", "r2"]
    assert not hasattr(result, "rows")


def test_order_by_offset_and_distinct_run_on_the_result():
    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)
    session = quail.Session(CONFIG, tokenizer=fake_tok, registry=registry)
    session.register("reviews", quail.DocumentProvider.from_table(
        pa.table({"id": [f"r{i}" for i in range(6)],
                  "review": [f"review {i} " + "pad " * 20 for i in range(6)],
                  "stars": [3, 1, 4, 5, None, 2]}), id_col="id"))
    # survivors of both predicates are r0 and r3; sorted by a column the
    # query does not return, so the first survivor to finish is not first
    result = _run(session.sql(FILTER_SQL + " ORDER BY r.stars DESC LIMIT 1"),
                  make_executor(TRUTH))
    assert result.to_rows() == [("r3",)]
    assert result.count() == 1
    assert result.observer(NodeTypes)["types"] == [
        "quail.scan", "quail.ai_filter", "quail.ai_filter",
        "quail.project", "quail.sort"]
    assert "Sort: r.stars DESC NULLS FIRST" in result.explain()

    result = _run(session.sql(
        "SELECT r.id, r.stars FROM reviews r WHERE AI_FILTER(PROMPT("
        "'q1: {0}', r.review)) ORDER BY r.stars NULLS FIRST OFFSET 1"
    ), make_executor(TRUTH))
    assert result.to_rows() == [("r1", 1), ("r0", 3), ("r3", 5)]

    result = _run(session.sql(
        "SELECT DISTINCT r.stars FROM reviews r WHERE AI_FILTER(PROMPT("
        "'q2: {0}', r.review)) ORDER BY r.stars LIMIT 2"
    ), make_executor({"r": {"q2:": [1, 1, 1, 1, 1, 1]}}))
    assert result.to_rows() == [(1,), (2,)]

    from quail.frontend.expressions import col, prompt
    built = (session.docs("reviews").alias("r")
             .ai_filter(prompt("q2: {0}", col("r.review")))
             .order_by(col("r.stars").desc().nulls_last()).offset(1).distinct()
             .limit(2).select("r.stars"))
    result = _run(built, make_executor({"r": {"q2:": [1, 1, 1, 1, 1, 1]}}))
    assert result.to_rows() == [(4,), (3,)]
    session.close()


class NodeTypes:
    """Observer that records the node types it saw."""

    name = "test.node_types"

    def __init__(self):
        self.types = []

    def after_node(self, node, result) -> None:
        self.types.append(node.type_name)

    def report(self) -> dict:
        return {"types": list(self.types)}


def _pair_query(session, on):
    claims = session.docs("claims").alias("c")
    partner = session.docs("evidence").alias("e")
    query = (claims.join(partner, on=quail.col("c.url") == quail.col("e.url"))
             if on else claims.join(partner))
    return query.ai_filter(
        quail.prompt("Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")),
        selectivity=0.5).select("c.id", "e.id")


def test_pair_table_lists_equal_keys_once_and_the_session_ships_it():
    pairs = pair_table(
        "c", [pa.array(["u1", "u2", None, "u1"])],
        "e", [pa.array(["u2", "u1", "u1"])])
    assert pairs.to_pydict() == {"c": [0, 0, 1, 3, 3], "e": [1, 2, 0, 1, 2]}
    # the planner estimates the fraction from key samples: u1 twice
    # against twice, u2 once against once, and a null never matches
    assert estimate_pair_fraction(
        [pa.array(["u1", "u2", None, "u1"])], [pa.array(["u2", "u1", "u1"])]
    ) == 5 / 12
    assert estimate_pair_fraction(
        [pa.array(["u1"])], [pa.array([], pa.string())]) == 0.0
    grouped = partner_map(pairs, "e", "c")
    assert [grouped.get(key).tolist() for key in range(3)] == [[1], [0, 3], [0, 3]]
    assert grouped.get(10) == ()
    unsorted = pa.table({"a": [5, 2, 5, 5], "b": [7, 3, 7, 1]})
    grouped = partner_map(unsorted, "a", "b")
    assert grouped.get(5).tolist() == [7, 7, 1]
    members = members_by_partner([[7], [1], [7], [3]], 0)
    assert allowed_members(grouped, members, 5).tolist() == [0, 0, 1, 2, 2]
    assert allowed_members({5: [7, 99]}, members, 5).tolist() == [0, 2]
    assert allowed_members(grouped, members, 9).size == 0
    assert allowed_members(grouped, members_by_partner([], 0), 5).size == 0
    assert partner_map(unsorted.slice(0, 0), "a", "b").get(5) == ()
    from quail.backends.quail.executor.operators.join import partner_list_builder

    streamed = {}
    group = [dict(anchor="a", partners=["b"], written_pos=j, pairs_from="pairs")
             for j in range(2)]
    lookup = partner_list_builder(
        group, [[[7], [1], [7], [3]]] * 2, {0: grouped, 1: streamed})
    selected = lookup(5, 0)
    assert selected.tolist() == [0, 0, 1, 2, 2]
    assert lookup(5, 0) is selected
    assert lookup(5, 1).size == 0
    streamed[5] = [1]
    assert lookup(5, 1).tolist() == [1]
    # Both equality keys must match, with integers promoted to a shared type.
    pairs = pair_table(
        "c", [pa.array(["u1", "u1"]), pa.array([1, 2])],
        "e", [pa.array(["u1", "u1"]), pa.array([2, 2], type=pa.int8())])
    assert pairs.to_pydict() == {"c": [1, 1], "e": [0, 1]}

    with quail.Session(CONFIG, tokenizer=lambda text: list(text.encode())) as session:
        register_claims_evidence(session)
        paired = _pair_query(session, on=True)
        cross = _pair_query(session, on=False)
        paired_plan, cross_plan = paired.plan(), cross.plan()
        paired_stage = paired_plan.graph.nodes_by_type(AiJoin.type_name)[0]
        cross_stage = cross_plan.graph.nodes_by_type(AiJoin.type_name)[0]
        # c1 and c2 pair with e0 and e2, c0 with e1: 5 of 12 pairs
        hash_join = paired_plan.graph.node("hash_join:c-e")
        assert (hash_join.left, hash_join.right,
                tuple((left.column, right.column) for left, right in hash_join.on)) == (
            "c", "e", (("url", "url"),))
        assert hash_join.pair_fraction == 5 / 12
        assert [port.source.node_id for port in hash_join.inputs] == [
            "scan:c", "scan:e"]
        assert paired_plan.graph.node("scan:c").shards == (range(0, 4),)
        assert paired_stage.pairs_from == "hash_join:c-e"
        assert not cross_stage.pairs_from
        assert "hash_join:c-e" not in {node.node_id for node in cross_plan.nodes}
        assert paired_stage.expected_tuples == round(
            cross_stage.expected_tuples * 5 / 12, 1)
        assert paired_plan.estimated_seconds < cross_plan.estimated_seconds
        assert "HashJoin" in paired.explain() and "c.url = e.url" in paired.explain()
        # the request carries the key columns, not the pairs
        request = paired._prepare_physical()
        assert request.column_tables()["c"].column("url").to_pylist() == [
            "u0", "u1", "u1", "u9"]
        assert cross._prepare_physical().relations == {}

        def answer(prompt, assignment):
            return (assignment["c"] + assignment["e"]) % 2 == 0

        estimate = quail.speed_of_light_estimate(paired, answer)
        assert estimate.join_pair_evaluations == 5
        assert estimate.join_stages[0]["passing_pairs"] == 2
        assert quail.speed_of_light_estimate(
            cross, answer).join_pair_evaluations == 12


IMPORT_TEXT = """
import sys
import quail
from bench import quailb
from quail.api import execute
from demos import quickstart

assert "modal" not in sys.modules
"""


def test_engine_import_gpu_requirement_and_quickstart_demos(
        sess, monkeypatch, tmp_path):
    subprocess.run([sys.executable, "-c", IMPORT_TEXT], check=True)
    monkeypatch.setattr(execution, "gpu_problem", lambda: "no CUDA GPU is visible")
    with pytest.raises(RuntimeError, match="process with a CUDA GPU") as error:
        sess.sql(FILTER_SQL).run()
    assert "no CUDA GPU is visible" in str(error.value)

    reviews = pa.table({"id": ["rv0", "rv1"], "body": [
        "The acting was excellent.", "The acting was poor."]})
    monkeypatch.setattr(quickstart, "load_reviews", lambda: reviews)
    commits = []
    monkeypatch.setattr(quickstart_modal, "RESULTS_DIR", tmp_path / "chosen-path")
    monkeypatch.setattr(quickstart_modal.volume, "commit",
                        lambda: commits.append("volume"))
    monkeypatch.setattr(quail.Session, "tokenizer", property(lambda self: str.split))
    monkeypatch.setattr(execution, "gpu_problem", lambda: None)
    monkeypatch.setattr(execution, "_prepare_backend", lambda *args: None)
    executor = make_executor({"r": {"Instruction": [1, 0]}})
    monkeypatch.setattr(execution, "_execute_physical",
                        lambda request, registry: executor(request))
    closed = []
    close = quail.Session.close

    def record_close(session):
        close(session)
        closed.append(session)

    monkeypatch.setattr(quail.Session, "close", record_close)

    for on_modal in (False, True):
        closed.clear()
        commits.clear()
        run = (quickstart_modal.run_query.get_raw_f() if on_modal
               else quickstart.run_query)
        rows, report = run()
        assert len(closed) == 1, on_modal
        assert closed[0].inputs._token_directory is None, on_modal
        assert rows.to_pylist() == [{"r.id": "rv0"}], on_modal
        assert report["fresh_tokens"] == 1234, on_modal
        assert report["worker_total_s"] >= 0, on_modal
        if not on_modal:
            assert "result_volume_path" not in report
            assert commits == []
            continue
        path = Path(report["result_volume_path"])
        assert path.parent == tmp_path / "chosen-path"
        assert json.loads(path.read_text()) == report
        assert commits == ["volume"]

    def fail():
        raise RuntimeError("query failed")

    commits.clear()
    monkeypatch.setattr(quickstart, "run_query", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        quickstart_modal.run_query.get_raw_f()()
    assert commits == ["volume"], "a failed query still commits the volume"


@pytest.mark.parametrize("left,right", [
    (pa.array([1, 2], pa.int8()), pa.array([2, 300], pa.int64())),
    (pa.array([1, 2], pa.int64()), pa.array([1.5, 2.0], pa.float64())),
    (pa.array([-1, 7, 2**63 - 1, None], pa.int64()),
     pa.array([7, 2**63 - 1, 2**63, 2**64 - 1, None], pa.uint64())),
    (pa.array([-1, 7], pa.int8()), pa.array([7, 2**64 - 1], pa.uint64())),
    (pa.array(["a", "b", None]).dictionary_encode(), pa.array(["b", "c", None])),
    (pa.chunked_array([pa.array(["a", "b"]).dictionary_encode(),
                       pa.array(["b", "a", None]).dictionary_encode()]),
     pa.array(["b", "a", None])),
    (pa.array([-1, 7], pa.int64()).dictionary_encode(),
     pa.array([7, 2**64 - 1], pa.uint64())),
    (pa.array([-2**63, 0, 7, 2**53, 2**53 + 1, 2**63 - 1, None], pa.int64()),
     pa.array([-float(2**63), 0.0, 7.0, 7.5, float(2**53), float(2**63),
               float("inf"), float("nan"), None])),
    (pa.array([0, 7, 2**63, 2**64 - 2048, 2**64 - 1], pa.uint64()),
     pa.array([-1.0, 0.0, 7.0, float(2**63), float(2**64 - 2048), float(2**64)])),
    (pa.array([None, None]), pa.array([1, 2], pa.int32())),
    (pa.array([None, None]), pa.array([None, None])),
    (pa.array([]), pa.array([])),
    (pa.array([0.0, -0.0, float("nan"), -float("nan"), None]),
     pa.array([-0.0, 0.0, float("nan"), -float("nan"), None])),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_join_keys_use_a_shared_type_for_execution_and_estimation(left, right, reverse):
    if reverse:
        left, right = right, left
    expected = [(i, j) for i, a in enumerate(left.to_pylist())
                for j, b in enumerate(right.to_pylist())
                if a is not None and b is not None and a == b]
    pairs = pair_table("l", [left], "r", [right])
    assert list(zip(pairs["l"].to_pylist(), pairs["r"].to_pylist())) == expected
    count = len(left) * len(right)
    assert estimate_pair_fraction([left], [right]) == (
        max(len(expected), 1) / count if count else 0.0)


@pytest.mark.parametrize("kind", [
    "signed_unsigned", "integer_float", "dictionary", "floating",
])
def test_mixed_join_keys_keep_exact_values_and_original_result_types(kind):
    left, right = {
        "signed_unsigned": (
            pa.array([-1, 7, 2**63 - 1], pa.int64()),
            pa.array([7, 2**63, 2**64 - 1], pa.uint64())),
        "integer_float": (
            pa.array([7, 2**53 + 1, 2**63 - 1], pa.int64()),
            pa.array([7.0, float(2**53), float(2**63)])),
        "dictionary": (pa.array(["a", "b", "c"]).dictionary_encode(),
                       pa.array(["b", "c", "d"])),
        "floating": (
            pa.array([-0.0, 0.0, float("nan"), -float("nan")], pa.float32()),
            pa.array([0.0, -0.0, float("nan"), -float("nan")], pa.float64())),
    }[kind]
    expected = [(i, j, a, b) for i, a in enumerate(left.to_pylist())
                for j, b in enumerate(right.to_pylist()) if a == b]
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        for name, keys in (("left_docs", left), ("right_docs", right)):
            session.register(name, quail.DocumentProvider.from_table(pa.table({
                "id": range(len(keys)), "key": keys, "text": ["document"] * len(keys),
            }), id_col="id"))
        query = session.sql("""
            SELECT l.id, r.id, l.key, r.key FROM left_docs l JOIN right_docs r
            ON l.key = r.key AND AI_FILTER(PROMPT('Pair? {0} {1}', l.text, r.text),
                                         {'anchor': 'l'})
        """)
        result = _run(query, make_executor({}, {("l", "r"): lambda *_args: True}))
        assert result.to_rows() == expected
        table = result.collect()
        assert table.schema.field("l.key").type == left.type
        assert table.schema.field("r.key").type == right.type


@pytest.mark.parametrize("left_alias,right_alias", [
    ("key0", "key1"), ("left", "right"), ("right", "left"),
])
def test_hash_join_keeps_ids_separate_from_keys_and_aliases(left_alias, right_alias):
    from types import SimpleNamespace

    from quail.catalog.pairs import columns_key
    from quail.execution.runtimes.dataflow import HashJoinRuntime
    from quail.physical import HashJoin, PortRef
    from quail.physical.base import input_ports

    node = HashJoin(
        node_id="join", left=left_alias, right=right_alias,
        inputs=input_ports((PortRef("l", f"ids:{left_alias}"),
                            PortRef("r", f"ids:{right_alias}"))),
        on=((quail.ColumnRef(left_alias, "", left_alias),
             quail.ColumnRef(right_alias, "", right_alias)),), written_pos=0)
    context = SimpleNamespace(sources={
        columns_key(left_alias): pa.table({left_alias: [30, 10, 20]}),
        columns_key(right_alias): pa.table({right_alias: [20, 40, 10]}),
    })
    result = HashJoinRuntime().execute(
        node, {"input:0": pa.array([1, 2]), "input:1": pa.array([0, 2])}, context)
    assert result.outputs["pairs:0"].to_pydict() == {
        left_alias: [1, 2], right_alias: [2, 0]}


@pytest.mark.parametrize("alias", ["answer", "predicate", "left,side"])
@pytest.mark.parametrize("joined", [False, True])
def test_answer_reporting_preserves_aliases_that_match_metadata_columns(alias, joined):
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(
            pa.table({"id": ["a", "b"], "text": ["one", "two"]}), id_col="id"))
        query = (session.docs("docs").alias(alias)
                 .ai_filter(quail.prompt("q: {0}", quail.col(f"{alias}.text"))))
        columns = [f"{alias}.id"]
        partner = "other,side"
        if joined:
            query = query.ai_join(
                session.docs("docs").alias(partner),
                quail.prompt("j: {0} {1}", quail.col(f"{alias}.text"),
                             quail.col(f"{partner}.text")), anchor=alias)
            columns.append(f"{partner}.id")
        result = _run(query.select(*columns), make_executor(
            {alias: {"q:": [True, False]}},
            {(alias, partner): lambda a, b: b == 1}))
        assert result.to_rows() == ([("a", "b")] if joined else [("a",)])
        if joined:
            join_stage = next(stage for stage in result.report["stages"]
                              if stage["op"] == "join")
            assert join_stage["partners"] == [partner]
        answers = result.answer_tables["filters"][(alias, 0)]
        assert len(set(answers.column_names)) == answers.num_columns
        assert answers.column(0).to_pylist() == [0, 1]
        assert answers.column(1).to_pylist() == [0, 0]
        assert answers.column(2).to_pylist() == [True, False]


def test_join_answer_constructor_preserves_an_alias_named_answer():
    from quail.execution.result import answer_table, true_answer_rows

    table = answer_table({"answer": [2, 4], "other": [1, 3]},
                         [False, True], "join_answers")
    assert len(set(table.column_names)) == table.num_columns
    assert true_answer_rows(table).to_pydict() == {"answer": [4], "other": [3]}
