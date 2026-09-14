"""End-to-end Session tests with a fake executor.

Covers gating, tuple assembly, projection, and the report.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fakes import register_claims_evidence

import quail
from quail.execution.pairs import pair_fraction, pair_table, partner_map
from quail.physical import AiJoin
from quail.planner.plan import EngineConfig


def _parquet(path, table):
    pq.write_table(pa.table(table), str(path))
    return str(path)


def fake_tok(text):
    return text.split()


def _run(query, execute):
    from quail.execution.execute import execute_query

    return execute_query(query, physical_executor=execute)


def runtime_plan(request):
    from quail.builtins import built_in_registry
    from quail.physical import (
        AiFilter,
        AiJoin,
        Scan,
        decode_graph,
    )

    graph = decode_graph(
        request.plan["graph"], built_in_registry().codecs
    )
    filter_nodes = {
        node.alias: node for node in graph.nodes
        if isinstance(node, AiFilter)
    }
    return {
        "graph": graph,
        "filter_nodes": filter_nodes,
        "filters": {
            alias: [list(question)
                    for question in node.question_token_ids]
            for alias, node in filter_nodes.items()
        },
        "joins": [stage.runtime_spec() for node in graph.nodes
                  if isinstance(node, AiJoin) for stage in node.stages],
        "shards": {
            node.alias: node.shards for node in graph.nodes
            if isinstance(node, Scan)
        },
    }


@pytest.fixture()
def sess(tmp_path):
    config = EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    s = quail.Session(config, tokenizer=fake_tok)
    # reviews: longer documents (they anchor); products: short
    s.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "r.parquet", {
            "id": [f"r{i}" for i in range(6)],
            "review": [f"review {i} " + "pad " * 20 for i in range(6)],
        }), id_col="id"))
    s.register("products", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "p.parquet", {
            "asin": [f"p{i}" for i in range(4)],
            "description": [f"product {i}" for i in range(4)],
        }), id_col="asin"))
    yield s
    s.close()


def test_session_requires_model_and_device():
    with pytest.raises(TypeError, match="'model' and 'device'"):
        EngineConfig()
    with pytest.raises(TypeError, match="'device'"):
        EngineConfig(model="qwen3-4b-fp8")
    with pytest.raises(TypeError, match="required positional argument: 'config'"):
        quail.Session()
    config = EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
    assert (config.gpus, config.backend) == (1, "quail")


def make_executor(filter_truth, join_truth=None, seen=None):
    """Build a fake executor from filter and join truth tables."""
    def _match_key(alias, q):
        for key in filter_truth[alias]:
            if any(t.startswith(key) for t in q):
                return key
        raise KeyError(f"no filter_truth key for alias {alias!r} "
                       f"matches tokens {q[:5]}")

    def _exec(request):
        from test_quail_backend import graph_state

        from quail.backends.quail.graph import execute_single_graph
        from quail.execution.runner import NodeResult
        from quail.execution.types import PhysicalResponse
        from quail.physical import AiFilter, Scan

        runtime = runtime_plan(request)
        if seen is not None:
            seen["request"] = request
            seen.update(runtime)
        graph = runtime["graph"]
        docs = {
            node.alias: request.inputs[node.input_id].documents
            for node in graph.nodes if isinstance(node, Scan)
        }

        class FixedAnswers:
            def execute(self, node, inputs):
                if isinstance(node, AiFilter):
                    rows = {}
                    for document in inputs["document_ids"]:
                        row = []
                        for question in node.question_token_ids:
                            bit = filter_truth[node.alias][
                                _match_key(node.alias, question)
                            ][document]
                            row.append(bit)
                            if not bit:
                                break
                        rows[document] = row
                    return NodeResult({
                        f"ids:{node.alias}": [d for d, row in rows.items()
                            if len(row) == len(node.stages) and all(row)],
                        f"filter_answers:{node.alias}": rows,
                    })
                anchors = list(inputs["anchor_ids"])
                outputs = {}
                for stage in node.stages:
                    partners = inputs["partner_indices"][stage.written_pos]
                    rule = join_truth[(stage.anchor, *stage.partners)]
                    rows = {i: [rule(a, *pair) for pair in partners]
                            for i, a in enumerate(anchors)}
                    outputs[f"join_answers:{stage.written_pos}"] = {
                        "rows": rows, "anchor_index": anchors,
                        "partner_index": partners, "anchor": stage.anchor,
                        "partners": list(stage.partners),
                        "semantics": stage.semantics,
                        "selectivity": stage.selectivity,
                        "written_pos": stage.written_pos,
                    }
                passing = {anchors[i] for i, row in rows.items() if any(row)}
                outputs[f"ids:{node.anchor}"] = [a for a in anchors
                    if (a not in passing if stage.semantics == "anti"
                        else a in passing)]
                return NodeResult(outputs)

        report = execute_single_graph(graph_state(FixedAnswers(), docs), {
            "filter_limit": None, "pre_ids": [],
        }, graph)
        outputs = report.pop("_outputs")
        return PhysicalResponse(outputs, {
            **report, "wall_s": 1.0, "boot_s": 0.5, "fresh_tokens": 1234,
        })

    return _exec


FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""


def test_query_rows_observers_and_saved_reports(sess, tmp_path):
    truth = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
                         "q2:": [1, 0, 1, 1, 0, 1]}}
    q = sess.sql(FILTER_SQL)
    res = _run(q, make_executor(truth))
    assert res.columns == ["r.id"]
    assert sorted(res.to_rows()) == [("r0",), ("r3",)]
    stages = [s for s in res.report["stages"] if s["op"] == "filter"]
    assert stages[0]["evaluated"] == 6
    assert stages[0]["observed_selectivity"] == pytest.approx(4 / 6,
                                                              abs=1e-3)
    # stage 2 only saw stage-1 survivors
    assert stages[1]["evaluated"] == 4
    assert res.report["wall_s"] == 1.0

    limited = _run(
        sess.sql(FILTER_SQL + " LIMIT 1"), make_executor(truth))
    assert limited.count() == 1

    registry = quail.ExtensionRegistry.with_built_ins().register_observer(
        NodeTypes)

    result = _observed_result(tmp_path, registry)

    assert result.observer(NodeTypes)["types"] == [
        "quail.scan",
        "quail.ai_filter",
        "quail.project",
        "quail.limit",
    ]
    assert result.observer(NodeTypes) is result.observer("test.node_types")
    with pytest.raises(KeyError):
        result.observer("example.missing")

    from quail.execution.result import QueryResult

    # Saved reports can be loaded separately from their result tables.
    table, report = result.collect(), dict(result.report)

    restored = QueryResult.from_table(table, report=report)
    assert restored.plan is None
    restored.attach_executed_plan(registry.codecs)

    assert [node.node_id for node in restored.plan.topological_nodes()] == [
        node.node_id for node in result.plan.topological_nodes()]
    assert restored.node_metrics == result.node_metrics
    assert restored.explain() == result.explain()


class NodeTypes:
    """Observer that records the node types it saw."""

    name = "test.node_types"

    def __init__(self):
        self.types = []

    def after_node(self, node, result) -> None:
        self.types.append(node.type_name)

    def report(self) -> dict:
        return {"types": list(self.types)}


def _observed_result(tmp_path, registry):
    session = quail.Session(
        EngineConfig(
            gpus=1,
            model="qwen3-4b-fp8",
            backend="quail",
            device="h100-sxm",
        ),
        tokenizer=fake_tok,
        registry=registry,
    )
    session.register("reviews", quail.DocumentProvider.from_parquet(
        _parquet(tmp_path / "reviews.parquet", {
            "id": ["r0", "r1"],
            "review": ["first", "second"],
        }),
        id_col="id",
    ))
    truth = {"r": {"q:": [1, 0]}}
    return _run(session.sql(
        "SELECT r.id FROM reviews r WHERE "
        "AI_FILTER(PROMPT('q: {0}', r.review)) LIMIT 1"
    ), make_executor(truth))


def test_pair_table_lists_equal_keys_once_each():
    pairs = pair_table(
        "c", [pa.array(["u1", "u2", None, "u1"])],
        "e", [pa.array(["u2", "u1", "u1"])])
    assert pairs.to_pydict() == {"c": [0, 0, 1, 3, 3], "e": [1, 2, 0, 1, 2]}
    assert pair_fraction(pairs, 4, 3) == 5 / 12
    assert partner_map(pairs, "e", "c") == {0: [1], 1: [0, 3], 2: [0, 3]}
    # two equalities: both key columns must match; integer keys on one
    # side are cast to the other side's type
    pairs = pair_table(
        "c", [pa.array(["u1", "u1"]), pa.array([1, 2])],
        "e", [pa.array(["u1", "u1"]), pa.array([2, 2], type=pa.int8())])
    assert pairs.to_pydict() == {"c": [1, 1], "e": [0, 1]}


def _pair_query(session, on):
    query = session.docs("claims").alias("c")
    partner = session.docs("evidence").alias("e")
    if on:
        query = query.join(partner, on=quail.col("c.url") == quail.col("e.url"))
    else:
        query = query.join(partner)
    return query.ai_filter(
        quail.prompt("Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")),
        selectivity=0.5).select("c.id", "e.id")


def test_session_plans_prices_and_ships_the_pair_table():
    config = EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    with quail.Session(
        config, tokenizer=lambda text: list(text.encode())
    ) as session:
        register_claims_evidence(session)
        paired = _pair_query(session, on=True)
        cross = _pair_query(session, on=False)
        paired_plan, cross_plan = paired.plan(), cross.plan()
        paired_stage = paired_plan.graph.nodes_by_type(
            AiJoin.type_name)[0].stages[0]
        cross_stage = cross_plan.graph.nodes_by_type(
            AiJoin.type_name)[0].stages[0]
        # c1 and c2 pair with e0 and e2, c0 with e1: 5 of 12 pairs
        hash_join = paired_plan.graph.node("hash_join:c-e")
        assert (hash_join.left, hash_join.right, hash_join.on) == (
            "c", "e", (("url", "url"),))
        assert hash_join.pair_fraction == 5 / 12
        assert [port.source.node_id for port in hash_join.inputs] == [
            "scan:c", "scan:e"]
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


def test_gpu_seconds_reach_the_result_report(sess):
    truth = {"r": {"q1:": [1, 0, 1, 0, 1, 0], "q2:": [1, 1, 1, 1, 1, 1]}}
    executor = make_executor(truth)

    def timed(request):
        response = executor(request)
        response.metrics["gpu_s"] = 0.75
        return response

    assert "gpu_s" not in _run(sess.sql(FILTER_SQL), executor).report
    assert _run(sess.sql(FILTER_SQL), timed).report["gpu_s"] == 0.75
