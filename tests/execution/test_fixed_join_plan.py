"""Fixed join execution, filter retention, and answer reconstruction."""

import pyarrow as pa
import pytest
from helpers.fakes import graph_state, run_graph
from helpers.oracle import Oracle
from helpers.sessions import TOKENIZER

import quail
from bench import quailb
from quail.api.config import EngineConfig
from quail.api.execute import execute_query
from quail.execution.metrics import NodeMetrics
from quail.execution.types import PhysicalResponse
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    PortRef,
    Scan,
)
from quail.physical.plan import PlanEditError, graph_from_envelope
from quail_b import prompts
from quail_b.queries import FILTER_SELECTIVITY_ESTIMATES

FEV9_ROWS = [{"c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1"}]


def _session(gpus=1, backend="quail"):
    session = quail.Session(
        EngineConfig(gpus=gpus, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=TOKENIZER)
    register_fever(session)
    return session


def register_fever(session):
    for name, column, prefix, length in (
        ("claims", "claim", "c", 20), ("evidence", "text", "e", 300),
    ):
        table = pa.table({
            "id": [f"{prefix}{index}" for index in range(3)],
            column: [f"{index} " + "word " * length for index in range(3)],
        })
        if name == "claims":
            # c0 and c1 name page e0, c2 names e2 (an evidence id)
            table = table.append_column(
                "evidence_wiki_url", pa.array(["e0", "e0", "e2"]))
        session.register(name, quail.DocumentProvider.from_table(
            table, id_col="id"))


# the label a classification gives claims c0, c1, c2
FEVER_LABELS = ("science", "sport", "science")
# the stance a classification gives each (claim, evidence) pair
FEVER_STANCES = {(0, 0): "supports", (1, 1): "refutes", (2, 0): "supports"}


def _fever_label(spec, ids):
    if len(ids) == 1:
        return FEVER_LABELS[ids[0]]
    # the stances are keyed (claim, evidence) whichever the anchor
    return FEVER_STANCES.get(ids if spec.anchor.startswith("c") else ids[::-1])


def _fever_join(node, assignment):
    claim = next(value for alias, value in assignment.items() if alias.startswith("c"))
    evidence = next(value for alias, value in assignment.items()
                    if alias.startswith("e"))
    passing = ({(1, 0), (2, 0), (0, 2)} if node.written_position == 1
               else {(0, 0), (1, 1), (2, 0), (1, 2)})
    return (claim, evidence) in passing


def fever_oracle(capacity, empty):
    """The oracle answering FEVER: c0 and c1 pass every filter unless empty."""
    return Oracle(
        filters=lambda alias, document, position: document < 2 and not empty,
        joins=_fever_join, labels=_fever_label, retain=capacity,
        join_metrics=NodeMetrics(gpu_s=0.02, peak_gpu_bytes=128, extension={
            "pack_s": 0.125, "borrowed_prefix_tokens": 16}))


def fever_executor(session, monkeypatch, gpus, check=None, capacity=1,
                   empty=False):
    """Return a physical executor that runs the FEVER fakes on one or two GPUs."""
    from helpers.fakes import thread_servers

    from quail.backends.quail.graph import execute_staged_graph
    from quail.specs import H100_SXM

    def execute(request):
        graph = graph_from_envelope(request.plan)
        documents = {node.alias: request.inputs[node.input_id].documents
                for node in graph.nodes if isinstance(node, Scan)}
        extra = check(request, graph) if check else {}
        if gpus == 1:
            model = fever_oracle(capacity, empty)
            state = graph_state(model, documents)
            model.arena = state["gpu"].arena
            state.update(extra)
            report = run_graph(state, request.plan["settings"], graph)
            assert not state["gpu"].arena.accounting.owned
            return PhysicalResponse(report.pop("_outputs"), report)
        states = []
        for _ in range(gpus):
            model = fever_oracle(capacity=1, empty=False)
            state = graph_state(model, documents)
            model.arena = state["gpu"].arena
            state.update(extra)
            states.append(state)
        payload = {**request.plan["settings"], **extra,
                   "columns": request.column_tables()}
        report = execute_staged_graph(
            payload, graph, gpus, thread_servers(states, payload, graph), H100_SXM)
        assert all(not state["gpu"].arena.accounting.owned for state in states)
        for node in graph.nodes:
            if isinstance(node, AiJoin):
                metrics = report["node_metrics"][node.node_id]
                assert metrics["pack_s"] == 0.25
                assert metrics["borrowed_prefix_tokens"] == 32
                assert metrics["gpu_s"] == 0.04
                assert metrics["peak_gpu_bytes"] == 128
        return PhysicalResponse(report.pop("_outputs"), report)

    return execute


FEV9_RUNS = [
    (1, 10, False, 0.001), (1, 1, False, 1.0), (1, 0, False, 0.5),
    (1, 0, True, 1.0), (2, 1, False, None),
]


def test_fev9_executes_bound_and_edited_join_nodes_without_the_optimizer(
        monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("execution planned the query again")

    for gpus, capacity, empty, estimate in FEV9_RUNS:
        case = f"gpus={gpus} capacity={capacity} empty={empty}"
        with monkeypatch.context() as patch:
            if estimate is not None:
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F11, estimate)
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F13, estimate)

            def check(request, graph):
                assert len(graph.nodes_by_type(AiJoin.type_name)) == 3
                patch.setattr("quail.api.query.plan_from_json",
                              unexpected_search)
                return {}

            with _session(gpus=gpus) as session:
                query = quailb.queries(session)["FEV-9"][1]()
                execute = fever_executor(
                    session, patch, gpus, check, capacity, empty)
                result = execute_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == ([] if empty else FEV9_ROWS), case

    with _session() as session:
        query = quailb.queries(session)["FEV-9"][1]()
        plan = query.plan()
        join = next(node for node in plan.nodes if isinstance(node, AiJoin))
        sequence = next(node for node in plan.nodes
                     if isinstance(node, AiFilter) and node.alias == join.anchor)
        assert not sequence.keep_kv
        edited = plan.insert(
            Barrier(node_id=f"barrier:{sequence.alias}", next_anchor=sequence.alias,
                    aliases=(sequence.alias,)),
            between=(sequence.node_id, join.node_id))
        # cut off from its join by the barrier, the sequence retains KV
        assert edited.graph.node(sequence.node_id).keep_kv
        assert edited != plan

        seen = []

        def check(request, graph):
            seen.append([node.node_id for node in graph.nodes])
            return {}

        execute = fever_executor(session, monkeypatch, 1, check, capacity=10)
        rows = execute_query(query, physical_executor=execute).collect()
        edited_result = execute_query(
            quailb.queries(session)["FEV-9"][1](), physical_executor=execute,
            plan=edited)
        assert f"barrier:{sequence.alias}" in seen[1]
        assert f"barrier:{sequence.alias}" not in seen[0]
        assert edited_result.collect().to_pylist() == rows.to_pylist() == FEV9_ROWS


TOPIC_CALL = ("AI.CLASSIFY(PROMPT('What is {0} about?', c.claim), "
              "['science', 'sport'])")
TOPIC_SQL = f"{TOPIC_CALL} AS topic"
PLAUSIBLE_SQL = ("AI_FILTER(PROMPT('Is {0} plausible?', c.claim), "
                 "{'selectivity': 0.9})")


def test_classification_runs_beside_filters_and_joins(monkeypatch):
    with _session() as session:
        # filter then classify: the label is computed for the survivors
        query = session.sql(f"SELECT c.id, {TOPIC_SQL} FROM claims c "
                            f"WHERE {PLAUSIBLE_SQL}")
        plan = query.plan()
        assert [type(node).__name__ for node in plan.nodes] == [
            "Scan", "AiFilter", "AiClassify"]
        classify = plan.nodes[2]
        assert classify.inputs[0].source == PortRef(plan.nodes[1].node_id,
                                                    "rows:c")
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        rows = execute_query(query, physical_executor=execute).collect()
        assert rows.to_pylist() == [{"c.id": "c0", "topic": "science"},
                                    {"c.id": "c1", "topic": "sport"}]

        # classify, keep one label, then join: the join takes the
        # accepted claims, and the label rides the output rows
        query = session.sql(f"""
            SELECT c.id, e.id, {TOPIC_SQL} FROM claims c
            JOIN evidence e
              ON AI_FILTER(PROMPT('Does {{1}} support {{0}}?', c.claim, e.text),
                           {{'selectivity': 0.5}})
            WHERE {PLAUSIBLE_SQL}
              AND HINT({TOPIC_CALL} IN ('science'), {{'selectivity': 0.5}})
        """)
        plan = query.plan()
        # the label test rejects half the claims and the filter a tenth,
        # so the classification goes first; the classification, the
        # filter on its label, and the sequence run as one pipeline,
        # and the join joins it only when it anchors on the claims
        assert [type(node).__name__ for node in plan.nodes] == [
            "Scan", "Scan", "AiClassify", "Filter", "AiFilter", "AiJoin"]
        test, ask, join = plan.nodes[3:6]
        members = tuple(
            member.node_id for member in plan.graph.pipelines()["ai-classify:0"])
        assert members == (("ai-classify:0", test.node_id, ask.node_id)
                           + ((join.node_id,) if join.anchor == "c" else ()))
        (claims_port,) = [port.source for port in join.inputs
                          if port.source.port == "rows:c"]
        assert claims_port == PortRef(ask.node_id, "rows:c")
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        result = execute_query(query, physical_executor=execute)
        assert result.collect().to_pylist() == [
            {"c.id": "c0", "e.id": "e0", "topic": "science"}]
        # the classification ran first, over every claim
        labels = result.answer_tables["classifies"]["topic"]
        assert labels.column("topic").to_pylist() == ["science", "sport", "science"]


def test_a_removed_barrier_restores_the_plan_and_a_filter_cannot_be_removed():
    with _session() as session:
        plan = quailb.queries(session)["FEV-9"][1]().plan()
        join = next(node for node in plan.nodes if isinstance(node, AiJoin))
        sequence = next(node for node in plan.nodes
                        if isinstance(node, AiFilter) and node.alias == join.anchor)
        barrier = f"barrier:{sequence.alias}"
        edited = plan.insert(
            Barrier(node_id=barrier, next_anchor=sequence.alias,
                    aliases=(sequence.alias,)),
            between=(sequence.node_id, join.node_id))
        restored = edited.remove(barrier)
        assert [node.node_id for node in restored.nodes] == [
            node.node_id for node in plan.nodes]
        assert not restored.graph.node(sequence.node_id).keep_kv
        with pytest.raises(PlanEditError, match="would change what the query means"):
            plan.remove(sequence.node_id)
