"""Fixed join execution, filter retention, and answer reconstruction."""

import pyarrow as pa
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import TOKENIZER

import quail
from bench import quailb
from quail.api.config import EngineConfig
from quail.api.execute import execute_query
from quail.execution.metrics import NodeMetrics
from quail.physical import (
    AiJoin,
    PortRef,
)
from quail.physical.plan import graph_from_envelope
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


def fever_oracle(empty=False):
    """The oracle answering FEVER: c0 and c1 pass every filter unless empty."""
    return Oracle(
        filters=lambda alias, document, position: document < 2 and not empty,
        joins=_fever_join, labels=_fever_label,
        join_metrics=NodeMetrics(gpu_s=0.02, peak_gpu_bytes=128, extension={
            "pack_s": 0.125, "borrowed_prefix_tokens": 16}))


def fever_executor(check=None, empty=False):
    """Return a physical executor that answers FEVER on the request's GPUs."""
    def execute(request):
        graph = graph_from_envelope(request.plan)
        if check:
            check(request, graph)
        response = oracle_executor(fever_oracle(empty))(request)
        if request.gpus > 1:
            # each GPU measured the planted join metrics: times and
            # counts add up, and the peak is the largest
            for node in graph.nodes_by_type(AiJoin.type_name):
                metrics = response.metrics["node_metrics"][node.node_id]
                assert metrics["pack_s"] == 0.25
                assert metrics["borrowed_prefix_tokens"] == 32
                assert metrics["gpu_s"] == 0.04
                assert metrics["peak_gpu_bytes"] == 128
        return response

    return execute


# GPUs, whether no document passes a filter, and the filters' estimate
FEV9_RUNS = [
    (1, False, 0.001), (1, False, 1.0), (1, False, 0.5), (1, True, 1.0),
    (2, False, None),
]


def test_fev9_executes_bound_join_nodes_without_the_optimizer(
        monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("execution planned the query again")

    for gpus, empty, estimate in FEV9_RUNS:
        case = f"gpus={gpus} empty={empty} estimate={estimate}"
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
                execute = fever_executor(check, empty)
                result = execute_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == ([] if empty else FEV9_ROWS), case


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
        execute = fever_executor()
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
        execute = fever_executor()
        result = execute_query(query, physical_executor=execute)
        assert result.collect().to_pylist() == [
            {"c.id": "c0", "e.id": "e0", "topic": "science"}]
        # the classification ran first, over every claim
        labels = result.answer_tables["classifies"]["topic"]
        assert labels.column("topic").to_pylist() == ["science", "sport", "science"]
