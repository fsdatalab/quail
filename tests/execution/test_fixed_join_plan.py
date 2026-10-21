"""Fixed join execution, filter retention, and answer reconstruction."""

import pyarrow as pa
import pytest
from helpers.fakes import graph_state, letter_tokens
from helpers.oracle import Oracle

import quail
from bench import quailb
from quail.api.execute import execute_query
from quail.backends.quail.graph import execute_single_graph
from quail.execution.metrics import NodeMetrics
from quail.execution.types import PhysicalResponse
from quail.logical.expressions import CompileError, column_name
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    PortRef,
    Project,
    Scan,
    decode_graph,
    encode_graph,
)
from quail.physical.pipelines import build_pipelines
from quail.planner.plan import EngineConfig
from quail_b import prompts
from quail_b.queries import FILTER_SELECTIVITY_ESTIMATES

FEV9_ROWS = [{"c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1"}]


def _session(gpus=1, backend="quail"):
    session = quail.Session(
        EngineConfig(gpus=gpus, model="qwen3-4b-fp8", backend=backend,
                     device="h100-sxm"),
        tokenizer=letter_tokens)
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
    from quail.backends.quail import worker
    from quail.backends.quail.distributed import execute_distributed_graph
    from quail.specs import H100_SXM

    def child(documents):
        model = fever_oracle(capacity=1, empty=False)
        state = graph_state(model, documents)
        model.arena = state["gpu"].arena
        return worker.ChildState(
            gpu_index=0, gpu=state["gpu"], registry=session.registry)

    def execute(request):
        graph = decode_graph(request.plan["graph"], session.registry.codecs)
        documents = {node.alias: request.inputs[node.input_id].documents
                for node in graph.nodes if isinstance(node, Scan)}
        extra = check(request, graph) if check else {}
        if gpus == 1:
            model = fever_oracle(capacity, empty)
            state = graph_state(model, documents)
            model.arena = state["gpu"].arena
            state.update(extra, functions=session.registry.functions)
            report = execute_single_graph(state, request.plan["settings"], graph)
            assert not state["gpu"].arena.accounting.owned
            return PhysicalResponse(report.pop("_outputs"), report)
        children = [child(documents), child(documents)]

        def round_fn(kind, subs):
            assert kind == "execute"
            return [worker._child_execute(state, sub)
                    for state, sub in zip(children, subs)]

        report = execute_distributed_graph(
            {**request.plan["settings"], "model": "qwen3-4b-fp8",
             "documents": documents, **extra, "physical_plan": request.plan},
            graph, 2, round_fn, H100_SXM, session.registry.runtimes,
            session.registry,
        )
        assert all(not child.gpu.arena.accounting.owned for child in children)
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

REQUEST_BACKENDS = ["stock_vllm", "pipelined_vllm", "pipelined_sglang"]


def test_fev9_executes_bound_and_edited_join_nodes_without_the_optimizer(
        monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("execution called the join optimizer")

    for gpus, capacity, empty, estimate in FEV9_RUNS:
        case = f"gpus={gpus} capacity={capacity} empty={empty}"
        with monkeypatch.context() as patch:
            if estimate is not None:
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F11, estimate)
                patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F13, estimate)

            def check(request, graph):
                assert len(graph.nodes_by_type(AiJoin.type_name)) == 3
                patch.setattr("quail.planner.search.join_order.search_joins",
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


def test_classification_runs_beside_filters_and_joins(monkeypatch):
    with _session() as session:
        claims = session.docs("claims").alias("c")
        evidence = session.docs("evidence").alias("e")
        topic = quail.prompt("What is {0} about?", quail.col("c.claim"))
        plausible = quail.prompt("Is {0} plausible?", quail.col("c.claim"))
        # filter then classify: the label is computed for the survivors
        query = (claims.ai_filter(plausible, selectivity=0.9)
                 .ai_classify(topic, ["science", "sport"], name="topic")
                 .select("c.id", "topic"))
        plan = query.plan()
        assert [type(node).__name__ for node in plan.nodes] == [
            "Scan", "AiFilter", "AiClassify", "Project"]
        classify = plan.nodes[2]
        assert classify.inputs[0].source == PortRef(plan.nodes[1].node_id,
                                                    "rows:c")
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        rows = execute_query(query, physical_executor=execute).collect()
        assert rows.to_pylist() == [{"c.id": "c0", "topic": "science"},
                                    {"c.id": "c1", "topic": "sport"}]

        # classify, keep one label, then join: the join takes the
        # accepted claims, and the label rides the output rows
        claims = session.docs("claims").alias("c")
        evidence = session.docs("evidence").alias("e")
        query = (claims.ai_filter(plausible, selectivity=0.9)
                 .ai_classify(topic, ["science", "sport"], name="topic")
                 .label_in("topic", ["science"], selectivity=0.5)
                 .ai_join(evidence, quail.prompt(
                     "Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")), selectivity=0.5)
                 .select("c.id", "e.id", "topic"))
        plan = query.plan()
        # the label test rejects half the claims and the filter a tenth,
        # so the classification goes first; the classification, the
        # filter on its label, and the sequence run as one pipeline,
        # and the join joins it only when it anchors on the claims
        assert [type(node).__name__ for node in plan.nodes] == [
            "Scan", "Scan", "AiClassify", "Filter", "AiFilter", "AiJoin",
            "Project"]
        test, ask, join, project = plan.nodes[3:7]
        members = tuple(
            member.node_id for member in build_pipelines(plan.graph)["ai-classify:0"])
        assert members == (("ai-classify:0", test.node_id, ask.node_id)
                           + ((join.node_id,) if join.anchor == "c" else ()))
        (claims_port,) = [port.source for port in join.inputs
                          if port.source.port == "rows:c"]
        assert claims_port == PortRef(ask.node_id, "rows:c")
        assert [port.source.port for port in project.inputs] == [
            "join_answers:0", "rows:c"]
        codecs = session.registry.codecs
        assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        result = execute_query(query, physical_executor=execute)
        assert result.collect().to_pylist() == [
            {"c.id": "c0", "e.id": "e0", "topic": "science"}]
        # the classification ran first, over every claim
        labels = result.answer_tables["classifies"]["topic"]
        assert labels.column("topic").to_pylist() == ["science", "sport", "science"]


def test_a_classification_of_joined_rows_follows_its_join(monkeypatch):
    """A join, then a classification of each row it keeps.

    The classification runs in the join's pipeline on its anchor, its
    label rides the output rows, and a row without a label leaves the
    result.
    """
    with _session() as session:
        claims = session.docs("claims").alias("c")
        evidence = session.docs("evidence").alias("e")
        stance = quail.prompt("How does {1} bear on {0}?",
                              quail.col("c.claim"), quail.col("e.text"))
        query = (claims.ai_join(evidence, quail.prompt(
                     "Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")), selectivity=0.5)
                 .ai_classify(stance, ["supports", "refutes"], name="stance")
                 .select("c.id", "e.id", "stance"))
        plan = query.plan()
        assert [type(node).__name__ for node in plan.nodes] == [
            "Scan", "Scan", "AiJoin", "AiClassify", "Project"]
        join, classify, project = plan.nodes[2], plan.nodes[3], plan.nodes[4]
        assert classify.spec.aliases == (join.anchor,) + tuple(
            alias for alias in ("c", "e") if alias != join.anchor)
        assert classify.spec.join_layout is not None and classify.spec.scoring == \
            "letters"
        assert classify.inputs[0].source == PortRef(join.node_id,
                                                    "join_answers:0")
        assert [port.source.port for port in project.inputs] == [
            "join_answers:0", "rows"]
        members = tuple(
            member.node_id for member in build_pipelines(plan.graph)[join.node_id])
        assert members == (join.node_id, classify.node_id)
        codecs = session.registry.codecs
        assert decode_graph(encode_graph(plan.graph, codecs), codecs) == plan.graph
        explained = query.explain()
        assert "stance" in explained
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        result = execute_query(query, physical_executor=execute)
        # (c1, e2) is a true pair with no stance and leaves the result
        rows = result.collect().to_pylist()
        assert sorted((row["c.id"], row["e.id"], row["stance"])
                      for row in rows) == [
            ("c0", "e0", "supports"), ("c1", "e1", "refutes"),
            ("c2", "e0", "supports")]
        labels = result.answer_tables["classifies"]["stance"]
        assert set(labels.column_names) == {"c", "e", "stance"}
        assert labels.column_names[0] == join.anchor

        # a filter on a joined row's label is refused
        claims = session.docs("claims").alias("c")
        evidence = session.docs("evidence").alias("e")
        with pytest.raises(CompileError, match="filter on a label"):
            (claims.ai_join(evidence, quail.prompt(
                 "Does {1} support {0}?", quail.col("c.claim"),
                 quail.col("e.text")), selectivity=0.5)
             .ai_classify(stance, ["supports", "refutes"], name="stance")
             .label_in("stance", ["supports"]))


def test_request_backends_plan_fev9_and_a_single_join():
    for backend in REQUEST_BACKENDS:
        with _session(backend=backend) as session:
            plan = quailb.queries(session)["FEV-9"][1]().plan()
            execution = next(
                node for node in plan.nodes if hasattr(node, "joins"))
            assert execution.filters[-1].alias == execution.joins[0].anchor, \
                backend
            plan = (session.docs("claims").alias("c")
                    .ai_join(session.docs("evidence").alias("e"),
                             quail.prompt("m {0} {1}", quail.col("c.claim"),
                                          quail.col("e.text")))
                    .select("c.id", "e.id")).plan()
        sink = next(node for node in plan.nodes if isinstance(node, Project))
        assert sink.inputs[0].source.port == "join_answers:0", backend


def test_fev10_asks_the_model_about_same_page_pairs_only(monkeypatch):
    def check(request, graph):
        columns = request.column_tables()
        assert columns["c"].column("evidence_wiki_url").to_pylist() == [
            "e0", "e0", "e2"]
        return {"columns": columns}

    for gpus in (1, 2):
        with _session(gpus=gpus) as session:
            query = quailb.queries(session)["FEV-10"][1]()
            plan = query.plan()
            stage = plan.graph.nodes_by_type(AiJoin.type_name)[0]
            hash_join = plan.graph.node("hash_join:c-e")
            assert [(column_name(left), column_name(right))
                    for left, right in hash_join.on] == [
                        ("c.evidence_wiki_url", "e.id")]
            assert hash_join.pair_fraction == 3 / 9
            assert stage.pairs_from == "hash_join:c-e"
            result = execute_query(
                query, physical_executor=fever_executor(session, monkeypatch, gpus,
                                                        check))
            # of the same-page pairs (c0, e0) and (c1, e0), only c0 is
            # supported; a cross join would also have returned (c1, e1)
            answers = result.answer_tables["joins"][0]
            assert sorted(zip(answers.column("c").to_pylist(),
                              answers.column("e").to_pylist())) == [(0, 0), (1, 0)]
            assert result.collect().to_pylist() == [{"c.id": "c0", "e.id": "e0"}]


def test_classification_moves_after_a_selective_join(monkeypatch):
    with _session() as session:
        topic = quail.prompt("What is {0} about?", quail.col("c.claim"))
        claims = session.docs("claims").alias("c")
        evidence = session.docs("evidence").alias("e")
        # a join that almost nothing survives and a filter on a label that
        # keeps almost everything: classifying the few matched claims
        # after the join is cheaper than classifying every claim before
        query = (claims
                 .ai_classify(topic, ["science", "sport"], name="topic")
                 .label_in("topic", ["science"], selectivity=0.99)
                 .ai_join(evidence, quail.prompt(
                     "Does {1} support {0}?", quail.col("c.claim"),
                     quail.col("e.text")), selectivity=0.001)
                 .select("c.id", "e.id", "topic"))
        plan = query.plan()
        assert plan.settings["classify_placement"] == "after joins"
        names = [type(node).__name__ for node in plan.nodes]
        assert names.index("AiJoin") < names.index("AiClassify") < names.index(
            "Filter") < names.index("Recombine")
        execute = fever_executor(session, monkeypatch, 1, capacity=10)
        result = execute_query(query, physical_executor=execute)
        rows = result.collect().to_pylist()
        # c0 and c2 are supported by e0 and labeled science; c1 is not
        assert rows == [{"c.id": "c0", "e.id": "e0", "topic": "science"},
                        {"c.id": "c2", "e.id": "e0", "topic": "science"}]
        # the join matched every claim (c1 with e1 and e2), so every
        # claim was classified after it and c1's sport rows fell out
        labels = result.answer_tables["classifies"]["topic"]
        assert labels.column("c").to_pylist() == [0, 1, 2]
        assert labels.column("topic").to_pylist() == ["science", "sport", "science"]
