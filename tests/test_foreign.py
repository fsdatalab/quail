"""The Foreign operator: a user function between two GPU operators."""

import random
from types import SimpleNamespace

import pyarrow as pa
import pytest
from test_fixed_join_plan import FixedFeverAnswers, _fever_children, register_fever
from test_quail_backend import graph_state
from test_streamed_loop import (
    DOC,
    FRAME,
    PARTNER,
    QUESTION,
    FakeModel,
    cpu_arena,
    fake_pack,
    fake_torch,
)

import quail
from quail.backends.quail.graph import execute_single_graph
from quail.builder import col, docs, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.execution import PhysicalResponse
from quail.executor import loop
from quail.executor.attention import JOIN_ATTENTION
from quail.logical import (
    Apply,
    CompileError,
    Join,
    SemanticFilter,
    SemanticJoin,
    join_applies,
)
from quail.physical import (
    AiFilter,
    AiJoin,
    FilterStage,
    Foreign,
    GraphValidationError,
    JoinStage,
    PhysicalGraph,
    PortRef,
    Scan,
    decode_graph,
    validate_streams,
)
from quail.physical.base import input_ports
from quail.planner.plan import EngineConfig, Refusal
from quail.runtime.execute import execute_query
from quail.runtime.session import RefusalError
from quail_b import prompts


def _catalog():
    cat = Catalog()
    cat.register("claims", DocumentProvider.from_table(pa.table({
        "id": ["c0", "c1"], "claim": ["a b", "c d"], "url": ["u", "v"]}),
        id_col="id"))
    cat.register("evidence", DocumentProvider.from_table(pa.table({
        "id": ["u", "v"], "text": ["e f", "g h"]}), id_col="id"))
    return cat


def keep_even(tables):
    (alias,) = tables
    table = tables[alias]
    return [document for document in table.column(alias).to_pylist()
            if document % 2 == 0]


def same_key(tables):
    left, right = tables["r"], tables["p"]
    return left.join(right, keys=["key"], join_type="inner").select(["r", "p"])


def test_builder_places_apply_nodes_in_the_logical_tree():
    cat = _catalog()
    tok = str.split
    plan = (docs(cat, "claims", tok).alias("c")
            .ai_filter(prompt("about a person: {0}", col("c.claim")))
            .apply(keep_even, columns=[col("c.url")])
            .join(docs(cat, "evidence", tok).alias("e"))
            .apply(same_key, columns=[col("c.url"), col("e.id")],
                   kind="barrier")
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")))
            .select("c.id", "e.id"))
    join = plan.root.input
    assert isinstance(join, SemanticJoin)
    (pairs,) = join_applies(join)
    assert (pairs.function, pairs.kind, pairs.ids, pairs.written_pos) == (
        "same_key", "barrier", "pairs", 0)
    assert pairs.aliases == ("c", "e")
    assert isinstance(pairs.input, Join)
    chain = pairs.input.left
    assert isinstance(chain, Apply)
    assert (chain.function, chain.kind, chain.ids, chain.aliases) == (
        "keep_even", "per_batch", "drop", ("c",))
    assert isinstance(chain.input, SemanticFilter)
    assert [str(ref.column) for ref in chain.columns] == ["url"]

    base = docs(cat, "claims", tok).alias("c")
    with pytest.raises(CompileError, match="must work on one table"):
        base.apply(keep_even)
    with pytest.raises(CompileError, match="needs a name for a lambda"):
        base.apply(lambda tables: [], columns=[col("c.url")])
    with pytest.raises(CompileError, match="returning pairs follows join"):
        base.apply(keep_even, columns=[col("c.url")], ids="pairs")
    with pytest.raises(CompileError, match="already used by another"):
        (base.apply(keep_even, columns=[col("c.url")])
         .apply(same_key, columns=[col("c.url")], name="keep_even"))
    with pytest.raises(CompileError, match="ids must be 'pairs'"):
        (docs(cat, "claims", tok).alias("c")
         .join(docs(cat, "evidence", tok).alias("e"))
         .apply(same_key, columns=[col("c.url")], ids="preserve"))


def _register(session):
    session.register("claims", quail.DocumentProvider.from_table(pa.table({
        "id": [f"c{i}" for i in range(6)],
        # claims are the long side, so the planner anchors on them
        "claim": [f"{i} " + "word " * 200 for i in range(6)],
        "url": ["u0", "u1", "u1", "u9", "u2", "u2"],
    }), id_col="id"))
    session.register("evidence", quail.DocumentProvider.from_table(pa.table({
        "id": [f"e{i}" for i in range(3)],
        "text": [f"{i} " + "word " * 10 for i in range(3)],
        "url": ["u1", "u0", "u2"],
    }), id_col="id"))


def _query(session, kind):
    claims = (session.docs("claims").alias("c")
              .ai_filter(prompt("about a person: {0}", col("c.claim")),
                         selectivity=0.5)
              .apply(keep_even, columns=[col("c.url")], kind=kind))
    return (claims.join(session.docs("evidence").alias("e"))
            .ai_filter(prompt("{1} supports {0}", col("c.claim"),
                              col("e.text")), selectivity=0.5)
            .select("c.id", "e.id"))


def test_planner_places_foreign_nodes_and_keeps_or_drops_the_stream():
    with quail.Session(EngineConfig(),
                       tokenizer=lambda text: list(text.encode())) as session:
        _register(session)
        per_batch = _query(session, "per_batch").plan()
        chain = per_batch.graph.node("ai_filter:c")
        foreign = per_batch.graph.node("apply:keep_even")
        join = per_batch.graph.nodes_by_type(AiJoin.type_name)[0]
        assert chain.pin_survivors
        assert isinstance(foreign, Foreign) and foreign.kind == "per_batch"
        assert foreign.inputs[0].source == PortRef("ai_filter:c", "ids:c")
        assert PortRef("apply:keep_even", "ids:c") in {
            port.source for port in join.inputs}
        assert "apply:keep_even" in session.registry.functions or \
            "keep_even" in session.registry.functions
        text = per_batch.graph.explain()
        assert "Foreign: keep_even (per_batch, drop) on c" in text
        # a barrier needs every survivor at once: the chain materializes
        barrier = _query(session, "barrier").plan()
        assert not barrier.graph.node("ai_filter:c").pin_survivors
        assert barrier.graph.node("apply:keep_even").kind == "barrier"
        request = _query(session, "barrier")._prepare_physical()
        assert request.column_tables()["c"].column_names == ["c", "url"]
        assert request.column_tables()["c"].column("url").to_pylist()[:2] == [
            "u0", "u1"]

    with quail.Session(EngineConfig(gpus=2),
                       tokenizer=lambda text: list(text.encode())) as session:
        _register(session)
        plan = _query(session, "per_batch").plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "per_batch_apply_needs_one_gpu"
        assert not isinstance(_query(session, "barrier").plan(), Refusal)
    with quail.Session(EngineConfig(backend="stock_vllm"),
                       tokenizer=lambda text: list(text.encode())) as session:
        _register(session)
        plan = _query(session, "barrier").plan()
        assert isinstance(plan, Refusal)
        assert plan.constraint == "apply_needs_quail_backend"


def _graph(foreign_kind, foreign_ids, pin_survivors):
    scan_r = Scan(node_id="input:r", alias="r", input_id="r")
    scan_p = Scan(node_id="input:p", alias="p", input_id="p")
    chain = AiFilter(
        node_id="filter:r",
        inputs=input_ports((PortRef("input:r", "ids:r"),)),
        alias="r", arena_writes=True, pin_survivors=pin_survivors,
        keep_kv=not pin_survivors, hold_tokens=1 if pin_survivors else 0,
        stages=(FilterStage(0, 1, 0, 0.8, 14),),
        question_token_ids=((QUESTION,),))
    nodes = [scan_r, chain, scan_p]
    anchor_src = PortRef("filter:r", "ids:r")
    join_inputs = []
    pairs_from = ""
    if foreign_ids == "pairs":
        nodes.append(Foreign(
            node_id="apply:same_key",
            inputs=input_ports((anchor_src, PortRef("input:p", "ids:p"))),
            function="same_key", kind=foreign_kind, ids="pairs",
            columns=(("r", "key"), ("p", "key")), aliases=("r", "p"),
            written_pos=0))
        join_inputs.append(PortRef("apply:same_key", "pairs:0"))
        pairs_from = "same_key"
    elif foreign_ids is not None:
        nodes.append(Foreign(
            node_id="apply:keep_even",
            inputs=input_ports((anchor_src,)),
            function="keep_even", kind=foreign_kind, ids=foreign_ids,
            columns=(), aliases=("r",)))
        anchor_src = PortRef("apply:keep_even", "ids:r")
    join = AiJoin(
        node_id="group:0", anchor="r",
        anchor_resident="filter" if pin_survivors else "none",
        inputs=input_ports((anchor_src, PortRef("input:p", "ids:p"),
                            *join_inputs)),
        stages=(JoinStage(
            written_pos=0, exec_idx=0, anchor="r", partners=("p",),
            semantics="full", selectivity=0.5, expected_tuples=1,
            anchor_frame_tokens=1, pair_tail_tokens=0,
            anchor_resident="filter" if pin_survivors else "none",
            tuple_tokens=0, pairs_from=pairs_from,
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
            tail_token_ids=()),))
    nodes.append(join)
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "ids:r"))


def _run(monkeypatch, graph, functions):
    from quail.backends.quail import QuailModelExecution
    from quail.builtins import built_in_registry
    from quail.specs import DEVICES, MODELS

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    rng = random.Random(9)
    n_docs, n_partners = 14, 4
    docs = {
        "r": [[DOC + d] * rng.randrange(10, 40) for d in range(n_docs)],
        "p": [[PARTNER + i] * 20 for i in range(n_partners)],
    }
    filter_truth = [[1 if rng.random() < 0.8 else 0] for _ in range(n_docs)]
    join_truth = {("r", d): [1 if rng.random() < 0.5 else 0
                             for _ in range(n_partners)] for d in range(n_docs)}
    model = FakeModel(filter_truth, join_truth)
    torch = SimpleNamespace(
        inference_mode=fake_torch().inference_mode,
        cuda=SimpleNamespace(
            Event=lambda **kw: SimpleNamespace(record=lambda: None),
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0))
    arena = cpu_arena(64)
    pipeline = SimpleNamespace(attention_mode=JOIN_ATTENTION,
                               forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace())
    execution.bind_loaded_model(model=object(), arena=arena,
                                pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v),
        chunk_tokens=120)
    registry = built_in_registry()
    # document d has key d % 4; partner i has key i
    columns = {
        "r": pa.table({"r": pa.array(range(n_docs), pa.int32()),
                       "key": pa.array([d % 4 for d in range(n_docs)])}),
        "p": pa.table({"p": pa.array(range(n_partners), pa.int32()),
                       "key": pa.array(list(range(n_partners)))}),
    }
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution, "runtimes": registry.runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs, "columns": columns,
        "functions": functions,
    }
    result = execute_single_graph(
        state, {"filter_limit": None, "pre_ids": [], "retention": {}}, graph)
    assert not arena.accounting.owned
    survivors = [d for d, truth in enumerate(filter_truth) if all(truth)]
    return result, survivors, join_truth


def test_foreign_runs_per_batch_on_the_stream_and_once_as_a_barrier(monkeypatch):
    functions = {"keep_even": keep_even, "same_key": same_key}
    # a per-batch drop on the pinned chain: only even survivors reach
    # the join, the filter still reports every survivor, and the
    # function ran once per chunk of survivors
    result, survivors, join_truth = _run(
        monkeypatch, _graph("per_batch", "drop", True), functions)
    assert sorted(result["filters"]["r"]) == list(range(14))
    stage = result["joins"][0]
    kept = [d for d in survivors if d % 2 == 0]
    assert sorted(stage["anchor_index"]) == kept
    foreign = result["node_metrics"]["apply:keep_even"]
    assert foreign["input_rows"] == len(survivors)
    assert foreign["output_rows"] == len(kept)
    assert result["_outputs"][PortRef("apply:keep_even", "ids:r")].column(
        "r").to_pylist() == kept
    assert result["node_metrics"]["group:0"]["kv_hits"] == len(kept)

    # the same function as a barrier over a materialized chain
    result, survivors, _ = _run(
        monkeypatch, _graph("barrier", "drop", False), functions)
    assert sorted(result["joins"][0]["anchor_index"]) == kept

    # pairs from a per-batch function equal pairs from a barrier one,
    # and both equal the key equality: document d pairs with partner
    # d % 4 only
    per_batch, survivors, join_truth = _run(
        monkeypatch, _graph("per_batch", "pairs", True), functions)
    barrier, _, _ = _run(monkeypatch, _graph("barrier", "pairs", False),
                         functions)
    for result in (per_batch, barrier):
        table = result["_outputs"][PortRef("group:0", "join_answers:0")]
        assert sorted(zip(table.column("r").to_pylist(),
                          table.column("p").to_pylist())) == [
            (d, d % 4) for d in survivors]
        assert {(r, p): a for r, p, a in zip(
            table.column("r").to_pylist(), table.column("p").to_pylist(),
            table.column("answer").to_pylist())} == {
            (d, d % 4): bool(join_truth[("r", d)][d % 4]) for d in survivors}
        pairs = result["_outputs"][PortRef("apply:same_key", "pairs:0")]
        assert sorted(zip(pairs.column("r").to_pylist(),
                          pairs.column("p").to_pylist())) == [
            (d, d % 4) for d in survivors]
    assert per_batch["node_metrics"]["apply:same_key"]["output_rows"] == len(
        survivors)
    assert barrier["node_metrics"]["apply:same_key"]["output_rows"] == len(
        survivors)

    # a function never invents an id, and preserve means every id
    def invent(tables):
        return [99]

    def lose_one(tables):
        (alias,) = tables
        return tables[alias].column(alias).to_pylist()[1:]

    def outer(tables):
        (alias,) = tables
        return [None] + tables[alias].column(alias).to_pylist()[1:]

    with pytest.raises(ValueError, match="never invents an id"):
        _run(monkeypatch, _graph("per_batch", "drop", True),
             {"keep_even": invent})
    with pytest.raises(ValueError, match="preserves ids but dropped"):
        _run(monkeypatch, _graph("barrier", "preserve", False),
             {"keep_even": lose_one})
    with pytest.raises(ValueError, match="returned a null id"):
        _run(monkeypatch, _graph("barrier", "drop", False),
             {"keep_even": outer})


def test_stream_validator_refuses_a_barrier_on_a_pinned_edge():
    validate_streams(_graph("per_batch", "drop", True))
    validate_streams(_graph("per_batch", "pairs", True))
    with pytest.raises(GraphValidationError, match="per-batch apply"):
        validate_streams(_graph("barrier", "drop", True))
    with pytest.raises(GraphValidationError, match="per-batch apply"):
        validate_streams(_graph("barrier", "pairs", True))
    # a pinned chain that no join consumes is refused too
    graph = _graph(None, None, True)
    orphan = PhysicalGraph(
        tuple(node for node in graph.nodes if node.node_id != "group:0")
        + (graph.node("group:0").with_inputs(input_ports(
            (PortRef("input:r", "ids:r"), PortRef("input:p", "ids:p")))),),
        graph.root)
    with pytest.raises(GraphValidationError, match="no join anchored"):
        validate_streams(orphan)


def same_page(tables):
    claims, evidence = tables["c"], tables["e"]
    return claims.join(evidence, keys=["evidence_wiki_url"],
                       right_keys=["id"], join_type="inner").select(["c", "e"])


def _fev10_by_apply(session, kind):
    claims = (session.docs("claims").alias("c")
              .ai_filter(prompt(prompts.F11, col("c.claim")),
                         selectivity=0.5))
    evidence = (session.docs("evidence").alias("e")
                .ai_filter(prompt(prompts.F13, col("e.text")),
                           selectivity=0.5))
    return (claims.join(evidence)
            .apply(same_page, columns=[col("c.evidence_wiki_url"),
                                       col("e.id")], kind=kind)
            .ai_filter(prompt(prompts.SUPPORT, col("c.claim"), col("e.text")),
                       selectivity=0.5)
            .select("c.id", "e.id"))


def test_fev10_written_with_apply_matches_the_equality(monkeypatch):
    from quail.backends.quail import worker
    from quail.backends.quail.distributed import execute_distributed_graph
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    for gpus, kind in ((1, "per_batch"), (1, "barrier"), (2, "barrier")):
        with quail.Session(EngineConfig(gpus=gpus),
                           tokenizer=lambda text: list(text.encode())) as session:
            register_fever(session)
            query = _fev10_by_apply(session, kind)
            plan = query.plan()
            (join,) = plan.graph.nodes_by_type(AiJoin.type_name)
            (stage,) = join.stages
            assert stage.pairs_from == "same_page" and not stage.equalities
            # the anchor's chain streams only when the function runs per
            # batch; a barrier needs every survivor first
            chain = next(node for node in plan.nodes
                         if isinstance(node, AiFilter)
                         and node.alias == join.anchor)
            assert chain.pin_survivors is (kind == "per_batch")

            def execute(request, gpus=gpus):
                graph = decode_graph(request.plan["graph"], session.registry.codecs)
                docs = {node.alias: request.inputs[node.input_id].documents
                        for node in graph.nodes if isinstance(node, Scan)}
                assert request.pair_tables() == {}
                columns = request.column_tables()
                assert columns["c"].column("evidence_wiki_url").to_pylist() == [
                    "e0", "e0", "e2"]
                if gpus == 1:
                    state = graph_state(None, docs)
                    state["model_execution"] = FixedFeverAnswers(state, 1, False)
                    state.update(columns=columns,
                                 functions=session.registry.functions)
                    report = execute_single_graph(
                        state, request.plan["settings"], graph)
                    return PhysicalResponse(report.pop("_outputs"), report)
                children = _fever_children(session, docs, 2)

                def round_fn(kind, subs):
                    function = (worker._child_filters if kind == "filters"
                                else worker._child_joins)
                    return [function(state, sub)
                            for state, sub in zip(children, subs)]

                monkeypatch.setattr(worker, "_child_boot", lambda state, sub: None)
                report = execute_distributed_graph(
                    {**request.plan["settings"], "model": "qwen3-4b-fp8",
                     "docs": docs, "columns": columns,
                     "physical_plan": request.plan},
                    graph, 2, round_fn, QWEN3_4B_FP8, H100_SXM,
                    session.registry.runtimes, session.registry,
                )
                return PhysicalResponse(report.pop("_outputs"), report)

            result = execute_query(query, physical_executor=execute)
            answers = result.answer_tables["joins"][0]
            assert sorted(zip(answers.column("c").to_pylist(),
                              answers.column("e").to_pylist())) == [(0, 0), (1, 0)]
            assert result.collect().to_pylist() == [{"c.id": "c0", "e.id": "e0"}]

    with quail.Session(EngineConfig(gpus=2),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_fever(session)
        with pytest.raises(RefusalError):
            execute_query(_fev10_by_apply(session, "per_batch"),
                          physical_executor=lambda request: None)
