"""Per-node estimates and DAG edits: insert, remove, move."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_fixed_join_plan import FixedFeverAnswers, register_fever
from test_quail_backend import graph_state

import quail
from quail.backends.quail.graph import execute_single_graph
from quail.bench import quailb
from quail.builder import col, docs, prompt
from quail.catalog import Catalog, DocumentProvider
from quail.execution import PhysicalResponse
from quail.physical import AiFilter, AiJoin, Barrier, Foreign, Scan, decode_graph
from quail.planner.decide import explain, plan_query
from quail.planner.plan import EngineConfig, PlanEditError
from quail.runtime.execute import execute_query
from quail.specs import H100_SXM, QWEN3_4B_FP8


def _catalog(tmp_path):
    cat = Catalog()
    for name, columns in (("reviews", ["id", "review"]),
                          ("products", ["asin", "description"])):
        pq.write_table(pa.table({c: ["x"] for c in columns}),
                       str(tmp_path / f"{name}.parquet"))
        cat.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col=columns[0]))
    return cat


def _big_plan(tmp_path):
    logical = (docs(_catalog(tmp_path), "reviews", str.split).alias("r")
               .ai_filter(prompt("negative: {0}", col("r.review")),
                          selectivity=0.5)
               .ai_join(docs(_catalog(tmp_path), "products", str.split)
                        .alias("p"),
                        prompt("about {0} {1}", col("r.review"),
                               col("p.description")), selectivity=0.1)
               .select("r.id", "p.asin"))
    return logical, plan_query(
        logical, model=QWEN3_4B_FP8, device=H100_SXM,
        doc_tokens={"r": [400] * 20000, "p": [20] * 10})


def test_node_ids_estimates_and_the_recompute_column(tmp_path):
    logical, plan = _big_plan(tmp_path)
    assert [node.node_id for node in plan.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "ai_join:r", "project"]
    chain = plan.graph.node("ai_filter:r")
    assert chain.pin_survivors
    seconds = {node_id: entry["seconds"]
               for node_id, entry in plan.estimates.items()
               if "seconds" in entry}
    assert set(seconds) == {"ai_filter:r", "ai_join:r"}
    # each node priced alone: the parts add up to at least the packed
    # whole, and to no more than three times it
    assert plan.estimated_seconds <= sum(seconds.values()) \
        <= 3 * plan.estimated_seconds
    # 10,000 expected survivors of 400 tokens do not fit the retention
    # pool, so releasing the chain's KV would recompute most of them
    recompute = plan.estimates["ai_filter:r"]
    assert recompute["release_recompute_tokens"] > 0.9 * 10000 * 401
    assert recompute["release_recompute_seconds"] > 0
    text = explain(logical, plan)
    assert "estimated_seconds=" in text
    assert "if the KV were released here instead of pinned" in text
    assert "do not add up to the plan estimate" in text

    # a Barrier on the pinned edge turns the pin off; the chain keeps
    # its survivors in the pool and the recompute becomes expected
    edited = plan.insert(
        Barrier(node_id="barrier:r", next_anchor="r", aliases=("r",)),
        between=("ai_filter:r", "ai_join:r"))
    assert [node.node_id for node in edited.nodes] == [
        "scan:r", "scan:p", "ai_filter:r", "barrier:r", "ai_join:r", "project"]
    new_chain = edited.graph.node("ai_filter:r")
    assert not new_chain.pin_survivors and new_chain.keep_kv
    assert new_chain.hold_tokens == 0
    assert edited.graph.node("barrier:r").inputs[0].source.node_id == "ai_filter:r"
    assert [port.source.node_id for port in edited.graph.node("ai_join:r").inputs] == [
        "barrier:r", "scan:p"]
    # unpinned, a survivor holds no frame room, so a few more fit
    assert edited.estimates["ai_filter:r"]["release_recompute_tokens"] == \
        pytest.approx(recompute["release_recompute_tokens"], rel=0.01)
    assert "expected recompute at the join" in explain(logical, edited)
    # the edited plan's total carries the recompute the edit causes
    assert edited.estimated_seconds == pytest.approx(
        plan.estimated_seconds
        + edited.estimates["ai_filter:r"]["release_recompute_seconds"])
    # the input plan is untouched, and remove gives the plan back
    assert plan.graph.node("ai_filter:r").pin_survivors
    assert edited.remove("barrier:r") == plan
    moved = edited.move("barrier:r", between=("scan:r", "ai_filter:r"))
    assert [node.node_id for node in moved.nodes][:4] == [
        "scan:r", "scan:p", "barrier:r", "ai_filter:r"]
    assert moved.graph.node("ai_filter:r").pin_survivors


def test_refused_edits_name_their_rule(tmp_path):
    _, plan = _big_plan(tmp_path)
    barrier = Barrier(node_id="barrier:r", next_anchor="r", aliases=("r",))
    with pytest.raises(PlanEditError, match="exactly one edge"):
        plan.insert(barrier, between=("scan:p", "ai_filter:r"))
    with pytest.raises(PlanEditError, match="no node 'nowhere'"):
        plan.insert(barrier, between=("nowhere", "ai_join:r"))
    with pytest.raises(PlanEditError, match="exactly one output of type"):
        plan.insert(Barrier(node_id="barrier:rp", next_anchor="r",
                            aliases=("r", "p")),
                    between=("ai_filter:r", "ai_join:r"))
    with pytest.raises(PlanEditError, match="already has a node"):
        plan.insert(Barrier(node_id="ai_join:r", next_anchor="r",
                            aliases=("r",)),
                    between=("ai_filter:r", "ai_join:r"))
    with pytest.raises(PlanEditError, match="would change what the query"):
        plan.remove("ai_filter:r")
    with pytest.raises(PlanEditError, match="no node 'barrier:r'"):
        plan.remove("barrier:r")
    # a per-batch Foreign keeps the pin; a barrier Foreign drops it
    per_batch = Foreign(node_id="apply:keep", function="keep", kind="per_batch",
                        ids="drop", aliases=("r",))
    kept = plan.insert(per_batch, between=("ai_filter:r", "ai_join:r"))
    assert kept.graph.node("ai_filter:r").pin_survivors
    dropped = plan.insert(
        Foreign(node_id="apply:keep", function="keep", kind="barrier",
                ids="drop", aliases=("r",)),
        between=("ai_filter:r", "ai_join:r"))
    assert not dropped.graph.node("ai_filter:r").pin_survivors
    assert kept.remove("apply:keep") == plan


def test_an_edited_fev9_plan_executes(monkeypatch):
    with quail.Session(EngineConfig(),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_fever(session)
        query = quailb.queries(session)["FEV-9"][1]()
        plan = query.plan()
        chain = next(node for node in plan.nodes
                     if isinstance(node, AiFilter) and node.pin_survivors)
        join = next(node for node in plan.nodes
                    if isinstance(node, AiJoin) and node.anchor == chain.alias)
        edited = plan.insert(
            Barrier(node_id=f"barrier:{chain.alias}", next_anchor=chain.alias,
                    aliases=(chain.alias,)),
            between=(chain.node_id, join.node_id))
        assert not edited.graph.node(chain.node_id).pin_survivors
        assert edited != plan

        seen = []

        def execute(request):
            graph = decode_graph(request.plan["graph"], session.registry.codecs)
            seen.append([node.node_id for node in graph.nodes])
            docs = {node.alias: request.inputs[node.input_id].documents
                    for node in graph.nodes if isinstance(node, Scan)}
            state = graph_state(None, docs)
            state["model_execution"] = FixedFeverAnswers(state, 10, False)
            report = execute_single_graph(state, request.plan["settings"], graph)
            assert not state["arena"].accounting.owned
            return PhysicalResponse(report.pop("_outputs"), report)

        rows = execute_query(query, physical_executor=execute).collect()
        edited_result = execute_query(
            quailb.queries(session)["FEV-9"][1](), physical_executor=execute,
            plan=edited)
        assert f"barrier:{chain.alias}" in seen[1]
        assert f"barrier:{chain.alias}" not in seen[0]
        assert edited_result.collect().to_pylist() == rows.to_pylist() == [{
            "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1"}]
