"""Fixed join execution, filter retention, and answer reconstruction."""

import itertools

import pyarrow as pa
import pytest

import quail
from quail.backends.quail.graph import execute_single_graph
from quail.bench import quailb
from quail.execution import PhysicalResponse
from quail.physical import AnchoredJoin, DocumentInput, PackedFilter, decode_graph
from quail.planner.plan import EngineConfig
from quail.runtime.local import execute_worker_query
from quail.runtime.runner import NodeMetrics, NodeResult
from test_quail_backend import graph_state


def register_fever(session):
    for name, column, prefix, length in (
        ("claims", "claim", "c", 20), ("evidence", "text", "e", 300),
    ):
        session.register(name, quail.DocumentProvider.from_table(pa.table({
            "id": [f"{prefix}{index}" for index in range(3)],
            column: [f"{index} " + "word " * length for index in range(3)],
        }), id_col="id"))


class FixedFeverAnswers:
    def __init__(self, state, capacity, empty):
        self.state = state
        self.capacity = capacity
        self.empty = empty
        self.filters = []

    def execute(self, node, inputs):
        if isinstance(node, PackedFilter):
            self.filters.append((node.alias, inputs["retain_survivors"]))
            answers = {document: [document < 2 and not self.empty]
                       for document in inputs["document_ids"]}
            live = [document for document, row in answers.items() if all(row)]
            if inputs["retain_survivors"]:
                for document in live[:self.capacity]:
                    self.state["arena"].retain((node.alias, document), 16)
            return NodeResult({f"ids:{node.alias}": live,
                               f"filter_answers:{node.alias}": answers})

        outputs = {}
        anchor_ids = inputs["anchor_ids"]
        live = set(range(len(anchor_ids)))
        all_answers = []
        for stage in node.stages:
            aliases = (stage.anchor, *stage.partners)
            claim = next(alias for alias in aliases if alias.startswith("c"))
            evidence = next(alias for alias in aliases if alias.startswith("e"))
            passing = ({(1, 0), (2, 0), (0, 2)} if stage.written_pos == 1
                       else {(0, 0), (1, 1), (2, 0), (1, 2)})
            partners = inputs["partner_indices"][stage.written_pos]
            rows = {}
            for local in sorted(live):
                rows[local] = []
                for partner in partners:
                    assignment = dict(zip(aliases, (anchor_ids[local], *partner)))
                    rows[local].append((assignment[claim], assignment[evidence])
                                       in passing)
            live = {local for local, row in rows.items() if any(row)}
            all_answers.append(rows)
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor": stage.anchor,
                "partners": list(stage.partners), "semantics": stage.semantics,
                "selectivity": stage.selectivity, "written_pos": stage.written_pos,
            }
        outputs[f"ids:{node.anchor}"] = [anchor_ids[local] for local in sorted(live)]
        return NodeResult(outputs, NodeMetrics(extension={"answers": all_answers}))


@pytest.mark.parametrize("capacity,empty,estimate", [
    (10, False, 0.001), (1, False, 1.0), (0, False, 0.5), (0, True, 1.0),
])
def test_fev9_executes_saved_order_with_actual_survivors(
        monkeypatch, capacity, empty, estimate):
    monkeypatch.setitem(quailb.FILTER_SELECTIVITY_ESTIMATES, quailb.F11, estimate)
    monkeypatch.setitem(quailb.FILTER_SELECTIVITY_ESTIMATES, quailb.F13, estimate)
    with quail.Session(EngineConfig(), tokenizer=lambda text: list(text.encode())) as session:
        register_fever(session)
        query = quailb.queries(session)["FEV-9"][1]()

        def execute(request):
            graph = decode_graph(request.plan["graph"], session.registry.codecs)
            groups = graph.nodes_by_type(AnchoredJoin.type_name)
            assert sum(len(group.stages) for group in groups) == 3
            docs = {node.alias: request.inputs[node.input_id].documents
                    for node in graph.nodes if isinstance(node, DocumentInput)}
            state = graph_state(None, docs)
            model = FixedFeverAnswers(state, capacity, empty)
            state["model_execution"] = model

            def unexpected_search(*args, **kwargs):
                raise AssertionError("execution called the join optimizer")

            with monkeypatch.context() as execution_patch:
                execution_patch.setattr("quail.planner.joins.search_joins", unexpected_search)
                report = execute_single_graph(state, request.plan["settings"], graph)
            assert model.filters[-1] == (groups[0].anchor, True)
            anchors = {group.anchor for group in groups}
            assert {alias for alias, keep in model.filters if keep} == anchors
            assert not state["arena"].accounting.owned
            assert report["kv_manager"]["retained_after_filters"] == (
                0 if empty else min(capacity, 2) * len(anchors))
            assert [step["id"] for step in report["executed_join_plan"]
                    if step["type"] == AnchoredJoin.type_name] == [g.node_id for g in groups]
            return PhysicalResponse(report.pop("_outputs"), report)

        result = execute_worker_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == ([] if empty else [{
            "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1",
        }])


@pytest.mark.parametrize("backend", ["quail", "stock_vllm", "pipelined_vllm", "pipelined_sglang"])
def test_every_backend_filters_its_first_anchor_last(backend):
    with quail.Session(EngineConfig(backend=backend),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_fever(session)
        plan = quailb.queries(session)["FEV-9"][1]().plan()
        if backend == "quail":
            first = plan.graph.nodes_by_type(AnchoredJoin.type_name)[0].anchor
            filters = plan.graph.nodes_by_type(PackedFilter.type_name)
            assert {node.alias for node in filters if node.keep_kv} == {
                group.anchor for group in plan.graph.nodes_by_type(AnchoredJoin.type_name)}
        else:
            execution = next(node for node in plan.nodes if hasattr(node, "joins"))
            first = execution.joins[0].anchor
            filters = execution.filters
        assert filters[-1].alias == first


def test_distributed_fev9_executes_bound_join_nodes(monkeypatch):
    from quail.backends.quail import worker
    from quail.backends.quail.distributed import execute_distributed_graph
    from quail.runtime.runner import ExecutionContext
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    with quail.Session(EngineConfig(gpus=2),
                       tokenizer=lambda text: list(text.encode())) as session:
        register_fever(session)
        query = quailb.queries(session)["FEV-9"][1]()

        def execute(request):
            graph = decode_graph(request.plan["graph"], session.registry.codecs)
            docs = {node.alias: request.inputs[node.input_id].documents
                    for node in graph.nodes if isinstance(node, DocumentInput)}
            children = []
            for _ in range(2):
                state = graph_state(None, docs)
                model = FixedFeverAnswers(state, capacity=1, empty=False)
                state.update(
                    model_execution=model, registry=session.registry,
                    boot={"boot_s": 0, "kind": "warm"}, seen=set(),
                    runtime_context=ExecutionContext(
                        runtimes=session.registry.runtimes, model_execution=model,
                    ),
                )
                children.append(state)

            def round_fn(kind, subs):
                function = worker._child_filters if kind == "filters" else worker._child_joins
                return [function(state, sub) for state, sub in zip(children, subs)]

            def unexpected_search(*args, **kwargs):
                raise AssertionError("execution called the join optimizer")

            with monkeypatch.context() as execution_patch:
                execution_patch.setattr(worker, "_child_boot", lambda state, sub: None)
                execution_patch.setattr("quail.planner.joins.search_joins", unexpected_search)
                report = execute_distributed_graph(
                    {**request.plan["settings"], "model": "qwen3-4b-fp8",
                     "docs": docs, "physical_plan": request.plan},
                    graph, 2, round_fn, QWEN3_4B_FP8, H100_SXM,
                    session.registry.runtimes, session.registry,
                )
            assert all(not child["arena"].accounting.owned for child in children)
            return PhysicalResponse(report.pop("_outputs"), report)

        result = execute_worker_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == [{
            "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1",
        }]


def test_retention_search_matches_enumeration():
    from quail.planner.joins import search_joins, summarize_alias, walk
    from quail.planner.sol import speed_of_light
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    specs = [{"written_pos": index, "aliases": list(aliases),
              "anchor": aliases[0], "anchor_free": True, "semantics": "full",
              "selectivity": selectivity,
              "frame_tokens": {alias: 8 for alias in aliases},
              "label_tokens": {alias: 4 for alias in aliases}, "tail_tokens": 2}
             for index, (aliases, selectivity) in enumerate([
                 (("a", "b"), 0.01), (("b", "c"), 0.1),
             ])]
    lengths = {alias: summarize_alias(tokens).with_resident_fraction(fraction)
               for alias, tokens, fraction in [
                   ("a", [800] * 20, 0.4), ("b", [100] * 30, 1.0),
                   ("c", [400] * 10, 0.7),
               ]}
    live = {"a": 10.0, "b": 6.0, "c": 8.0}
    result = search_joins(specs, live, lengths, {}, 5, 8192, QWEN3_4B_FP8, H100_SXM)
    costs = []
    for order in itertools.permutations(specs):
        for anchors in itertools.product(*(spec["aliases"] for spec in order)):
            work, records = walk(list(zip(order, anchors)), live, lengths, {},
                                 5, QWEN3_4B_FP8, H100_SXM)
            seen = set()
            for record in records:
                if record["anchor"] in seen:
                    assert record["resident"] != "filter"
                seen.add(record["anchor"])
            costs.append(speed_of_light(work, QWEN3_4B_FP8, H100_SXM, 8192).seconds)
    assert speed_of_light(result["work"], QWEN3_4B_FP8, H100_SXM, 8192).seconds == min(costs)
