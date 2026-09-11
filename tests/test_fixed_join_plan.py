"""Fixed join execution, filter retention, and answer reconstruction."""

import itertools

import pyarrow as pa
import pytest
from fakes import same_page
from test_quail_backend import graph_state

import quail
from quail.backends.quail.graph import execute_single_graph, filter_result
from quail.bench import quailb
from quail.builder import col, prompt
from quail.execution import PhysicalResponse
from quail.physical import AiFilter, AiJoin, Barrier, Scan, decode_graph
from quail.planner.plan import EngineConfig
from quail.runtime.execute import execute_query
from quail.runtime.runner import NodeMetrics, NodeResult, SurvivorStream
from quail.runtime.session import RefusalError
from quail_b import prompts
from quail_b.queries import FILTER_SELECTIVITY_ESTIMATES


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


class FixedFeverAnswers:
    def __init__(self, state, capacity, empty):
        self.state = state
        self.capacity = capacity
        self.empty = empty
        self.filters = []

    def _answers(self, document_ids):
        """Answers keyed by local position, and the passing global ids."""
        answers = {local: [document < 2 and not self.empty]
                   for local, document in enumerate(document_ids)}
        live = [document for local, document in enumerate(document_ids)
                if all(answers[local])]
        return answers, live

    def execute(self, node, inputs):
        if isinstance(node, AiFilter):
            self.filters.append((node.alias, inputs["retain_survivors"]))
            if node.pin_survivors:
                # the consuming join runs the chain and fills the holder
                stream = SurvivorStream(node, inputs["document_ids"])
                return NodeResult(
                    {f"ids:{node.alias}": stream,
                     f"filter_answers:{node.alias}": {}},
                    finalize=lambda: filter_result(
                        node, stream.holder["answers"],
                        stream.holder["tokens"], stream.document_ids))
            document_ids = list(inputs["document_ids"])
            answers, live = self._answers(document_ids)
            if inputs["retain_survivors"]:
                for document in live[:self.capacity]:
                    self.state["arena"].retain((node.alias, document), 16)
            return NodeResult({
                f"ids:{node.alias}": live,
                f"filter_answers:{node.alias}": {
                    document_ids[local]: row for local, row in answers.items()
                },
            })

        outputs = {}
        stream = inputs.get("anchor_stream")
        if stream is None:
            anchor_ids = inputs["anchor_ids"]
        else:
            # the anchor's chain runs inside the join; like the real
            # driver, fill the holder and append each streamed anchor's
            # key and prefix
            answers, anchor_ids = self._answers(list(stream["document_ids"]))
            stream["holder"].update(answers=answers, tokens=0)
            keys = [(node.anchor, document) for document in anchor_ids]
            if inputs.get("anchor_batch") is not None:
                # like the real driver: per-batch functions run on each
                # batch before admission and may drop survivors
                keys = inputs["anchor_batch"](keys)
            anchor_ids = [key[1] for key in keys]
            for key in keys:
                inputs["anchor_keys"].append(key)
                inputs["prefixes"].append([])
        live = set(range(len(anchor_ids)))
        all_answers = []
        lists_for = inputs.get("anchor_partners")
        for stage_index, stage in enumerate(node.stages):
            aliases = (stage.anchor, *stage.partners)
            claim = next(alias for alias in aliases if alias.startswith("c"))
            evidence = next(alias for alias in aliases if alias.startswith("e"))
            passing = ({(1, 0), (2, 0), (0, 2)} if stage.written_pos == 1
                       else {(0, 0), (1, 1), (2, 0), (1, 2)})
            partners = inputs["partner_indices"][stage.written_pos]
            # a stage over pairs asks each anchor about its own members
            members = ({} if lists_for and (stage.equalities or stage.pairs_from)
                       else None)
            rows = {}
            for local in sorted(live):
                mine = (range(len(partners)) if members is None
                        else lists_for(anchor_ids[local])[stage_index])
                if members is not None:
                    members[local] = list(mine)
                rows[local] = []
                for member in mine:
                    assignment = dict(zip(
                        aliases, (anchor_ids[local], *partners[member])))
                    rows[local].append((assignment[claim], assignment[evidence])
                                       in passing)
            live = {local for local, row in rows.items() if any(row)}
            all_answers.append(rows)
            outputs[f"join_answers:{stage.written_pos}"] = {
                "rows": rows, "anchor_index": anchor_ids,
                "partner_index": partners, "anchor_partners": members,
                "anchor": stage.anchor,
                "partners": list(stage.partners), "semantics": stage.semantics,
                "selectivity": stage.selectivity, "written_pos": stage.written_pos,
            }
        outputs[f"ids:{node.anchor}"] = [anchor_ids[local] for local in sorted(live)]
        return NodeResult(outputs, NodeMetrics(extension={"answers": all_answers}))


def test_fixed_order_execution_and_backend_planning(monkeypatch):
    with monkeypatch.context() as patch:
        for capacity, empty, estimate in [
        (10, False, 0.001), (1, False, 1.0), (0, False, 0.5), (0, True, 1.0),
    ]:
            patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F11, estimate)
            patch.setitem(FILTER_SELECTIVITY_ESTIMATES, prompts.F13, estimate)
            with quail.Session(EngineConfig(),
                               tokenizer=lambda text: list(text.encode())) as session:
                register_fever(session)
                query = quailb.queries(session)["FEV-9"][1]()

                def execute(request):
                    graph = decode_graph(request.plan["graph"], session.registry.codecs)
                    groups = graph.nodes_by_type(AiJoin.type_name)
                    assert sum(len(group.stages) for group in groups) == 3
                    docs = {node.alias: request.inputs[node.input_id].documents
                            for node in graph.nodes if isinstance(node, Scan)}
                    state = graph_state(None, docs)
                    model = FixedFeverAnswers(state, capacity, empty)
                    state["model_execution"] = model

                    def unexpected_search(*args, **kwargs):
                        raise AssertionError("execution called the join optimizer")

                    with patch.context() as execution_patch:
                        execution_patch.setattr(
                            "quail.planner.joins.search_joins", unexpected_search)
                        report = execute_single_graph(
                            state, request.plan["settings"], graph)
                    # an anchor's chain streams into its join unless the
                    # alias was a partner in an earlier group; then it
                    # finishes first and retains survivors in the pool
                    chains = {node.alias: node for node in graph.nodes
                              if isinstance(node, AiFilter)}
                    partners_before = set()
                    pooled = 0
                    for group in groups:
                        chain = chains[group.anchor]
                        pinned = group.anchor not in partners_before
                        assert chain.pin_survivors is pinned
                        assert chain.keep_kv is not pinned
                        pooled += not pinned
                        partners_before.update(
                            alias for stage in group.stages
                            for alias in stage.partners)
                    assert {alias for alias, keep in model.filters if keep} \
                        == {group.anchor for group in groups
                            if chains[group.anchor].keep_kv}
                    assert not state["arena"].accounting.owned
                    assert report["kv_manager"]["retained_after_filters"] == (
                        0 if empty else min(capacity, 2) * pooled)
                    assert [step["id"] for step in report["executed_join_plan"]
                            if step["type"] == AiJoin.type_name] == [
                                g.node_id for g in groups]
                    return PhysicalResponse(report.pop("_outputs"), report)

                result = execute_query(query, physical_executor=execute).collect()
                assert result.to_pylist() == ([] if empty else [{
                    "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1",
                }])

    for backend in ["stock_vllm", "pipelined_vllm", "pipelined_sglang"]:
        with quail.Session(EngineConfig(backend=backend),
                           tokenizer=lambda text: list(text.encode())) as session:
            register_fever(session)
            plan = quailb.queries(session)["FEV-9"][1]().plan()
            execution = next(node for node in plan.nodes if hasattr(node, "joins"))
            assert execution.filters[-1].alias == execution.joins[0].anchor


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
                    for node in graph.nodes if isinstance(node, Scan)}
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
                function = (worker._child_filters if kind == "filters"
                            else worker._child_joins)
                return [function(state, sub) for state, sub in zip(children, subs)]

            def unexpected_search(*args, **kwargs):
                raise AssertionError("execution called the join optimizer")

            with monkeypatch.context() as execution_patch:
                execution_patch.setattr(worker, "_child_boot", lambda state, sub: None)
                execution_patch.setattr(
                    "quail.planner.joins.search_joins", unexpected_search)
                report = execute_distributed_graph(
                    {**request.plan["settings"], "model": "qwen3-4b-fp8",
                     "docs": docs, "physical_plan": request.plan},
                    graph, 2, round_fn, QWEN3_4B_FP8, H100_SXM,
                    session.registry.runtimes, session.registry,
                )
            assert all(not child["arena"].accounting.owned for child in children)
            return PhysicalResponse(report.pop("_outputs"), report)

        result = execute_query(query, physical_executor=execute).collect()
        assert result.to_pylist() == [{
            "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1",
        }]


def _fever_children(session, docs, count):
    from quail.runtime.runner import ExecutionContext

    children = []
    for _ in range(count):
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
    return children


def fever_executor(session, monkeypatch, gpus, check=None, capacity=1):
    """A physical executor over the FEVER fakes.

    Args:
        session: Its registry decodes the plan and holds the functions.
        monkeypatch: Turns the child boot off on two GPUs.
        gpus: One runs the graph in process; two runs the coordinator.
        check: Called with the request and its decoded graph before the
            run; returns extra state entries (a pair table, columns).
        capacity: Documents the fake model keeps resident.
    """
    from quail.backends.quail import worker
    from quail.backends.quail.distributed import execute_distributed_graph
    from quail.specs import H100_SXM, QWEN3_4B_FP8

    def execute(request):
        graph = decode_graph(request.plan["graph"], session.registry.codecs)
        docs = {node.alias: request.inputs[node.input_id].documents
                for node in graph.nodes if isinstance(node, Scan)}
        extra = check(request, graph) if check else {}
        if gpus == 1:
            state = graph_state(None, docs)
            state["model_execution"] = FixedFeverAnswers(state, capacity, False)
            state.update(extra, functions=session.registry.functions)
            report = execute_single_graph(state, request.plan["settings"], graph)
            assert not state["arena"].accounting.owned
            return PhysicalResponse(report.pop("_outputs"), report)
        children = _fever_children(session, docs, 2)

        def round_fn(kind, subs):
            function = (worker._child_filters if kind == "filters"
                        else worker._child_joins)
            return [function(state, sub) for state, sub in zip(children, subs)]

        monkeypatch.setattr(worker, "_child_boot", lambda state, sub: None)
        report = execute_distributed_graph(
            {**request.plan["settings"], "model": "qwen3-4b-fp8",
             "docs": docs, **extra, "physical_plan": request.plan},
            graph, 2, round_fn, QWEN3_4B_FP8, H100_SXM,
            session.registry.runtimes, session.registry,
        )
        assert all(not child["arena"].accounting.owned for child in children)
        return PhysicalResponse(report.pop("_outputs"), report)

    return execute


def test_fev10_asks_the_model_about_same_page_pairs_only(monkeypatch):
    """FEV-10 on one GPU and on two: the join runs over the pair table."""
    def check(request, graph):
        pairs = request.pair_tables()
        assert pairs[0].to_pydict() == {"c": [0, 1, 2], "e": [0, 0, 2]}
        return {"pairs": pairs}

    for gpus in (1, 2):
        with quail.Session(EngineConfig(gpus=gpus),
                           tokenizer=lambda text: list(text.encode())) as session:
            register_fever(session)
            query = quailb.queries(session)["FEV-10"][1]()
            (stage,) = query.plan().graph.nodes_by_type(AiJoin.type_name)[0].stages
            assert stage.equalities == (("c", "evidence_wiki_url", "e", "id"),)
            assert stage.pair_fraction == 3 / 9
            result = execute_query(
                query, physical_executor=fever_executor(session, monkeypatch, gpus,
                                                        check))
            # the filters keep c0, c1, e0, e1; the pairs on those are
            # (c0, e0) and (c1, e0); only c0 is supported by e0. The
            # cross join would also have asked about (c0, e1) and
            # (c1, e1) and returned (c1, e1)
            answers = result.answer_tables["joins"][0]
            assert sorted(zip(answers.column("c").to_pylist(),
                              answers.column("e").to_pylist())) == [(0, 0), (1, 0)]
            assert result.collect().to_pylist() == [{"c.id": "c0", "e.id": "e0"}]


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
    def check(request, graph):
        assert request.pair_tables() == {}
        columns = request.column_tables()
        assert columns["c"].column("evidence_wiki_url").to_pylist() == [
            "e0", "e0", "e2"]
        return {"columns": columns}

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
            result = execute_query(
                query, physical_executor=fever_executor(session, monkeypatch, gpus,
                                                        check))
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

        def check(request, graph):
            seen.append([node.node_id for node in graph.nodes])
            return {}

        execute = fever_executor(session, monkeypatch, 1, check, capacity=10)
        rows = execute_query(query, physical_executor=execute).collect()
        edited_result = execute_query(
            quailb.queries(session)["FEV-9"][1](), physical_executor=execute,
            plan=edited)
        assert f"barrier:{chain.alias}" in seen[1]
        assert f"barrier:{chain.alias}" not in seen[0]
        assert edited_result.collect().to_pylist() == rows.to_pylist() == [{
            "c1.id": "c0", "e1.id": "e0", "c2.id": "c1", "e2.id": "e1"}]


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
    lengths = {alias: summarize_alias(tokens) for alias, tokens in [
        ("a", [800] * 20), ("b", [100] * 30), ("c", [400] * 10),
    ]}
    filtered = {"a", "c"}
    live = {"a": 10.0, "b": 6.0, "c": 8.0}
    result = search_joins(specs, live, lengths, filtered, 5, 8192,
                          QWEN3_4B_FP8, H100_SXM)
    costs = []
    for order in itertools.permutations(specs):
        for anchors in itertools.product(*(spec["aliases"] for spec in order)):
            work, records = walk(list(zip(order, anchors)), live, lengths,
                                 filtered, 5, QWEN3_4B_FP8, H100_SXM)
            seen = set()
            for record in records:
                if record["anchor"] in seen:
                    assert record["resident"] != "filter"
                seen.add(record["anchor"])
            costs.append(speed_of_light(work, QWEN3_4B_FP8, H100_SXM, 8192).seconds)
    assert speed_of_light(
        result["work"], QWEN3_4B_FP8, H100_SXM, 8192).seconds == min(costs)
