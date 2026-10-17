"""Model work placed after joins reads the surviving document IDs."""

import itertools

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens
from helpers.sessions import CONFIG

import quail
from quail.builtins import built_in_registry
from quail.catalog.pairs import columns_key
from quail.execution.metrics import NodeResult
from quail.execution.result import answer_table, extract_answers
from quail.execution.rows import Rows
from quail.execution.runner import ExecutionContext, GenericRunner, compute_subgraph
from quail.execution.runtimes import BarrierRuntime
from quail.logical import classified_above_joins, extracted_above_joins
from quail.physical import AiClassify, AiExtract, AiJoin, Barrier, PortRef
from quail.physical.base import input_ports
from quail.planner.optimize import search_candidates
from quail.planner.optimizer.context import PlanningContext


@pytest.mark.parametrize("shape", ["chain", "cycle", "disconnected"])
def test_join_survivors_do_not_depend_on_answer_relation_order(shape):
    if shape == "disconnected":
        edges = [("a", "b", [(1, 0)]), ("c", "d", [(0, 1)])]
        expected = {"a": [1], "b": [0], "c": [0], "d": [1]}
    else:
        edges = [("a", "b", [(0, 0), (1, 1)]),
                 ("b", "c", [(1, 0)] if shape == "chain" else [(0, 0), (1, 1)])]
        if shape == "cycle":
            edges.append(("a", "c", [(0, 1), (1, 0)]))
        expected = ({"a": [1], "b": [1], "c": [0], "d": [0, 1]}
                    if shape == "chain" else
                    {"a": [], "b": [], "c": [], "d": [0, 1]})
    for ordered in itertools.permutations(edges):
        ports = [PortRef(f"scan:{alias}", f"ids:{alias}") for alias in expected]
        values = [pa.array([0, 1], pa.int32()) for _ in ports]
        for index, (left, right, pairs) in enumerate(ordered):
            ports.append(PortRef(f"join:{index}", f"join_answers:{index}"))
            values.append(answer_table(
                {left: [pair[0] for pair in pairs], right: [pair[1] for pair in pairs]},
                [True] * len(pairs), "join_answers"))
        node = Barrier(node_id="survivors", aliases=tuple(expected),
                       inputs=input_ports(tuple(ports)))
        result = BarrierRuntime().execute(
            node, {f"input:{index}": value for index, value in enumerate(values)}, None)
        assert {alias: result.outputs[f"ids:{alias}"].to_pylist()
                for alias in expected} == expected


@pytest.mark.parametrize("kind", ["classify", "extract"])
@pytest.mark.parametrize("alias", ["r", "p"])
@pytest.mark.parametrize("matched", [False, True])
@pytest.mark.parametrize("callback", [False, True])
def test_deferred_models_read_only_join_survivors(kind, alias, matched, callback):
    with quail.Session(CONFIG, tokenizer=letter_tokens) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [0, 1, 2], "body": ["document " * 20] * 3,
        }), id_col="id"))
        tables = {name: session.docs("docs").alias(name) for name in ("r", "p")}
        seen_callbacks = []

        def preserve(rows):
            seen_callbacks.append(rows[alias].ids[alias].to_pylist())
            return rows[alias]

        def joined():
            return tables["r"].ai_join(tables["p"], quail.prompt(
                "Pair? {0} {1}", quail.col("r.body"), quail.col("p.body")),
                anchor="r", selectivity=1 / 9 if matched else 0.0)

        builder = (joined().apply_table(preserve, columns=[quail.col(f"{alias}.id")])
                   if callback else tables[alias])
        if kind == "classify":
            builder.ai_classify(
                quail.prompt("Which? {0}", quail.col(f"{alias}.body")),
                ["a", "b"], name="label")
            selected = "label"
        else:
            builder.ai_extract(f"{alias}.body", {"value": "What?"}, name="answer")
            selected = "answer"
        query = (builder if callback else joined()).select("r.id", "p.id", selected)
        query.plan()
        context = PlanningContext(
            model=session.model, device=session.device, gpu_count=1,
            document_tokens=query._doc_tokens, backend="quail", tokenizer=letter_tokens)
        candidates = search_candidates(query._logical_input, context, session.registry)
        placed = classified_above_joins if kind == "classify" else extracted_above_joins
        (candidate,) = [candidate for candidate in candidates
                        if alias in placed(candidate.logical_plan.root)]
        assert all(node.anchor_resident == "none" for node in candidate.plan.nodes
                   if isinstance(node, AiJoin))
        if not matched:
            assert all(node.spec.expected_inputs == 0 for node in candidate.plan.nodes
                       if isinstance(node, (AiClassify, AiExtract)))
        seen = []

        class Answers:
            def execute(self, node, inputs, context):
                if isinstance(node, AiJoin):
                    ids = {port.source.port.split(":", 1)[1]:
                           inputs[port.name].to_pylist()
                           for port in node.inputs}
                    pairs = list(itertools.product(ids["r"], ids["p"]))
                    bits = [matched and pair == (1, 2) for pair in pairs]
                    return NodeResult({
                        "ids:r": pa.array([1] if matched else [], pa.int32()),
                        f"join_answers:{node.written_pos}": answer_table(
                            {name: [pair[index] for pair in pairs]
                             for index, name in enumerate(("r", "p"))}, bits,
                            "join_answers"),
                    })
                (ids,) = inputs.values()
                seen.append(ids.to_pylist())
                if isinstance(node, AiClassify):
                    rows = Rows({alias: ids}, pa.table({
                        node.spec.name: pa.array(["a"] * len(ids), pa.string())}))
                else:
                    rows = extract_answers(alias, node.spec.name, node.spec.span_name,
                                           ids, ["document"] * len(ids),
                                           [(0, 8)] * len(ids))
                return NodeResult({"rows": rows, f"ids:{alias}": ids})

        runtimes = dict(built_in_registry().runtimes)
        for node_type in (AiJoin, AiClassify, AiExtract):
            runtimes[node_type.runtime_key] = Answers()
        sources = {"r": range(3), "p": range(3),
                   columns_key(alias): pa.table({"id": range(3)})}
        execution = ExecutionContext(runtimes=runtimes, sources=sources,
                                     functions=session.registry.functions)
        GenericRunner().run(compute_subgraph(candidate.plan.graph), execution)
        assert seen == [([1 if alias == "r" else 2] if matched else [])]
        if callback:
            assert seen_callbacks == seen
