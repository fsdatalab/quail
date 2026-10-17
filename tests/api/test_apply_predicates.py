"""User functions retain their input rows when AI predicates are reordered."""

import pyarrow as pa
import pytest
from helpers.fakes import graph_state, letter_tokens
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail.api.execute import execute_query
from quail.backends.quail.graph import execute_single_graph
from quail.execution.metrics import NodeResult
from quail.execution.rows import Rows
from quail.execution.types import PhysicalResponse
from quail.physical import AiClassify, AiFilter, Foreign, Scan, decode_graph
from quail.planner.plan import EngineConfig


@pytest.mark.parametrize("order", ["as_written", "by_cost"])
@pytest.mark.parametrize("kind", ["barrier", "per_batch"])
@pytest.mark.parametrize("position", ["before", "between", "after"])
def test_callbacks_keep_their_position_among_ai_filters(order, kind, position):
    seen = []

    def first(rows):
        seen.append(rows["r"].ids["r"].to_pylist())
        return rows["r"].ids["r"].slice(0, 1)

    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [10, 20], "body": ["first", "second"],
        }), id_col="id"))
        builder = session.docs("docs").alias("r")
        for step in {
            "before": ("apply", "q1", "q2"),
            "between": ("q1", "apply", "q2"),
            "after": ("q1", "q2", "apply"),
        }[position]:
            if step == "apply":
                builder.apply(first, columns=[quail.col("r.id")], kind=kind)
            else:
                builder.ai_filter(quail.prompt(f"{step}: {{0}}", quail.col("r.body")))
        query = builder.select("r.id", order=order)
        result = _run(query, make_executor(
            {"r": {"q1:": [1, 1], "q2:": [0, 1]}},
            functions=session.registry.functions))
        assert seen == ([[1]] if position == "after" else [[0, 1]])
        assert result.to_rows() == ([(20,)] if position == "after" else [])


def test_repeated_callback_calls_have_distinct_nodes():
    seen = []

    def first(rows):
        seen.append(rows["r"].ids["r"].to_pylist())
        return rows["r"].ids["r"].slice(0, 1)

    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [10, 20], "body": ["first", "second"],
        }), id_col="id"))
        query = (session.docs("docs").alias("r")
                 .apply_table(first, columns=[quail.col("r.id")])
                 .ai_filter(quail.prompt("q: {0}", quail.col("r.body")))
                 .apply_table(first, columns=[quail.col("r.id")])
                 .select("r.id"))
        result = _run(query, make_executor(
            {"r": {"q:": [1, 1]}}, functions=session.registry.functions))
        assert seen == [[0, 1], [0]]
        assert result.to_rows() == [(10,)]


@pytest.mark.parametrize("semantics", ["full", "exists", "anti"])
def test_empty_join_input_matches_estimated_later_work(semantics):
    from quail.physical import AiFilter, AiJoin

    seen = []

    def preserve(rows):
        seen.append(rows["r"].ids["r"].to_pylist())
        return rows["r"]

    def impossible_pair(*_args):
        pytest.fail("an empty partner input cannot produce a model request")

    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [10, 20], "text": ["first", "second"],
        }), id_col="id"))
        partner = session.docs("docs").alias("p").where(quail.col("p.id") < 0)
        query = (session.docs("docs").alias("r")
                 .ai_join(partner, quail.prompt(
                     "Pair? {0} {1}", quail.col("r.text"), quail.col("p.text")),
                     anchor="r", semantics=semantics)
                 .apply_table(preserve, columns=[quail.col("r.id")])
                 .ai_filter(quail.prompt("Keep? {0}", quail.col("r.text")))
                 .select("r.id"))
        plan = query.plan()
        (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
        (later,) = [node for node in plan.nodes if isinstance(node, AiFilter)]
        assert join.expected_tuples == join.tuple_tokens == 0
        assert later.expected_docs == (2 if semantics == "anti" else 0)
        result = _run(query, make_executor(
            {"r": {"Keep?": [True, True]}}, {("r", "p"): impossible_pair},
            functions=session.registry.functions))
        assert seen == ([[0, 1]] if semantics == "anti" else [[]])
        assert result.to_rows() == ([(10,), (20,)] if semantics == "anti" else [])


@pytest.mark.parametrize("order", ["as_written", "by_cost"])
@pytest.mark.parametrize("alias", ["r", "p"])
@pytest.mark.parametrize("kind", ["barrier", "per_batch"])
@pytest.mark.parametrize("position", ["between", "after"])
@pytest.mark.parametrize("reject_after", [False, True])
def test_callbacks_receive_join_survivors_before_later_operators(
        order, alias, kind, position, reject_after):
    seen, second_pairs = [], []

    def first(rows):
        seen.append(rows[alias].ids[alias].to_pylist())
        return rows[alias].ids[alias].slice(0, 1)

    def second(r, q):
        second_pairs.append((r, q))
        return r == q == 1

    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [10, 20], "text": ["first", "second"],
        }), id_col="id"))
        builder = (session.docs("docs").alias("r")
                   .ai_join(session.docs("docs").alias("p"), quail.prompt(
                       "Pair? {0} {1}", quail.col("r.text"), quail.col("p.text")),
                       anchor=alias, selectivity=1.0))
        for step in (("apply", "filter", "join") if position == "between"
                     else ("join", "apply", "filter")):
            if step == "apply":
                builder.apply(first, columns=[quail.col(f"{alias}.id")], kind=kind)
            elif step == "filter":
                builder.ai_filter(quail.prompt("tail: {0}", quail.col(f"{alias}.text")),
                                  selectivity=0.25)
            else:
                builder.ai_join(session.docs("docs").alias("q"), quail.prompt(
                    "Other? {0} {1}", quail.col("r.text"), quail.col("q.text")),
                    anchor="r", selectivity=1.0)
        query = builder.select("r.id", "p.id", "q.id", order=order)
        if alias == "r":
            from quail.physical import AiJoin

            joins = [node for node in query.plan().nodes if isinstance(node, AiJoin)]
            counts = {node.written_pos: node.expected_tuples for node in joins}
            assert counts == {0: 4, 1: 1 if position == "between" else 4}
            assert joins[0].anchor_resident == "none"
        def first_rule(a, b):
            return a == b == 1
        result = _run(query, make_executor(
            {alias: {"tail:": [True, not reject_after]}},
            {("r", "p"): first_rule, ("p", "r"): first_rule, ("r", "q"): second},
            functions=session.registry.functions))
        assert seen == [[1]]
        assert result.to_rows() == ([] if reject_after else [(20, 20, 20)])
        if position == "between" and reject_after:
            assert second_pairs == []


@pytest.mark.parametrize("order", ["as_written", "by_cost"])
@pytest.mark.parametrize("kind", ["barrier", "per_batch"])
@pytest.mark.parametrize("filters", [False, True])
@pytest.mark.parametrize("position", ["before_classify", "before_test", "after_test"])
def test_apply_keeps_its_position_between_filters_and_classifications(
        order, kind, filters, position):
    seen = []

    def keep_first(tables):
        ids = tables["c"].ids["c"]
        seen.append(ids.to_pylist())
        return ids.slice(0, 1)

    with quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens,
    ) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [0, 1], "text": ["June " * 30, "No dates " * 30],
        }), id_col="id"))
        builder = session.docs("docs").alias("c")
        if filters:
            builder.ai_filter(quail.prompt("Yes? {0}", quail.col("c.text")),
                              selectivity=0.1)
            builder.ai_filter(quail.prompt("Question " * 10 + "{0}",
                                           quail.col("c.text")), selectivity=0.9)
        for step in {
            "before_classify": ("apply", "classify", "test"),
            "before_test": ("classify", "apply", "test"),
            "after_test": ("classify", "test", "apply"),
        }[position]:
            if step == "apply":
                builder.apply(keep_first, columns=[quail.col("c.id")], kind=kind)
            elif step == "classify":
                builder.ai_classify(quail.prompt("Which? {0}", quail.col("c.text")),
                                    ["a", "b"], name="kind")
            else:
                builder.label_in("kind", ["a"], selectivity=0.1)
        query = builder.select("c.id", "kind", order=order)
        plan = query.plan()
        (foreign,) = [node for node in plan.nodes if isinstance(node, Foreign)]
        (classify,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
        if position == "before_classify":
            assert classify.inputs[0].source.node_id == foreign.node_id
            before = plan.graph.node(foreign.inputs[0].source.node_id)
            assert isinstance(before, AiFilter if filters else Scan)

        class Answers:
            def execute(self, node, inputs):
                if isinstance(node, AiFilter):
                    ids = list(map(int, inputs["document_ids"]))
                    return NodeResult({
                        "ids:c": ids,
                        "filter_answers:c": {i: [True] for i in ids},
                    })
                assert isinstance(node, AiClassify)
                (ids,) = inputs["score_inputs"].values()
                labels = ["b" if i == 0 else "a" for i in pa.array(ids).to_pylist()]
                rows = Rows({"c": ids}, pa.table({"kind": labels}))
                return NodeResult({"rows": rows, "ids:c": rows.ids["c"]})

        def execute(request):
            graph = decode_graph(request.plan["graph"], session.registry.codecs)
            docs = {node.alias: request.inputs[node.input_id].documents
                    for node in graph.nodes if isinstance(node, Scan)}
            state = graph_state(Answers(), docs)
            state["columns"] = request.column_tables()
            state["functions"] = session.registry.functions
            report = execute_single_graph(state, {}, graph)
            return PhysicalResponse(report.pop("_outputs"), report)

        result = execute_query(query, physical_executor=execute).collect()
        assert seen == ([[1]] if position == "after_test" else [[0, 1]])
        assert result.to_pylist() == (
            [{"c.id": 1, "kind": "a"}] if position == "after_test" else [])
