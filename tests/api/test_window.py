"""Window functions: the Window node, its Arrow runtime, and the plan."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, _run, fake_tok

import quail
from quail import col, prompt
from quail.execution.runtimes.relational import window_table
from quail.frontend.expressions import count, lag, row_number, sum_
from quail.logical import CompileError, WindowCall
from quail.physical import Window
from quail.physical.base import input_ports
from quail.planner.plan import EngineConfig, Refusal

STEPS = pa.table({
    "id": ["s0", "s1", "s2", "s3", "s4"],
    "run_id": ["a", "a", "b", "a", "b"],
    "step_index": [2, 1, 1, 3, 2],
    "cost": [1.0, None, 2.0, 4.0, 8.0],
    "text": ["one", "two", "three", "four", "five"],
})
OVER = {"partition_by": "s.run_id", "order_by": "s.step_index"}
SQL = (
    "SELECT s.id, AI.CLASSIFY(PROMPT('tone: {0}', s.text), ['good', 'bad']) AS tone, "
    "LAG(tone) OVER (PARTITION BY s.run_id ORDER BY s.step_index) AS previous, "
    "ROW_NUMBER() OVER (PARTITION BY s.run_id ORDER BY s.step_index) AS n, "
    "SUM(s.cost) OVER (PARTITION BY s.run_id ORDER BY s.step_index) AS spent "
    "FROM steps s ORDER BY s.run_id, s.step_index")


def _call(function, argument=None, **fields):
    return WindowCall(function, argument, ("run_id",),
                      (("step_index", False, False),), fields.get("name", "w"))


def test_window_table_computes_each_function_in_the_tables_row_order():
    table = STEPS.drop_columns(["text"])
    node = Window(node_id="window", inputs=input_ports(()), calls=(
        _call("row_number", name="n"), _call("lag", "cost", name="before"),
        _call("lead", "cost", name="after"), _call("sum", "cost", name="spent"),
        _call("count", "cost", name="costed"), _call("count", name="seen"),
    ), columns=("id", "n", "before", "after", "spent", "costed", "seen"))
    assert window_table(table, node).to_pydict() == {
        "id": ["s0", "s1", "s2", "s3", "s4"],
        "n": [2, 1, 1, 3, 2],
        "before": [None, None, None, 1.0, 2.0],
        "after": [4.0, 1.0, 8.0, None, None],
        "spent": [1.0, 0.0, 2.0, 5.0, 10.0],
        "costed": [1, 0, 1, 2, 2],
        "seen": [2, 1, 1, 3, 2],
    }
    # a null partition key is one partition; a descending order reverses
    nulls = pa.table({"run_id": [None, "a", None], "step_index": [1, 1, 2]})
    node = Window(node_id="window", inputs=input_ports(()), calls=(
        WindowCall("row_number", None, ("run_id",), (("step_index", True, False),),
                   "n"),), columns=("n",))
    assert window_table(nulls, node).to_pydict() == {"n": [2, 1, 1]}


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("steps", quail.DocumentProvider.from_table(
            STEPS, id_col="id"))
        yield session


def _query(session):
    return (session.docs("steps").alias("s")
            .ai_classify(prompt("tone: {0}", col("s.text")), ["good", "bad"],
                         name="tone")
            .window(previous=lag("tone").over(**OVER),
                    n=row_number().over(**OVER),
                    spent=sum_("s.cost").over(**OVER))
            .order_by("s.run_id", "s.step_index")
            .select("s.id", "tone", "previous", "n", "spent"))


def test_windows_run_after_the_model_and_before_the_sort(session):
    built = _query(session)
    assert session.sql(SQL).logical == built.logical
    plan = built.plan()
    assert [node.node_id for node in plan.nodes][-3:] == ["project", "window", "sort"]
    (window,) = [node for node in plan.nodes if isinstance(node, Window)]
    # the sort keys no window output names pass through to the Sort
    assert window.columns == ("s.id", "tone", "previous", "n", "spent",
                              "s.run_id", "s.step_index")
    assert "Window:" in built.explain()
    labels = ["good", "bad", "good", "bad", "good"]
    rows = _run(built, oracle_executor(Oracle(
        labels=lambda spec, ids: labels[ids[0]], probabilities=True)))
    assert rows.to_rows() == [
        ("s1", "bad", None, 1, 0.0), ("s0", "good", "bad", 2, 1.0),
        ("s3", "bad", "good", 3, 5.0), ("s2", "good", None, 1, 2.0),
        ("s4", "good", "good", 2, 10.0)]
    # a window reads every survivor, so a LIMIT does not end the run early
    limited = (session.docs("steps").alias("s")
               .ai_filter(prompt("q: {0}", col("s.text")))
               .window(n=row_number().over(**OVER)).limit(2).select("s.id", "n"))
    assert "filter_limit" not in limited.plan().settings
    stock = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                       backend="stock_vllm"), tokenizer=fake_tok)
    stock.register("steps", session.catalog.get("steps"))
    with stock:
        refused = (stock.docs("steps").alias("s")
                   .ai_filter(prompt("q: {0}", col("s.text")))
                   .window(n=row_number().over(**OVER)).select("s.id", "n")).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "sort_needs_quail_backend"


def test_window_refusals(session):
    steps = session.docs("steps").alias("s")
    with pytest.raises(CompileError, match="takes no column"):
        _call("row_number", "cost").validate(("run_id", "step_index", "cost"))
    with pytest.raises(CompileError, match="needs an ORDER BY"):
        WindowCall("lag", "cost", (), (), "w").validate(("cost",))
    with pytest.raises(CompileError, match="not a projected column"):
        _call("lag", "cost").validate(("run_id", "step_index"))
    with pytest.raises(CompileError, match="unsupported window function"):
        _call("rank").validate(("run_id", "step_index"))
    with pytest.raises(CompileError, match="window\\(\\) takes"):
        steps.window(n=count())
    with pytest.raises(CompileError, match="group_by\\(\\) in one query"):
        (steps.ai_classify(prompt("tone: {0}", col("s.text")), ["good", "bad"],
                           name="tone")
         .window(n=row_number().over(**OVER)).group_by("tone").agg(k=count())
         .select("tone", "k"))
    head = ("SELECT s.id, AI.CLASSIFY(PROMPT('tone: {0}', s.text), ['good', 'bad'])"
            " AS tone, ")
    for clause, message in (
            ("LAG(tone, 2) OVER (ORDER BY s.step_index) AS p", "row before or after"),
            ("SUM(s.cost) OVER (ORDER BY s.step_index ROWS BETWEEN 1 PRECEDING "
             "AND CURRENT ROW) AS c", "not supported"),
            ("RANK() OVER (ORDER BY s.step_index) AS r", "not supported"),
            ("ROW_NUMBER() OVER (ORDER BY s.step_index)", "needs an AS name"),
            ("ROW_NUMBER() OVER (ORDER BY s.step_index) + 1 AS n", "direct function"),
    ):
        with pytest.raises(CompileError, match=message):
            session.sql(f"{head}{clause} FROM steps s")
    with pytest.raises(CompileError, match="GROUP BY in one SELECT"):
        session.sql(f"{head}ROW_NUMBER() OVER (ORDER BY s.step_index) AS n, "
                    "COUNT(*) AS k FROM steps s GROUP BY tone")
