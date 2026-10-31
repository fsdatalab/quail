"""Window functions: the Window node, its Arrow runtime, and the plan."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, _run, fake_tok

import quail
from quail.physical import Window
from quail.planner.plan import EngineConfig, Refusal

STEPS = pa.table({
    "id": ["s0", "s1", "s2", "s3", "s4"],
    "run_id": ["a", "a", "b", "a", "b"],
    "step_index": [2, 1, 1, 3, 2],
    "cost": [1.0, None, 2.0, 4.0, 8.0],
    "text": ["one", "two", "three", "four", "five"],
})
OVER = "OVER (PARTITION BY s.run_id ORDER BY s.step_index)"
TONE = "AI.CLASSIFY(PROMPT('tone: {0}', s.text), ['good', 'bad']) AS tone"
WINDOWS = (f"SELECT s.id, {TONE}, LAG(tone) {OVER} AS previous, "
           f"ROW_NUMBER() {OVER} AS n, SUM(s.cost) {OVER} AS spent FROM steps s ")
SQL = WINDOWS + "ORDER BY s.run_id, s.step_index"
NUMBERED = (f"SELECT s.id, ROW_NUMBER() {OVER} AS n FROM steps s "
            "WHERE AI_FILTER(PROMPT('q: {0}', s.text))")


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("steps", quail.DocumentProvider.from_table(
            STEPS, id_col="id"))
        yield session


def test_windows_run_after_the_model_and_before_the_sort(session):
    query = session.sql(SQL)
    plan = query.plan()
    # the ORDER BY repeats the window's partition and order, so the
    # Sort goes and the Window returns its columns
    assert [node.node_id for node in plan.nodes][-2:] == ["project", "window"]
    (window,) = [node for node in plan.nodes if isinstance(node, Window)]
    assert window.columns == ("s.id", "tone", "previous", "n", "spent")
    assert "Window:" in query.explain()
    by_step = session.sql(f"SELECT s.id, {TONE}, ROW_NUMBER() {OVER} AS n "
                          "FROM steps s ORDER BY s.step_index").plan()
    assert [node.node_id for node in by_step.nodes][-3:] == [
        "project", "window", "sort"]
    (window,) = [node for node in by_step.nodes if isinstance(node, Window)]
    # the sort key no window output names passes through to the Sort
    assert window.columns == ("s.id", "tone", "n", "s.step_index")
    labels = ["good", "bad", "good", "bad", "good"]
    rows = _run(query, oracle_executor(Oracle(
        labels=lambda spec, ids: labels[ids[0]], probabilities=True)))
    assert rows.to_rows() == [
        ("s1", "bad", None, 1, 0.0), ("s0", "good", "bad", 2, 1.0),
        ("s3", "bad", "good", 3, 5.0), ("s2", "good", None, 1, 2.0),
        ("s4", "good", "good", 2, 10.0)]
    # a window reads every survivor, so a LIMIT does not end the run early
    limited = session.sql(NUMBERED + " LIMIT 2")
    assert "filter_limit" not in limited.plan().settings
    # an unpartitioned window gives the first k rows in its order the
    # same values with or without the rest, so the LIMIT becomes a
    # top-k before it
    topped = session.sql(
        "SELECT s.id, ROW_NUMBER() OVER (ORDER BY s.step_index DESC) AS n "
        "FROM steps s WHERE AI_FILTER(PROMPT('q: {0}', s.text)) LIMIT 2")
    plan = topped.plan()
    assert [node.node_id for node in plan.nodes][-3:] == ["project", "top", "window"]
    assert plan.nodes[-2].fetch == 2 and "filter_limit" not in plan.settings
    rows = _run(topped, oracle_executor(Oracle(filters=lambda *_: True)))
    assert rows.to_rows() == [("s3", 1), ("s0", 2)]
    # a regular predicate on the partition column runs at the scan,
    # below the model, so the window sees the survivors only
    filtered = session.sql(
        WINDOWS + "WHERE s.run_id = 'a' ORDER BY s.run_id, s.step_index")
    plan = filtered.plan()
    assert [str(p) for p in plan.nodes[0].predicates] == ["s.run_id = 'a'"]
    rows = _run(filtered, oracle_executor(Oracle(
        labels=lambda spec, ids: "good", probabilities=True)))
    assert rows.to_rows() == [("s1", "good", None, 1, 0.0),
                              ("s0", "good", "good", 2, 1.0),
                              ("s3", "good", "good", 3, 5.0)]
    stock = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                       backend="stock_vllm"), tokenizer=fake_tok)
    stock.register("steps", session.catalog.get("steps"))
    with stock:
        refused = stock.sql(NUMBERED).plan()
    assert isinstance(refused, Refusal)
    assert refused.constraint == "sort_needs_quail_backend"
