"""Window functions: the Window node, its Arrow runtime, and the plan."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER, _run

import quail
from quail.api.config import EngineConfig
from quail.physical.plan import Refusal

STEPS = pa.table({
    "id": ["s0", "s1", "s2", "s3", "s4"],
    "run_id": ["a", "a", "b", "a", "b"],
    "step_index": [2, 1, 1, 3, 2],
    "cost": [1.0, None, 2.0, 4.0, 8.0],
    "text": ["one", "two", "three", "four", "five"],
})
OVER = "OVER (PARTITION BY s.run_id ORDER BY s.step_index)"
TONE_CALL = "AI.CLASSIFY(PROMPT('tone: {0}', s.text), ['good', 'bad'])"
TONE = f"{TONE_CALL} AS tone"
WINDOWS = (f"SELECT s.id, {TONE}, LAG({TONE_CALL}) {OVER} AS previous, "
           f"ROW_NUMBER() {OVER} AS n, SUM(s.cost) {OVER} AS spent FROM steps s ")
SQL = WINDOWS + "ORDER BY s.run_id, s.step_index"
NUMBERED = (f"SELECT s.id, ROW_NUMBER() {OVER} AS n FROM steps s "
            "WHERE AI_FILTER(PROMPT('q: {0}', s.text))")


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("steps", quail.DocumentProvider.from_table(
            STEPS, id_col="id"))
        yield session


def test_windows_run_in_datafusion_on_the_model_regions_rows(session):
    query = session.sql(SQL)
    # the region classifies; DataFusion runs the window and the sort
    assert [node.node_id for node in query.plan().nodes] == [
        "scan:s", "ai-classify:0"]
    explained = query.explain()
    assert "WindowAggExec:" in explained and "SortExec:" in explained
    labels = ["good", "bad", "good", "bad", "good"]
    rows = _run(query, oracle_executor(Oracle(
        labels=lambda spec, ids: labels[ids[0]])))
    # SUM over only NULL costs is NULL, as SQL defines it
    assert rows.to_rows() == [
        ("s1", "bad", None, 1, None), ("s0", "good", "bad", 2, 1.0),
        ("s3", "bad", "good", 3, 5.0), ("s2", "good", None, 1, 2.0),
        ("s4", "good", "good", 2, 10.0)]
    # a window reads every survivor, so a LIMIT above it does not end
    # the model's run early
    limited = session.sql(NUMBERED + " LIMIT 2")
    assert "filter_limit" not in limited.plan().settings
    rows = _run(limited, oracle_executor(Oracle(filters=lambda *_: True)))
    assert rows.count() == 2
    # a regular predicate on the partition column runs in DataFusion,
    # before the model, so the window sees the survivors only
    filtered = session.sql(
        WINDOWS + "WHERE s.run_id = 'a' ORDER BY s.run_id, s.step_index")
    plan = filtered.plan()
    assert plan.nodes[0].predicates == ()
    assert "FilterExec: run_id@1 = a" in filtered.explain()
    rows = _run(filtered, oracle_executor(Oracle(
        labels=lambda spec, ids: "good")))
    assert rows.to_rows() == [("s1", "good", None, 1, None),
                              ("s0", "good", "good", 2, 1.0),
                              ("s3", "good", "good", 3, 5.0)]
    # stock vLLM runs the region, a filter, and DataFusion the window
    stock = quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                       backend="stock_vllm"), tokenizer=TOKENIZER)
    stock.register("steps", session.catalog.get("steps"))
    with stock:
        assert not isinstance(stock.sql(NUMBERED).plan(), Refusal)


def test_a_window_a_later_model_call_reads_runs_in_datafusion_between_regions(session):
    query = session.sql(
        "WITH w AS (SELECT s.id, s.text, "
        "ROW_NUMBER() OVER (ORDER BY s.step_index, s.id) AS n FROM steps s "
        "WHERE AI_FILTER(PROMPT('q: {0}', s.text))) "
        "SELECT w.id, w.n FROM w WHERE AI_FILTER(PROMPT('later: {0}', w.text)) "
        "AND w.n <= 2 ORDER BY w.n")
    # the inner filter's region feeds DataFusion's window, whose rows the
    # outer region reads as the table w
    assert "WindowAggExec:" in query.explain()
    # 'q:' drops "one"; the window numbers the rest by step_index
    rows = _run(query, oracle_executor(Oracle(
        filters=lambda alias, document, position: document != 0 or alias != "s")))
    assert rows.to_rows() == [("s1", 1), ("s2", 2)]
