"""DataFusion parses and executes statements around Quail's model work."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail


@pytest.fixture()
def context():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 1]}})
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": ["r0", "r1", "r2", "r3"],
        "body": ["good ending", "slow pacing", "great cast", "weak ending"],
    }), id_col="id"))
    yield context
    session.close()


def test_default_dialect_accepts_standard_aggregate_filter(context):
    result = context.sql(
        "SELECT COUNT(*) FILTER (WHERE AI.IF(PROMPT('q1: {0}', r.body))) AS n "
        "FROM reviews r")
    assert result.to_pylist() == [{"n": 3}]


def test_standard_aggregate_filter_without_ai(context):
    result = context.sql(
        "SELECT COUNT(*) FILTER (WHERE r.id <> 'r1') AS n FROM reviews r")
    assert result.to_pylist() == [{"n": 3}]


def test_create_table_as_runs_model_input_and_registers_rows(context):
    result = context.sql(
        "CREATE TABLE kept AS SELECT r.id FROM reviews r "
        "WHERE AI.IF(PROMPT('q1: {0}', r.body))")
    assert result.num_rows == 0
    assert context.sql("SELECT id FROM kept ORDER BY id").column(0).to_pylist() == [
        "r0", "r2", "r3"]
    # DataFusion, not Quail's Python catalog, owns the created table.
    assert "kept" not in context.session.catalog.providers
    context.sql("DROP TABLE kept")
    with pytest.raises(RuntimeError, match="kept.*not found"):
        context.sql("SELECT * FROM kept")


def test_planning_create_table_does_not_execute_it(context):
    context.region_plans(
        "CREATE TABLE planned AS SELECT r.id FROM reviews r "
        "WHERE AI.IF(PROMPT('q1: {0}', r.body))")
    with pytest.raises(RuntimeError, match="planned.*not found"):
        context.sql("SELECT * FROM planned")


def test_create_view_and_set_use_the_same_session(context):
    context.sql("CREATE VIEW all_reviews AS SELECT id FROM reviews")
    result = context.sql("SELECT COUNT(*) AS n FROM all_reviews")
    assert result.to_pylist() == [{"n": 4}]
    context.sql("SET datafusion.catalog.information_schema = true")
    context.sql("SET datafusion.execution.target_partitions = 1")
    value = context.sql("SHOW datafusion.execution.target_partitions")
    assert value.column(1).to_pylist() == ["1"]
    context.sql("DROP VIEW all_reviews")
