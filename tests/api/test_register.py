"""Session.register takes each kind of table, and the file loaders read files."""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.common.expressions import CompileError

REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3"],
    "body": ["good ending", "slow pacing", "great cast", "weak ending"],
})
QUERY = ("SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('q1: {0}', r.body)) "
         "ORDER BY r.id")


@pytest.fixture()
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as value:
        yield value


def _kept(session) -> list[str]:
    """The ids of the reviews the oracle keeps for ``q1:``."""
    oracle = Oracle(filters={"r": {"q1:": [1, 0, 1, 1]}})
    oracle.catalog = session.catalog
    result = session.sql(QUERY).run(physical_executor=oracle_executor(oracle))
    return result.collect().column(0).to_pylist()


@pytest.mark.parametrize("table", [
    REVIEWS,
    pa.RecordBatchReader.from_batches(REVIEWS.schema, REVIEWS.to_batches()),
    ds.dataset(REVIEWS),
], ids=["table", "reader", "dataset"])
def test_register_takes_an_arrow_object_with_or_without_an_id_column(session, table):
    session.register("reviews", table, id_col="id")
    assert _kept(session) == ["r0", "r2", "r3"]
    session.register("unkeyed", REVIEWS)
    counted = session.sql("SELECT COUNT(*) AS n FROM unkeyed").run()
    assert counted.collect().to_pylist() == [{"n": 4}]


def test_register_refuses_an_object_that_holds_no_table(session):
    with pytest.raises(TypeError, match="got int"):
        session.register("reviews", 4)


def test_register_refuses_an_id_column_the_table_lacks(session):
    with pytest.raises(CompileError, match="'key' not in table schema"):
        session.register("reviews", REVIEWS, id_col="key")


@pytest.mark.parametrize("id_col", ["id", ""], ids=["keyed", "unkeyed"])
@pytest.mark.parametrize("file_format", ["parquet", "json", "csv"])
def test_a_file_loader_registers_files_that_datafusion_reads(session, tmp_path,
                                                             file_format, id_col):
    path = tmp_path / f"reviews.{file_format}"
    if file_format == "parquet":
        pq.write_table(REVIEWS, path)
    elif file_format == "json":
        path.write_text("".join(f'{{"id": "{row["id"]}", "body": "{row["body"]}"}}\n'
                                for row in REVIEWS.to_pylist()))
    else:
        path.write_text("id,body\n" + "".join(f"{row['id']},{row['body']}\n"
                                              for row in REVIEWS.to_pylist()))
    getattr(session, f"register_{file_format}")("reviews", str(path), id_col=id_col)
    assert _kept(session) == ["r0", "r2", "r3"]


def test_a_file_loader_refuses_an_id_column_the_files_lack(session, tmp_path):
    pq.write_table(REVIEWS, tmp_path / "reviews.parquet")
    with pytest.raises(CompileError, match="'key' not in"):
        session.register_parquet("reviews", str(tmp_path), id_col="key")
