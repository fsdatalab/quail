"""Session.register takes each kind of table, and the file loaders read files."""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail

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
def test_register_takes_an_arrow_object(session, table):
    session.register("reviews", table, id_col="id")
    assert _kept(session) == ["r0", "r2", "r3"]


@pytest.mark.parametrize("file_format", ["parquet", "json", "csv"])
def test_a_file_loader_registers_files_that_datafusion_reads(session, tmp_path,
                                                             file_format):
    path = tmp_path / f"reviews.{file_format}"
    if file_format == "parquet":
        pq.write_table(REVIEWS, path)
    elif file_format == "json":
        path.write_text("".join(f'{{"id": "{row["id"]}", "body": "{row["body"]}"}}\n'
                                for row in REVIEWS.to_pylist()))
    else:
        path.write_text("id,body\n" + "".join(f"{row['id']},{row['body']}\n"
                                              for row in REVIEWS.to_pylist()))
    getattr(session, f"register_{file_format}")("reviews", str(path), id_col="id")
    assert _kept(session) == ["r0", "r2", "r3"]

