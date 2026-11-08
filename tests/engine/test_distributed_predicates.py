"""Several GPUs each run a region over their shard as one GPU does."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import TOKENIZER

import quail
from quail.api.config import EngineConfig
from quail.api.execute import execute_query

TEXTS = ["No date", "June", "No date", "August"]


def contracts(gpus):
    session = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm", gpus=gpus),
        tokenizer=TOKENIZER)
    session.register("contracts", quail.DocumentProvider.from_table(
        pa.table({"id": range(len(TEXTS)), "text": TEXTS,
                  "group": ["lease", "sale", "lease", "sale"]}), id_col="id"))
    return session


def run(sql, oracle, gpus, **options):
    with contracts(gpus) as session:
        return execute_query(session.sql(sql, **options),
                             physical_executor=oracle_executor(oracle)).collect()


def dated():
    return Oracle(filters=lambda alias, document, position: document % 2 == 1,
                  answers=[None if text == "No date" else text for text in TEXTS])


@pytest.mark.parametrize("gpus", [2, 4])
def test_a_filter_on_several_gpus_returns_the_rows_of_one_gpu(gpus):
    sql = ("SELECT c.id FROM contracts c "
           "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text))")
    assert run(sql, dated(), gpus).to_pylist() == run(sql, dated(), 1).to_pylist()
    assert run(sql, dated(), gpus).column("c.id").to_pylist() == [1, 3]


def test_an_extraction_after_a_test_receives_the_surviving_documents():
    oracle = dated()
    result = run("SELECT c.id, AI.EXTRACT(c.text, "
                 "{'ends': 'When does it end?', 'starts': 'When does it start?'}) "
                 "AS lease FROM contracts c WHERE lease.ends IS NOT NULL", oracle, 4,
                 order="as_written")
    assert result.column("c.id").to_pylist() == [1, 3]
    assert result.column("lease").to_pylist() == [
        {"ends": TEXTS[i], "starts": TEXTS[i]} for i in (1, 3)]


def test_a_filter_on_several_gpus_asks_each_text_once_and_returns_every_copy():
    oracle = Oracle(filters=lambda alias, document, position: True)
    result = run("SELECT c.id FROM contracts c "
                 "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text))", oracle, 4)
    # the two "No date" contracts are one text, asked on one GPU
    assert sorted(document for _, document in oracle.asked) == [0, 1, 3]
    assert result.column("c.id").to_pylist() == [0, 1, 2, 3]


def test_a_limit_waits_for_the_tests_after_the_filters():
    result = run("SELECT c.id, AI.EXTRACT(c.text, {'ends': 'When does it end?'}) "
                 "AS lease FROM contracts c "
                 "WHERE AI_FILTER(PROMPT('Is {0} relevant?', c.text)) "
                 "AND lease.ends IS NOT NULL LIMIT 3", dated(), 4, order="as_written")
    assert result.column("c.id").to_pylist() == [1, 3]
