"""Every QUAIL-B query derives its region's operators from DataFusion's plan."""

import json

import pytest
from helpers.sessions import CONFIG, TOKENIZER, check_table

import quail

native = pytest.importorskip("_quail_native")


def quail_b_queries():
    """Yield each QUAIL-B query's id and its query, drafts included."""
    from helpers.check_queries import check_query_entries

    from quail_b.queries import queries

    for entry in check_query_entries(list(queries(include_draft=True))):
        with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
            for name, columns in entry["tables"].items():
                session.register(name, quail.DocumentProvider.from_table(
                    check_table(columns), id_col="id"))
            yield entry["id"], session.sql(entry["sql"])


def test_every_quail_b_query_derives_operators_that_ask_the_model():
    derived, silent = 0, []
    for query_id, query in quail_b_queries():
        derived += 1
        facts = json.loads(native.region_facts(query.operators))
        if not facts["asks_model"]:
            silent.append(query_id)
        assert "TableScan: " in query.logical, query_id
    assert derived > 100
    assert silent == []
