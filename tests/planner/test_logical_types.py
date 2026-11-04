"""The Rust logical plan types (crates/quail-planner) read and write every plan."""

import json
import zlib

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, check_table

import quail
from quail.api.config import EngineConfig
from quail.logical import LogicalPlan
from quail.physical.codec import decode_value, encode_value

native = pytest.importorskip("_quail_native")


def token_ids(text):
    """Integer token ids, as a real tokenizer returns them."""
    return [zlib.crc32(word.encode()) % 151_936 for word in text.split()]


def _round_trip(plan: LogicalPlan) -> tuple[dict, dict]:
    text = json.dumps(encode_value(plan))
    return json.loads(text), json.loads(native.read_and_write_logical_plan(text))


def test_every_quail_b_logical_plan_reads_into_the_rust_types_unchanged():
    from quail_b.queries import queries
    from tools.dump_check_queries import check_query_entries

    changed, decoded_differently, planned = [], [], 0
    for entry in check_query_entries(list(queries(include_draft=True))):
        with quail.Session(CONFIG, tokenizer=token_ids) as session:
            for name, columns in entry["tables"].items():
                session.register(name, quail.DocumentProvider.from_table(
                    check_table(columns), id_col="id"))
            plan = session.sql(entry["sql"]).logical
        planned += 1
        written, read_back = _round_trip(plan)
        if read_back != written:
            changed.append(entry["id"])
        if decode_value(LogicalPlan, read_back) != plan:
            decoded_differently.append(entry["id"])
    assert planned > 100
    assert changed == decoded_differently == []


TABLES = {
    "reviews": pa.table({"id": ["r0", "r1", "r2"], "stars": [5, 2, 4],
                         "body": ["good film " * 8, "slow " * 8, "fine " * 8]}),
    "contracts": pa.table({"id": ["c0", "c1"],
                           "text": ["Lease ends 2027. " * 4, "No term. " * 4]}),
}

# Queries whose logical plans hold the node types quail-b's plans do not.
NODE_CASES = {
    "Window": ("qwen3-4b-fp8",
               "SELECT r.id, ROW_NUMBER() OVER (ORDER BY r.stars DESC) AS n "
               "FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"),
    "WithCTE": ("qwen3-4b-fp8",
                "WITH clauses AS (SELECT c.id, AI.EXTRACT(c.text, "
                "{'terms': 'What is the termination clause?'}) AS clause "
                "FROM contracts c) SELECT x.id, AI.SCORE(PROMPT("
                "'Early? {0}', x.clause.terms)) AS score FROM clauses x"),
    "Unnest": ("qwen3.5-2b-sae-bf16",
               "SELECT r.id, f.feature FROM reviews r, UNNEST(AI.SAE_FEATURES("
               "PROMPT('{0} What is this about?', r.body), pooling => 'last')) AS f"),
}


@pytest.mark.parametrize("node_type", NODE_CASES)
def test_a_logical_plan_with_each_node_type_reads_into_the_rust_types(node_type):
    model, text = NODE_CASES[node_type]
    config = EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config, tokenizer=token_ids) as session:
        for name, table in TABLES.items():
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        plan = session.sql(text).logical
    written, read_back = _round_trip(plan)
    assert node_type in json.dumps(written)
    assert read_back == written
    assert decode_value(LogicalPlan, read_back) == plan


def test_a_logical_plan_with_an_unknown_field_is_refused():
    with quail.Session(CONFIG, tokenizer=token_ids) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            TABLES["reviews"], id_col="id"))
        plan = session.sql(
            "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"
        ).logical
    written = encode_value(plan)
    written["root"]["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        native.read_and_write_logical_plan(json.dumps(written))


def test_every_quail_b_logical_plan_is_valid_in_rust():
    from quail_b.queries import queries
    from tools.dump_check_queries import check_query_entries

    refused = {}
    for entry in check_query_entries(list(queries(include_draft=True))):
        with quail.Session(CONFIG, tokenizer=token_ids) as session:
            for name, columns in entry["tables"].items():
                session.register(name, quail.DocumentProvider.from_table(
                    check_table(columns), id_col="id"))
            plan = session.sql(entry["sql"]).logical
        message = native.validate_logical_plan(json.dumps(encode_value(plan)))
        if message is not None:
            refused[entry["id"]] = message
    assert refused == {}
