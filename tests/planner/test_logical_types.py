"""The Rust logical plan types (crates/quail-planner) read and write every plan."""

import json

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, TOKENIZER, check_table

import quail
from quail.api.config import EngineConfig

native = pytest.importorskip("_quail_native")


def quail_b_plans():
    """Yield each quail-b query's id and its logical plan as JSON."""
    from quail_b.queries import queries
    from tools.dump_check_queries import check_query_entries

    for entry in check_query_entries(list(queries(include_draft=True))):
        with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
            for name, columns in entry["tables"].items():
                session.register(name, quail.DocumentProvider.from_table(
                    check_table(columns), id_col="id"))
            yield entry["id"], session.sql(entry["sql"]).logical


def test_every_quail_b_logical_plan_is_valid_and_reads_back_unchanged():
    changed, refused, planned = [], {}, 0
    for query_id, logical in quail_b_plans():
        planned += 1
        if json.loads(native.read_and_write_logical_plan(logical)) != json.loads(
                logical):
            changed.append(query_id)
        message = native.validate_logical_plan(logical)
        if message is not None:
            refused[query_id] = message
    assert planned > 100
    assert changed == []
    assert refused == {}


TABLES = {
    "reviews": pa.table({"id": ["r0", "r1", "r2"], "stars": [5, 2, 4],
                         "body": ["good film " * 8, "slow " * 8, "fine " * 8]}),
    "contracts": pa.table({"id": ["c0", "c1"],
                           "text": ["Lease ends 2027. " * 4, "No term. " * 4]}),
}

# Queries whose logical plans hold the node types quail-b's plans do not.
NODE_CASES = {
    "Unnest": ("qwen3.5-2b-sae-bf16",
               "SELECT r.id, f.feature FROM reviews r, UNNEST(AI.SAE_FEATURES("
               "PROMPT('{0} What is this about?', r.body), pooling => 'last')) AS f"),
}


@pytest.mark.parametrize("node_type", NODE_CASES)
def test_a_logical_plan_with_each_node_type_reads_back_unchanged(node_type):
    model, text = NODE_CASES[node_type]
    config = EngineConfig(model=model, device="h100-sxm")
    with quail.Session(config, tokenizer=TOKENIZER) as session:
        for name, table in TABLES.items():
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        logical = session.sql(text).logical
    assert f".{node_type}\"" in logical
    assert json.loads(native.read_and_write_logical_plan(logical)) == json.loads(
        logical)
    assert native.validate_logical_plan(logical) is None


def test_a_logical_plan_with_an_unknown_field_is_refused():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            TABLES["reviews"], id_col="id"))
        logical = session.sql(
            "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"
        ).logical
    written = json.loads(logical)
    written["root"]["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        native.read_and_write_logical_plan(json.dumps(written))
