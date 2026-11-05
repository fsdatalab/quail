"""The Rust plan types (crates/quail-physical) read and write every plan unchanged."""

import json
import zlib
from contextlib import closing

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG, check_table

import quail
from quail.api.config import EngineConfig
from quail.api.stages import StageExecution
from quail.logical import WithCTE
from quail.physical.plan import Refusal

native = pytest.importorskip("_quail_native")


def token_ids(text):
    """Integer token ids, as a real tokenizer returns them."""
    return [zlib.crc32(word.encode()) % 151_936 for word in text.split()]


def _round_trip(plan, session) -> tuple[dict, dict]:
    text = json.dumps(plan.to_envelope(session.registry.codecs))
    return json.loads(text), json.loads(native.read_and_write_plan(text))


def test_every_quail_b_plan_reads_into_the_rust_types_unchanged():
    from quail_b.queries import queries
    from tools.dump_check_queries import check_query_entries

    changed, planned = [], 0
    for entry in check_query_entries(list(queries(include_draft=True))):
        with quail.Session(CONFIG, tokenizer=token_ids) as session:
            for name, columns in entry["tables"].items():
                session.register(name, quail.DocumentProvider.from_table(
                    check_table(columns), id_col="id"))
            plan = session.sql(entry["sql"]).plan()
            if isinstance(plan, Refusal):
                continue
            planned += 1
            written, read_back = _round_trip(plan, session)
            if read_back != written:
                changed.append(entry["id"])
    assert planned > 100
    assert changed == []


TABLES = {
    "reviews": pa.table({"id": ["r0", "r1", "r2"], "stars": [5, 2, 4],
                         "body": ["good film " * 8, "slow " * 8, "fine " * 8]}),
    "aspects": pa.table({"id": ["a0", "a1"], "aspect": ["ending", "pacing"]}),
    "contracts": pa.table({"id": ["c0", "c1"],
                           "text": ["Lease ends 2027. " * 4, "No term. " * 4]}),
}

# Queries whose plans hold the node types quail-b's plans do not.
NODE_CASES = {
    "quail.window": ("qwen3-4b-fp8", 1,
                     "WITH w AS (SELECT r.id, r.body, ROW_NUMBER() OVER "
                     "(ORDER BY r.stars DESC) AS n FROM reviews r "
                     "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))) SELECT w.id FROM w "
                     "WHERE AI_FILTER(PROMPT('q2: {0}', w.body)) AND w.n <= 2"),
    "quail.limit": ("qwen3-4b-fp8", 1,
                    "SELECT r.id FROM reviews r "
                    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) LIMIT 2"),
    "quail.exchange": ("qwen3-4b-fp8", 2,
                       "SELECT r.id, a.id FROM reviews r JOIN aspects a "
                       "ON AI_FILTER(PROMPT('q4: {0} {1}', r.body, a.aspect))"),
    "quail.query_stage": ("qwen3-4b-fp8", 1,
                          "WITH clauses AS (SELECT c.id, AI.EXTRACT(c.text, "
                          "{'terms': 'What is the termination clause?'}) AS clause "
                          "FROM contracts c) SELECT x.id, AI.SCORE(PROMPT("
                          "'Early? {0}', x.clause.terms)) AS score FROM clauses x"),
    "quail.unnest": ("qwen3.5-2b-sae-bf16", 1,
                     "SELECT r.id, f.feature FROM reviews r, UNNEST(AI.SAE_FEATURES("
                     "PROMPT('{0} What is this about?', r.body), pooling => 'last')) "
                     "AS f"),
}


@pytest.mark.parametrize("node_type", NODE_CASES)
def test_a_plan_with_each_node_type_reads_into_the_rust_types_unchanged(node_type):
    model, gpus, text = NODE_CASES[node_type]
    config = EngineConfig(model=model, device="h100-sxm", gpus=gpus)
    with quail.Session(config, tokenizer=token_ids) as session:
        for name, table in TABLES.items():
            session.register(
                name, quail.DocumentProvider.from_table(table, id_col="id"))
        query = session.sql(text)
        plans = [query.plan()]
        if isinstance(query.logical.root, WithCTE):
            # each CTE stage plans its own graph
            with closing(StageExecution(query)) as stages:
                stages.plan()
                plans.extend(stages.plans.values())
        trips = [_round_trip(plan, session) for plan in plans]
    assert node_type in {node["type"] for written, _ in trips
                         for node in written["graph"]["nodes"]}
    for written, read_back in trips:
        assert read_back == written


def test_a_plan_with_an_unknown_field_is_refused():
    with quail.Session(CONFIG, tokenizer=token_ids) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            TABLES["reviews"], id_col="id"))
        plan = session.sql(
            "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))"
        ).plan()
        written = plan.to_envelope(session.registry.codecs)
    written["graph"]["nodes"][0]["attributes"]["surprise"] = 1
    with pytest.raises(ValueError, match="surprise"):
        native.read_and_write_plan(json.dumps(written))
