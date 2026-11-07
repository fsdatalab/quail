"""A filter of a table's columns runs before the AI operator above it.

σ_q(AI(R)) = AI(σ_q(R)) when q reads only R's columns and the operator
has no fetch. Quail moves such a filter to its table while it finds the
model regions, before DataFusion's optimizer runs, and DataFusion's
push_down_filter moves it further down. Each test runs a query with
push_down_filter on and off and checks that both return the same rows
and that the model reads only the rows that pass the filter.
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.builtins import LOGICAL_RULES, built_in_registry

RULE = "push_down_filter"
REVIEWS = pa.table({
    "id": ["r0", "r1", "r2", "r3", "r4", "r5"],
    "stars": [5, 2, 4, 1, 5, 3],
    "body": ["b0", "b1", "b2", "b3", "b4", "b5"],
})
AI_IF = "AI_FILTER(PROMPT('kind: {0}', r.body))"
# reviews r0, r2, and r4 have more than three stars
STARRED = {0, 2, 4}


def _session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("reviews", quail.DocumentProvider.from_table(REVIEWS, id_col="id"))
    return session


@pytest.fixture
def sessions():
    on = _session(LOGICAL_RULES)
    off = _session([name for name in LOGICAL_RULES if name != RULE])
    yield on, off
    on.close()
    off.close()


def _rows_and_read(session, sql) -> tuple[list, set]:
    """Return the query's rows, sorted, and the reviews the model read."""
    read = set()

    def passes(alias, document, position):
        read.add(document)
        return document in (1, 2, 4, 5)

    def label(spec, ids):
        read.add(ids[0])
        return "a" if ids[0] % 2 == 0 else "b"

    result = finish(session.sql(sql), Oracle(filters=passes, labels=label)).collect()
    return sorted(tuple(row.values()) for row in result.to_pylist()), read


def _region_reads_the_filtered_table(query) -> None:
    lines = [line.strip() for line in query.explain().splitlines()]
    region = lines.index("ModelRegionExec")
    assert lines[region + 1].startswith("FilterExec: stars@"), lines


def test_a_filter_above_an_ai_filter_runs_before_the_model(sessions):
    sql = (f"SELECT d.id FROM (SELECT r.id, r.stars FROM reviews r WHERE {AI_IF}) d "
           "WHERE d.stars > 3")
    for session in sessions:
        _region_reads_the_filtered_table(session.sql(sql))
        rows, read = _rows_and_read(session, sql)
        assert rows == [("r2",), ("r4",)]
        assert read == STARRED


def test_a_filter_above_a_classification_runs_before_the_model(sessions):
    sql = ("SELECT d.id, d.t FROM (SELECT r.id, r.stars, "
           "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) AS t "
           "FROM reviews r) d WHERE d.stars > 3")
    for session in sessions:
        _region_reads_the_filtered_table(session.sql(sql))
        rows, read = _rows_and_read(session, sql)
        assert rows == [("r0", "a"), ("r2", "a"), ("r4", "a")]
        assert read == STARRED
