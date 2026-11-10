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
    on, off = sessions
    _region_reads_the_filtered_table(on.sql(sql))
    rows, read = _rows_and_read(on, sql)
    assert rows == [("r2",), ("r4",)]
    assert read == STARRED
    # without the rule DataFusion tests the stars after the model answers
    assert _rows_and_read(off, sql)[0] == rows


def test_a_filter_above_a_classification_runs_before_the_model(sessions):
    sql = ("SELECT d.id, d.t FROM (SELECT r.id, r.stars, "
           "AI_CLASSIFY(PROMPT('tone: {0}', r.body), ['a', 'b']) AS t "
           "FROM reviews r) d WHERE d.stars > 3")
    on, off = sessions
    _region_reads_the_filtered_table(on.sql(sql))
    rows, read = _rows_and_read(on, sql)
    assert rows == [("r0", "a"), ("r2", "a"), ("r4", "a")]
    assert read == STARRED
    assert _rows_and_read(off, sql)[0] == rows


# R4: an extraction whose answer must contain a literal reads only the
# documents that contain it (push_down_substring_prechecks)
SUBSTRING_RULE = "push_down_substring_prechecks"
LEASES = pa.table({
    "id": ["c0", "c1", "c2", "c3", "c4"],
    "text": ["Ends in 2027.", "Ends in 2026.", "2027 start", "No date.",
             "ENDS 2027"],
})


def _leases_session(rules) -> quail.Session:
    session = quail.Session(CONFIG, tokenizer=TOKENIZER,
                            registry=built_in_registry(logical_rules=rules))
    session.register("leases", quail.DocumentProvider.from_table(LEASES, id_col="id"))
    return session


def _end_dates(session, sql) -> tuple[list, int]:
    """Return the query's ids, sorted, and how many documents the model read."""
    asked = []

    def copy_year(spec, document):
        asked.append(document)
        words = LEASES.column("text")[document].as_py().rstrip(".").split()
        return next((word for word in words if word.isdigit()), None)

    result = finish(session.sql(sql), Oracle(answers=copy_year)).collect()
    return sorted(row["l.id"] for row in result.to_pylist()), len(asked)


@pytest.mark.parametrize("test", ["{} LIKE '%2027%'", "{} ILIKE '20_7'"])
def test_an_extraction_tested_with_like_reads_only_documents_with_its_literal(test):
    call = "AI_EXTRACT(l.text, {'year': 'When does it end?'})"
    year = call + "['year']"
    sql = f"SELECT l.id, {call} AS e FROM leases l WHERE {test.format(year)}"
    on = _leases_session(LOGICAL_RULES)
    off = _leases_session([name for name in LOGICAL_RULES if name != SUBSTRING_RULE])
    found = [_end_dates(session, sql) for session in (on, off)]
    on.close()
    off.close()
    assert found[0][0] == found[1][0] == ["c0", "c2", "c4"]
    # LIKE '%2027%' needs "2027", and ILIKE '20_7' needs "20": the model
    # reads the documents that contain it
    num_read = {"{} LIKE '%2027%'": 3, "{} ILIKE '20_7'": 4}[test]
    assert (found[0][1], found[1][1]) == (num_read, 5)
