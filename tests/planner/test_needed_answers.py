"""Quail asks the model for every answer a query's rows depend on.

``helpers.needed`` changes each answer of a generated query in turn; an
answer is needed when a change alters the rows. Quail must ask every
needed answer. The answers it asks beyond those are what its rewrites
have yet to skip; ``tools/needed_answers.py`` reports them by query.
"""

import pytest
from helpers.generated import Queries, ai_sql
from helpers.needed import NeededAnswers

pytest.importorskip("_quail_native")

SEED = 0
NUM_QUERIES = 12
# the answers Quail asks over these queries that no change of one answer
# shows a need for; a rewrite that skips more lowers it
UNNEEDED = 41


def test_quail_asks_every_needed_answer_and_no_more_unneeded_ones_than_before():
    runs = NeededAnswers(SEED)
    queries = Queries(SEED)
    unneeded = 0
    try:
        for _ in range(NUM_QUERIES):
            found = runs.asked(ai_sql(queries.query()))
            assert not found.missed, found.sql
            unneeded += len(found.unneeded)
    finally:
        runs.close()
    assert unneeded <= UNNEEDED
    print(f"unneeded answers asked: {unneeded}")
