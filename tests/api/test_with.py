"""WITH: common table expressions whose results later AI operators read.

Each transcript test's transcript shows the plan, the
questions the model answered, and the rows (helpers/transcript.py). A
question about an extracted text names the text.
"""

import pyarrow as pa
from helpers.sessions import TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig
from quail.catalog import DocumentProvider

CONTRACTS = pa.table({
    "id": [1, 2, 3],
    "text": ["Lease ends on March 3. Tenant may terminate early with notice.",
             "No dates here. Nothing about ending.",
             "Lease ends in 2030. Early termination is forbidden."],
})
# the planted termination clause of each contract; the second has none
CLAUSES = {"1": "Tenant may terminate early with notice", "2": None,
           "3": "Early termination is forbidden"}

EXTRACT_THEN_SCORE = """
WITH clauses AS (
    SELECT c.id,
           AI.EXTRACT(c.text, {'terms': 'What is the termination clause?'}) AS clause
    FROM contracts c
)
SELECT x.id, x.clause,
       AI.SCORE(PROMPT('Can the tenant terminate early? {0}', x.clause.terms))
           AS score
FROM clauses x
"""
READ_TWICE = """
WITH clauses AS (
    SELECT c.id,
           AI.EXTRACT(c.text, STRUCT('What is the termination clause?' AS terms))
               AS clause
    FROM contracts c
),
unused AS (
    SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('never asked: {0}', c.text))
)
SELECT a.id, b.id AS other
FROM clauses a JOIN clauses b
  ON AI.IF(PROMPT('same clause? {0} {1}', a.clause.terms, b.clause.terms))
"""
LIMITED = """
WITH first AS (
    SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('keep {0}', c.text))
    ORDER BY c.id LIMIT 2)
SELECT f.id AS result_id FROM first f WHERE f.id > 0
"""
KEPT_ON_THE_RIGHT = """
WITH kept AS (
    SELECT c.id, c.text FROM contracts c WHERE AI.IF(PROMPT('keep {0}', c.text)))
SELECT a.id, b.id AS other FROM contracts a JOIN kept b
ON AI.IF(PROMPT('same? {0} {1}', a.text, b.text))
"""


def _answer(kind, prompt, rows):
    """The model's answers: planted clauses, and tests of them and of contracts."""
    if kind == "extract":
        return CLAUSES[rows[0]]
    if kind == "score":
        # scores a float32 holds exactly, as the executor reads them
        return 0.875 if "Tenant may" in rows[0] else 0.125
    if kind == "label":
        return "lease" if rows[0] in ("1", "3") else "note"
    if len(rows) == 2:
        # a join: a pair of the same text matches
        return rows[0] == rows[1]
    return rows[0] != "2"


def _transcript(*queries, answer=_answer) -> str:
    """Transcripts of queries, each SQL or (SQL, keyword arguments)."""
    with Transcripts({"contracts": CONTRACTS}, answer) as runs:
        return "\n".join(runs.query(query[0], **query[1])
                         if isinstance(query, tuple) else runs.query(query)
                         for query in queries)


def test_an_extraction_in_a_cte_is_scored_by_its_extracted_text():
    # a NULL clause has a NULL score: the model is not asked about it
    assert_transcript(_transcript(EXTRACT_THEN_SCORE), """\
---  WITH clauses AS (     SELECT c.id,            AI.EXTRACT(c.text, {'terms': 'What is
    the termination clause?'}) AS clause     FROM contracts c ) SELECT x.id, x.clause,
    AI.SCORE(PROMPT('Can the tenant terminate early? {0}', x.clause.terms))
    AS score FROM clauses x  ---
plan
  score = ai_score('Can the tenant terminate early? {0}', get_field(x.column_1,
      'terms'))
  clause = ai_extract(c.text, named_struct('terms', 'What is the termination clause?')),
      returns: [clause]
asked 5 of 5
  'What is the termination clause?': 1 'Tenant may terminate early with notice', 2 NULL,
      3 'Early termination is forbidden'
  'Can the tenant terminate early? {0}': 'Early termination is forbidden' 0.125, 'Tenant
      may terminate early with notice' 0.875
rows 3
  1, {'terms': 'Tenant may terminate early with notice'}, 0.875
  2, {'terms': None}, NULL
  3, {'terms': 'Early termination is forbidden'}, 0.125
""")


def test_a_cte_runs_once_however_many_times_it_is_read():
    # the extraction once, though the join reads it twice; the CTE nothing
    # reads never runs, and the contract with no clause matches nothing
    assert_transcript(_transcript((READ_TWICE, {"dialect": "bq"})), """\
---  WITH clauses AS (     SELECT c.id,            AI.EXTRACT(c.text, STRUCT('What is
    the termination clause?' AS terms))                AS clause     FROM contracts c ),
    unused AS (     SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('never asked: {0}',
    c.text)) ) SELECT a.id, b.id AS other FROM clauses a JOIN clauses b   ON
    AI.IF(PROMPT('same clause? {0} {1}', a.clause.terms, b.clause.terms))  ---
plan
  __quail_answer_1 = ai_filter('same clause? {0} {1}', get_field(a.clause, 'terms'),
      get_field(b.clause, 'terms')), filter: __quail_answer_1, returns: []
  clause = ai_extract(c.text, named_struct('terms', 'What is the termination clause?')),
      returns: [clause]
asked 7 of 10
  'What is the termination clause?': 1 'Tenant may terminate early with notice', 2 NULL,
      3 'Early termination is forbidden'
  'same clause? {0} {1}': 'Early termination is forbidden'+'Early termination is
      forbidden' TRUE, 'Early termination is forbidden'+'Tenant may terminate early with
      notice' FALSE, 'Tenant may terminate early with notice'+'Early termination is
      forbidden' FALSE, 'Tenant may terminate early with notice'+'Tenant may terminate
      early with notice' TRUE
rows 2
  1, 1
  3, 3
""")


def test_a_cte_that_filters_or_classifies_returns_the_rows_its_reader_selects():
    assert_transcript(_transcript(
        (LIMITED, {"dialect": "bq"}),
        # a filtered CTE on the right of a join keeps its filter
        (KEPT_ON_THE_RIGHT, {"dialect": "bq"}),
        # DataFusion names the reader's columns
        ("WITH k AS (SELECT c.id, AI.CLASSIFY(c.text, ['lease', 'note']) AS kind "
         "FROM contracts c) SELECT k.id, k.kind FROM k WHERE k.kind = 'lease'",
         {"columns": True})), """\
---  WITH first AS (     SELECT c.id FROM contracts c WHERE AI.IF(PROMPT('keep {0}',
    c.text))     ORDER BY c.id LIMIT 2) SELECT f.id AS result_id FROM first f WHERE f.id
    > 0  ---
plan
  __quail_answer_1 = ai_filter('keep {0}', c.text), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'keep {0}': 1 TRUE, 2 FALSE, 3 TRUE
rows 2
  1
  3

---  WITH kept AS (     SELECT c.id, c.text FROM contracts c WHERE AI.IF(PROMPT('keep
    {0}', c.text))) SELECT a.id, b.id AS other FROM contracts a JOIN kept b ON
    AI.IF(PROMPT('same? {0} {1}', a.text, b.text))  ---
plan
  __quail_answer_2 = ai_filter('same? {0} {1}', a.text, b.text), filter:
      __quail_answer_2, returns: []
  __quail_answer_1 = ai_filter('keep {0}', c.text), filter: __quail_answer_1, returns:
      []
asked 9 of 9
  'keep {0}': 1 TRUE, 2 FALSE, 3 TRUE
  'same? {0} {1}': 1+1 TRUE, 1+2 FALSE, 1+3 FALSE, 3+1 FALSE, 3+2 FALSE, 3+3 TRUE
rows 2
  1, 1
  3, 3

--- WITH k AS (SELECT c.id, AI.CLASSIFY(c.text, ['lease', 'note']) AS kind FROM
    contracts c) SELECT k.id, k.kind FROM k WHERE k.kind = 'lease' ---
plan
  kind = ai_classify(c.text, make_array('lease', 'note')), filter: kind = 'lease'
asked 3 of 3
  '?': 1 'lease', 2 'note', 3 'lease'
columns k.id int64, k.kind string
rows 2
  1, 'lease'
  3, 'lease'
""")


def test_a_cte_join_can_reuse_an_alias_from_the_ctes_scope():
    sql = ("WITH kept AS (SELECT c.id, c.text FROM contracts c "
           "WHERE AI.IF(PROMPT('keep {0}', c.text))) "
           "SELECT k.id AS contract_id, c.text AS contract_text "
           "FROM kept k JOIN contracts c ON k.id = c.id")
    query = (sql, {"columns": True})
    assert_transcript(_transcript(query, answer=lambda *_: True)
                      + "\n" + _transcript(query, answer=lambda *_: False), """\
--- WITH kept AS (SELECT c.id, c.text FROM contracts c WHERE AI.IF(PROMPT('keep {0}',
    c.text))) SELECT k.id AS contract_id, c.text AS contract_text FROM kept k JOIN
    contracts c ON k.id = c.id ---
plan
  __quail_answer_1 = ai_filter('keep {0}', c.text), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'keep {0}': 1 TRUE, 2 TRUE, 3 TRUE
columns contract_id int64, contract_text string
rows 3
  1, 'Lease ends on March 3. Tenant may terminate early with notice.'
  2, 'No dates here. Nothing about ending.'
  3, 'Lease ends in 2030. Early termination is forbidden.'

--- WITH kept AS (SELECT c.id, c.text FROM contracts c WHERE AI.IF(PROMPT('keep {0}',
    c.text))) SELECT k.id AS contract_id, c.text AS contract_text FROM kept k JOIN
    contracts c ON k.id = c.id ---
plan
  __quail_answer_1 = ai_filter('keep {0}', c.text), filter: __quail_answer_1, returns:
      []
asked 3 of 3
  'keep {0}': 1 FALSE, 2 FALSE, 3 FALSE
columns contract_id int64, contract_text string
rows 0
""")


def test_a_cte_is_a_model_region_of_its_own_when_an_ai_operator_reads_it():
    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=TOKENIZER) as session:
        session.register("contracts",
                         DocumentProvider.from_table(CONTRACTS, id_col="id"))
        regions = {sql: session.sql(sql, dialect="bq").num_regions
                   for sql in (READ_TWICE, LIMITED, KEPT_ON_THE_RIGHT)}
        regions[EXTRACT_THEN_SCORE] = session.sql(EXTRACT_THEN_SCORE).num_regions
    # the CTE nothing reads has no region
    assert regions == {EXTRACT_THEN_SCORE: 2, READ_TWICE: 2, LIMITED: 1,
                       KEPT_ON_THE_RIGHT: 1}
