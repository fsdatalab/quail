"""AI.PARSE_DOCUMENT: the text of PDFs held in a binary column.

Each transcript test holds its expected transcript: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.pdfs import pdf
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.context import SessionContext

FILES = pa.table({
    "id": ["f1", "f2", "f3"],
    "year": [2025, 2024, 2025],
    "pdf": pa.array([pdf("Lease from March 2025.", "Rent is $2,400 a month."),
                     pdf("Invoice for two chairs."),
                     pdf("Invoice for one desk.")], pa.binary()),
})


def _answer(kind, prompt, rows):
    """The model's answers: a file is a lease when its text says so.

    A parsed document is a computed column, so the oracle names each row
    by its text.
    """
    lease = "Lease" in rows[0]
    if kind == "filter":
        return lease
    if kind == "extract":
        return "March 2025" if lease else "two chairs"
    return "lease" if lease else "invoice"


def _transcript(*queries, tables=None) -> str:
    with Transcripts(tables or {"files": FILES}, _answer) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_a_parsed_pdf_is_the_text_of_its_pages():
    assert_transcript(_transcript(
        "SELECT f.id, AI.PARSE_DOCUMENT(f.pdf) AS text FROM files f",
        # Snowflake's name for the function
        "SELECT f.id, AI_PARSE_DOCUMENT(f.pdf) AS text FROM files f "
        "WHERE f.year = 2024"),
"""\
--- SELECT f.id, AI.PARSE_DOCUMENT(f.pdf) AS text FROM files f ---
plan
asked 0 of 0
rows 3
  f1, 'Lease from March 2025.\\x0cRent is $2,400 a month.'
  f2, 'Invoice for two chairs.'
  f3, 'Invoice for one desk.'

--- SELECT f.id, AI_PARSE_DOCUMENT(f.pdf) AS text FROM files f WHERE f.year = 2024 ---
plan
asked 0 of 0
rows 1
  f2, 'Invoice for two chairs.'
""")


def test_a_model_reads_a_parsed_pdf():
    assert_transcript(_transcript(
        "SELECT f.id, AI.CLASSIFY(AI.PARSE_DOCUMENT(f.pdf), ['lease', 'invoice']) "
        "AS kind FROM files f",
        "SELECT f.id FROM files f "
        "WHERE AI.IF(PROMPT('Is the file a lease? {0}', AI.PARSE_DOCUMENT(f.pdf)))",
        # the region evaluates the parse in its precondition
        "SELECT f.id FROM files f "
        "WHERE AI.IF(PROMPT('Is the file a lease? {0}', AI.PARSE_DOCUMENT(f.pdf))) "
        "OR AI.PARSE_DOCUMENT(f.pdf) LIKE '%desk%'"),
"""\
--- SELECT f.id, AI.CLASSIFY(AI.PARSE_DOCUMENT(f.pdf), ['lease', 'invoice']) AS kind
    FROM files f ---
plan
  kind = ai_classify(f.__quail_expr_0 AS AI.PARSE_DOCUMENT(f.pdf), make_array('lease',
      'invoice'))
asked 3 of 3
  '?': 'Invoice for one desk.' 'invoice', 'Invoice for two chairs.' 'invoice', 'Lease
      from March 2025.\\x0cRent is $2,400 a month.' 'lease'
rows 3
  f1, 'lease'
  f2, 'invoice'
  f3, 'invoice'

--- SELECT f.id FROM files f WHERE AI.IF(PROMPT('Is the file a lease? {0}',
    AI.PARSE_DOCUMENT(f.pdf))) ---
plan
  __quail_answer_1 = ai_filter('Is the file a lease? {0}', f.__quail_expr_0),
      inline_predicate: __quail_answer_1, returns: []
asked 3 of 3
  'Is the file a lease? {0}': 'Invoice for one desk.' FALSE, 'Invoice for two chairs.'
      FALSE, 'Lease from March 2025.\\x0cRent is $2,400 a month.' TRUE
rows 1
  f1

--- SELECT f.id FROM files f WHERE AI.IF(PROMPT('Is the file a lease? {0}',
    AI.PARSE_DOCUMENT(f.pdf))) OR AI.PARSE_DOCUMENT(f.pdf) LIKE '%desk%' ---
plan
  __quail_answer_1 = ai_filter('Is the file a lease? {0}', f.__quail_expr_0),
      precondition: f.__quail_expr_1, inline_predicate: __quail_answer_1 OR
      ai_parse_document(f.pdf) LIKE '%desk%', returns: []
asked 2 of 3
  'Is the file a lease? {0}': 'Invoice for two chairs.' FALSE, 'Lease from March
      2025.\\x0cRent is $2,400 a month.' TRUE
rows 2
  f1
  f3
""")


def test_an_extraction_copies_its_answer_from_the_parsed_text():
    assert_transcript(_transcript(
        "SELECT f.id, AI.EXTRACT(AI.PARSE_DOCUMENT(f.pdf), "
        "{'start': 'When does the lease start?'}) AS answers "
        "FROM files f WHERE f.year = 2025 AND f.id = 'f1'"), """\
--- SELECT f.id, AI.EXTRACT(AI.PARSE_DOCUMENT(f.pdf), {'start': 'When does the lease
    start?'}) AS answers FROM files f WHERE f.year = 2025 AND f.id = 'f1' ---
plan
  answers = ai_extract(f.__quail_expr_0 AS AI.PARSE_DOCUMENT(f.pdf),
      named_struct('start', 'When does the lease start?')), returns: [answers]
asked 1 of 1
  'When does the lease start?': 'Lease from March 2025.\\x0cRent is $2,400 a month.'
      'March 2025'
rows 1
  f1, {'start': 'March 2025'}
""")


def test_an_unreadable_or_null_pdf_has_null_text():
    pdfs = [pdf("Lease from March 2025."), b"not a pdf", None]
    files = pa.table({"id": ["f1", "f2", "f3"], "year": [2025, 2025, 2025],
                      "pdf": pa.array(pdfs, pa.binary())})
    assert_transcript(_transcript(
        "SELECT f.id, AI.PARSE_DOCUMENT(f.pdf) AS text FROM files f",
        tables={"files": files}), """\
--- SELECT f.id, AI.PARSE_DOCUMENT(f.pdf) AS text FROM files f ---
plan
asked 0 of 0
rows 3
  f1, 'Lease from March 2025.'
  f2, NULL
  f3, NULL
""")


@pytest.fixture()
def context():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        oracle = Oracle()
        oracle.catalog = session.catalog
        context = SessionContext(
            session, physical_executor=oracle_executor(oracle))
        context.register(
            "files", quail.DocumentProvider.from_table(FILES, id_col="id"))
        yield context


def test_a_filter_on_another_column_runs_before_the_parse(context):
    plans = context.sql("EXPLAIN SELECT f.id, AI.PARSE_DOCUMENT(f.pdf) AS text "
                        "FROM files f WHERE f.year = 2025")
    logical = plans.column("plan").to_pylist()[
        plans.column("plan_type").to_pylist().index("logical_plan")]
    projection, filtered = (index for index, line in enumerate(logical.splitlines())
                            if "ai_parse_document" in line or "year = " in line)
    assert projection < filtered, logical


def test_a_pdf_that_crashes_the_parser_has_null_text_and_prints_nothing(context, capfd):
    # pdf-extract panics on a font reference to a missing object
    broken = pdf("Lease from March 2025.").replace(b"/F1 5 0 R", b"/F1 9 0 R")
    context.register("broken", quail.DocumentProvider.from_table(
        pa.table({"id": ["f1"], "pdf": pa.array([broken], pa.binary())}), id_col="id"))
    capfd.readouterr()
    table = context.sql("SELECT AI.PARSE_DOCUMENT(b.pdf) AS text FROM broken b")
    assert table.column("text").to_pylist() == [None]
    assert "panicked" not in capfd.readouterr().err


def test_a_parse_of_text_is_refused(context):
    with pytest.raises(RuntimeError, match="reads a binary column of PDFs, not Utf8"):
        context.sql("SELECT AI.PARSE_DOCUMENT(f.id) FROM files f")
