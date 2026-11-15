"""Image documents: AI functions over a binary column of PNG page images.

Each transcript test holds its expected transcript: the plan's AI
operators, the questions the model answered, and the rows
(helpers/transcript.py).
"""

import pyarrow as pa
import pytest
from helpers.images import png
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER
from helpers.transcript import Transcripts, assert_transcript

import quail
from quail.api.config import EngineConfig

VISION = EngineConfig(model="qwen3.5-4b-fp8", device="h100-sxm")
# p1 and p3 hold the same image
PAGES = pa.table({
    "id": ["p1", "p2", "p3"],
    "page": [1, 2, 1],
    "image": pa.array([png(64, 96, 10), png(96, 64, 200), png(64, 96, 10)],
                      pa.binary()),
})


def _answer(kind, prompt, rows):
    """The model's answers: every page but p2 is a lease."""
    if kind == "filter":
        return rows[0] != "p2"
    return "invoice" if rows[0] == "p2" else "lease"


def _transcript(*queries, tables=None) -> str:
    with Transcripts(tables or {"pages": PAGES}, _answer, config=VISION) as runs:
        return "\n".join(runs.query(sql) for sql in queries)


def test_a_filter_asks_about_each_distinct_page_image_once():
    assert_transcript(_transcript(
        "SELECT p.id FROM pages p "
        "WHERE AI.IF(PROMPT('Is this page a lease? {0}', p.image))"), """\
--- SELECT p.id FROM pages p WHERE AI.IF(PROMPT('Is this page a lease? {0}', p.image))
    ---
plan
  __quail_answer_1 = ai_filter('Is this page a lease? {0}', p.image), filter:
      __quail_answer_1, returns: []
asked 2 of 3
  'Is this page a lease? {0}': p1/p3 TRUE, p2 FALSE
rows 2
  p1
  p3
""")


def test_a_classification_reads_page_images_that_pass_a_filter():
    assert_transcript(_transcript(
        "SELECT p.id, AI.CLASSIFY(p.image, ['lease', 'invoice']) AS kind "
        "FROM pages p WHERE p.page = 1"), """\
--- SELECT p.id, AI.CLASSIFY(p.image, ['lease', 'invoice']) AS kind FROM pages p WHERE
    p.page = 1 ---
plan
  kind = ai_classify(p.image, make_array('lease', 'invoice'))
asked 1 of 2
  '?': p1/p3 'lease'
rows 2
  p1, 'lease'
  p3, 'lease'
""")


def _context(config, table):
    session = quail.Session(config, tokenizer=TOKENIZER)
    oracle = Oracle()
    oracle.catalog = session.catalog
    context = quail.SessionContext(session, physical_executor=oracle_executor(oracle))
    context.register("pages", quail.DocumentProvider.from_table(table, id_col="id"))
    return session, context


@pytest.mark.parametrize(("image", "message"), [
    (png(60, 96), "multiple of 32 pixels"),
    (b"GIF89a, not a png", "must be a PNG"),
])
def test_an_image_quail_cannot_read_is_refused(image, message):
    table = pa.table({"id": ["p1"], "image": pa.array([image], pa.binary())})
    session, context = _context(VISION, table)
    with session, pytest.raises(RuntimeError, match=message):
        context.sql("SELECT p.id FROM pages p "
                    "WHERE AI.IF(PROMPT('Is this page a lease? {0}', p.image))")


def test_a_model_without_a_vision_tower_refuses_an_image():
    session, context = _context(CONFIG, PAGES)
    with session, pytest.raises(RuntimeError, match="reads no images"):
        context.sql("SELECT p.id FROM pages p "
                    "WHERE AI.IF(PROMPT('Is this page a lease? {0}', p.image))")


def test_an_extraction_from_an_image_is_refused():
    session, context = _context(VISION, PAGES)
    with session, pytest.raises(RuntimeError, match="an image has none"):
        context.sql("SELECT p.id, AI.EXTRACT(p.image, {'date': 'When does it start?'}) "
                    "AS e FROM pages p")


def test_projected_questions_about_a_page_share_one_pipeline():
    session, _ = _context(VISION, PAGES)
    with session:
        query = session.sql(
            "SELECT p.id, AI.IF(PROMPT('Is this page a lease? {0}', p.image)) "
            "AS lease, AI.IF(PROMPT('Is this page signed? {0}', p.image)) AS signed "
            "FROM pages p")
        pipelines = query.regions()[0].plan().graph.pipelines()
    assert len({tuple(node.node_id for node in members)
                for members in pipelines.values()}) == 1


def test_projected_questions_answer_every_page():
    assert_transcript(_transcript(
        "SELECT p.id, AI.IF(PROMPT('Is this page a lease? {0}', p.image)) AS lease, "
        "AI.IF(PROMPT('Is this page signed? {0}', p.image)) AS signed FROM pages p"),
        """\
--- SELECT p.id, AI.IF(PROMPT('Is this page a lease? {0}', p.image)) AS lease,
    AI.IF(PROMPT('Is this page signed? {0}', p.image)) AS signed FROM pages p ---
plan
  signed = ai_filter('Is this page signed? {0}', p.image)
  lease = ai_filter('Is this page a lease? {0}', p.image)
asked 4 of 6
  'Is this page a lease? {0}': p1/p3 TRUE, p2 FALSE
  'Is this page signed? {0}': p1/p3 TRUE, p2 FALSE
rows 3
  p1, TRUE, TRUE
  p2, FALSE, FALSE
  p3, TRUE, TRUE
""")


def test_a_map_reads_a_document_of_page_images_and_text():
    questions = pa.table({
        "id": ["q1", "q2"],
        "parts": pa.array([[png(64, 96, 10), b"How long is the lease?"],
                           [png(64, 96, 10), png(96, 64, 200), b"Who signed it?"]],
                          pa.list_(pa.binary())),
    })
    assert_transcript(_transcript(
        "SELECT q.id, AI.MAP(q.parts, 'Answer.', 16) AS answer FROM questions q",
        tables={"questions": questions}), """\
--- SELECT q.id, AI.MAP(q.parts, 'Answer.', 16) AS answer FROM questions q ---
plan
  answer = ai_map(q.parts, 'Answer.', Int64(16))
asked 2 of 2
  'Answer.': q1 'lease', q2 'lease'
rows 2
  q1, 'lease'
  q2, 'lease'
""")
