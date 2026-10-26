"""AI.MAP: the operator surface, its prompt, and its plan."""

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens
from helpers.oracle import Oracle, finish

import quail
from quail.logical import CompileError, mapped_above_joins
from quail.physical import AiMap, decode_graph, encode_graph
from quail.planner.plan import EngineConfig

INSTRUCTION = "Summarize the negative arguments the reviewer makes."
REVIEWS = ["Slow pacing, flat acting.", "A delight from start to end.",
           "Too long, but the score shines."]
SUMMARIES = ["slow and flat", "nothing negative", "too long"]


@pytest.fixture()
def session():
    value = quail.Session(
        EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
        tokenizer=letter_tokens)
    value.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": [0, 1, 2], "text": REVIEWS}), id_col="id"))
    yield value
    value.close()


@pytest.mark.parametrize("sql", [False, True])
def test_a_map_returns_ordered_answers_with_its_token_limit(session, sql):
    query = (session.sql(
        "SELECT r.id, AI.MAP(r.text, 'Summarize.', "
        "max_answer_tokens => 12) AS summary FROM reviews r "
        "ORDER BY summary LIMIT 2") if sql else
        session.docs("reviews").alias("r")
        .ai_map("r.text", "Summarize.", name="summary", max_answer_tokens=12)
        .order_by("summary").limit(2).select("r.id", "summary"))
    graph = decode_graph(encode_graph(query.plan().graph, session.registry.codecs),
                         session.registry.codecs)
    node = next(n for n in graph.nodes if isinstance(n, AiMap))
    assert node.spec.max_answer_tokens == 12
    result = finish(query, Oracle(maps=SUMMARIES))
    assert result.collect()["summary"].to_pylist() == sorted(SUMMARIES)[:2]


def test_maps_follow_filters_and_classifications(session):
    query = (session.docs("reviews").alias("r")
             .ai_filter(quail.prompt("Is {0} negative?", quail.col("r.text")))
             .ai_classify(quail.prompt("Tone of {0}?", quail.col("r.text")),
                          ["positive", "negative"], name="tone")
             .ai_map("r.text", INSTRUCTION, name="summary")
             .ai_map("r.text", "Suggest a change.", name="suggestion")
             .select("r.id", "tone", "summary", "suggestion"))
    oracle = Oracle(filters=lambda alias, document, position: document != 1,
                    labels=["negative", "positive", "negative"],
                    maps=lambda spec, document: SUMMARIES[document])
    rows = finish(query, oracle).collect().sort_by("r.id").to_pydict()
    assert rows == {"r.id": [0, 2], "tone": ["negative", "negative"],
                    "summary": [SUMMARIES[0], SUMMARIES[2]],
                    "suggestion": [SUMMARIES[0], SUMMARIES[2]]}


def test_a_map_of_a_joined_table_is_lifted_above_the_join(session):
    session.register("talks", quail.DocumentProvider.from_table(pa.table({
        "id": [0, 1], "text": ["Budget call.", "Roadmap call."]}), id_col="id"))
    query = (session.docs("reviews").alias("r")
             .ai_join(session.docs("talks").alias("t"),
                      quail.prompt("Does {0} discuss {1}?", quail.col("r.text"),
                                   quail.col("t.text")))
             .ai_map("r.text", INSTRUCTION, name="summary")
             .select("r.id", "t.id", "summary"))
    plan = query.plan()
    assert mapped_above_joins(query.logical.root) in (frozenset(), {"r"})
    oracle = Oracle(joins=lambda node, pair: pair["r"] == pair["t"],
                    maps=SUMMARIES)
    result = finish(query, oracle)
    rows = result.collect().to_pydict()
    assert rows["summary"] == [SUMMARIES[i] for i in rows["r.id"]]
    assert plan.estimated_seconds > 0


def test_a_cte_can_feed_a_map_answer_into_another_map(session):
    query = session.sql("""
        WITH summaries AS (
            SELECT r.id, AI.MAP(r.text, 'Summarize.') AS summary FROM reviews r
        )
        SELECT s.id, AI.MAP(s.summary, 'Shorten.') AS shorter FROM summaries s
    """)
    rows = finish(query, Oracle(maps=SUMMARIES)).collect().to_pydict()
    assert rows["shorter"] == SUMMARIES


@pytest.mark.parametrize("query,error", [
    ("SELECT AI.MAP(r.text, 'Summarize.') FROM reviews r", "AS name"),
    ("SELECT AI.MAP(r.text, '') AS summary FROM reviews r", "instruction"),
    ("SELECT AI.MAP(r.text, 'Summarize.', max_answer_tokens => 0) "
     "AS summary FROM reviews r", "positive integer"),
])
def test_invalid_maps_fail_before_execution(session, query, error):
    with pytest.raises(CompileError, match=error):
        session.sql(query)
