"""AI.MAP: the operator surface, its prompt, and its plan."""

import pyarrow as pa
import pytest
from helpers.fakes import letter_tokens
from helpers.oracle import Oracle, finish

import quail
from quail.logical import CompileError, mapped_above_joins
from quail.logical.prompts import MAP_ANSWER_CUE, bind_map_prompt
from quail.physical import AiFilter, AiMap, decode_graph, encode_graph
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


def _summaries(session):
    return (session.docs("reviews").alias("r")
            .ai_map("r.text", INSTRUCTION, name="summary")
            .select("r.id", "summary"))


def test_a_map_returns_one_answer_per_document(session):
    result = finish(_summaries(session), Oracle(maps=SUMMARIES))
    rows = result.collect().sort_by("r.id").to_pydict()
    assert rows["summary"] == SUMMARIES


def test_the_map_prompt_ends_in_the_answer_cue():
    prompt = bind_map_prompt(
        (quail.col("r.text"),), INSTRUCTION)
    assert prompt.kind == "map"
    assert prompt.tail.endswith(MAP_ANSWER_CUE)
    assert INSTRUCTION in prompt.tail
    # the document's KV stays instruction independent
    assert INSTRUCTION not in prompt.preamble


def test_a_map_runs_over_the_filter_survivors(session):
    query = (session.docs("reviews").alias("r")
             .ai_filter(quail.prompt("Is {0} negative?", quail.col("r.text")))
             .ai_map("r.text", INSTRUCTION, name="summary")
             .select("r.id", "summary"))
    oracle = Oracle(filters=lambda alias, document, position: document != 1,
                    maps=SUMMARIES)
    result = finish(query, oracle)
    rows = result.collect().sort_by("r.id").to_pydict()
    assert rows["r.id"] == [0, 2]
    assert rows["summary"] == [SUMMARIES[0], SUMMARIES[2]]
    plan = query.plan()
    graph = decode_graph(encode_graph(plan.graph, session.registry.codecs),
                         session.registry.codecs)
    kinds = [type(node).__name__ for node in graph.topological_nodes()
             if isinstance(node, (AiFilter, AiMap))]
    assert kinds == ["AiFilter", "AiMap"]


def test_three_maps_over_one_table_run_in_order(session):
    query = (session.docs("reviews").alias("r")
             .ai_map("r.text", "Summarize the negatives.", name="negatives")
             .ai_map("r.text", "Summarize the positives.", name="positives")
             .ai_map("r.text", "List ideas to pursue.", name="ideas")
             .select("r.id", "negatives", "positives", "ideas"))
    answers = {"negatives": SUMMARIES,
               "positives": ["none", "all of it", "the score"],
               "ideas": ["recut", "sequel", "shorten"]}
    oracle = Oracle(maps=lambda spec, document: answers[spec.name][document])
    result = finish(query, oracle)
    rows = result.collect().sort_by("r.id").to_pydict()
    for name, expected in answers.items():
        assert rows[name] == expected


def test_a_map_and_a_classification_share_a_query(session):
    query = (session.docs("reviews").alias("r")
             .ai_classify(quail.prompt("Tone of {0}?", quail.col("r.text")),
                          ["positive", "negative"], name="tone")
             .ai_map("r.text", INSTRUCTION, name="summary")
             .select("r.id", "tone", "summary"))
    oracle = Oracle(labels=["negative", "positive", "negative"], maps=SUMMARIES)
    result = finish(query, oracle)
    rows = result.collect().sort_by("r.id").to_pydict()
    assert rows["tone"] == ["negative", "positive", "negative"]
    assert rows["summary"] == SUMMARIES


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


def test_sql_ai_map_selects_the_answer(session):
    query = session.sql(
        "SELECT r.id, AI.MAP(r.text, "
        "'Summarize the negative arguments the reviewer makes.') AS summary "
        "FROM reviews r")
    result = finish(query, Oracle(maps=SUMMARIES))
    rows = result.collect().sort_by("r.id").to_pydict()
    assert rows["summary"] == SUMMARIES


def test_sql_ai_map_needs_an_as_name(session):
    with pytest.raises(CompileError, match="AS name"):
        session.sql("SELECT r.id, AI.MAP(r.text, 'Summarize.') FROM reviews r")


def test_a_map_needs_an_instruction(session):
    with pytest.raises(CompileError, match="instruction"):
        session.docs("reviews").alias("r").ai_map("r.text", "  ", name="summary")


def test_a_map_name_is_claimed_once(session):
    query = (session.docs("reviews").alias("r")
             .ai_map("r.text", INSTRUCTION, name="summary"))
    with pytest.raises(CompileError, match="already used"):
        query.ai_map("r.text", "List the positives.", name="summary")


def test_ordering_by_a_map_answer(session):
    query = (session.docs("reviews").alias("r")
             .ai_map("r.text", INSTRUCTION, name="summary")
             .order_by("summary").limit(2)
             .select("r.id", "summary"))
    result = finish(query, Oracle(maps=SUMMARIES))
    rows = result.collect().to_pydict()
    assert rows["summary"] == sorted(SUMMARIES)[:2]
