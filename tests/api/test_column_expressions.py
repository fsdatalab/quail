"""Column expressions retain their meaning across planning and execution."""

import json

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from helpers.sessions import CONFIG, _run, fake_tok, make_executor

import quail
from quail.frontend.expressions import col, count, prompt, sum_
from quail.logical import RegularPredicate


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=fake_tok) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": ["aa", "bb", "ccc", "d"],
            "body": ["document text " * 20] * 4,
            "text": ["z", "é東京", "abc", None],
            "details": [{"title": "z"}, {"title": "é東京"},
                        {"title": "abc"}, None],
        }), id_col="id"))
        yield session


def filtered(session, alias="r"):
    return (session.docs("docs").alias(alias)
            .ai_filter(prompt("q: {0}", col(f"{alias}.body"))))


def run(query, alias="r"):
    return _run(query, make_executor({alias: {"q:": [True] * 4}}))


@pytest.mark.parametrize("nested", [False, True])
def test_projection_sort_and_group_use_the_same_column_expression(session, nested):
    value = col("r.details").field("title") if nested else col("r.text")
    length = value.length()
    result = run(filtered(session).order_by(length.desc().nulls_last(), "r.id")
                 .select("r.id", value, length))
    assert result.to_rows() == [
        ("bb", "é東京", 3), ("ccc", "abc", 3), ("aa", "z", 1), ("d", None, None)]
    hidden = run(filtered(session).order_by(length, "r.id").select("r.id"))
    assert hidden.to_rows() == [("aa",), ("bb",), ("ccc",), ("d",)]
    grouped = run(filtered(session).group_by(length).agg(n=count())
                  .order_by(length).select(length, "n"))
    assert grouped.to_rows() == [(1, 1), (3, 2), (None, 1)]
    summed = run(filtered(session).agg(total=sum_(length)).select("total"))
    assert summed.to_rows() == [(7,)]


def test_distinct_keeps_transformed_ids_distinct_by_value(session):
    length = col("r.id").length()
    query = filtered(session).distinct().order_by(length).select(length)
    assert run(query).to_rows() == [(1,), (2,), (3,)]


def test_struct_predicate_survives_serialization_and_filters_before_ai(session):
    query = (filtered(session).where(col("r.details").field("title").length() > 1)
             .select("r.id"))
    predicate = query.logical.operators().scans[0].predicates[0]
    restored = RegularPredicate.from_dict(json.loads(json.dumps(predicate.to_dict())))
    assert restored == predicate
    assert hash(restored.column) == hash(predicate.column)
    result = run(query)
    assert result.to_rows() == [("bb",), ("ccc",)]
    assert result.report["evaluated_documents"] == 2


def test_projection_resolves_dotted_aliases_without_splitting_names(session):
    query = session.sql('''
        SELECT "r.source".id FROM docs "r.source"
        WHERE AI_FILTER(PROMPT('q: {0}', "r.source".body))
    ''')
    assert run(query, "r.source").to_rows() == [("aa",), ("bb",), ("ccc",), ("d",)]


@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
def test_sql_and_builder_bind_the_same_scalar_expressions(session, dialect):
    expression = col("r.details").field("title").length()
    built = (filtered(session).where(expression > 1).group_by(expression)
             .agg(n=count(), total=sum_(expression)).order_by(expression)
             .select(expression, "n", "total"))
    sql = session.sql('''
        SELECT LENGTH(r.details.title), COUNT(*) AS n,
               SUM(LENGTH(r.details.title)) AS total
        FROM docs r
        WHERE AI_FILTER(PROMPT('q: {0}', r.body)) AND LENGTH(r.details.title) > 1
        GROUP BY LENGTH(r.details.title) ORDER BY LENGTH(r.details.title) NULLS LAST
    ''', dialect=dialect)
    assert built.logical == sql.logical
    assert run(sql).to_rows() == [(3, 2, 6)]


@pytest.mark.parametrize("encoding", ["dictionary", "null"])
def test_length_uses_values_independently_of_arrow_storage(session, encoding):
    values = (pa.array(["é東京", "a", None, "abc"]).dictionary_encode()
              if encoding == "dictionary" else pa.nulls(4))
    session.register("encoded", quail.DocumentProvider.from_table(pa.table({
        "id": list(range(4)), "body": ["text " * 20] * 4, "text": values,
    }), id_col="id"))
    query = (session.docs("encoded").alias("r")
             .ai_filter(prompt("q: {0}", col("r.body"))))
    length = col("r.text").length()
    expected = [3, 1, None, 3] if encoding == "dictionary" else [None] * 4
    assert run(query.select(length)).to_rows() == [(v,) for v in expected]
    filtered_query = query.where(length > 1).select("r.id")
    assert run(filtered_query).to_rows() == (
        [(0,), (3,)] if encoding == "dictionary" else [])


@pytest.mark.parametrize("dialect", ["builder", "snowflake", "bq"])
def test_join_estimates_and_execution_use_transformed_keys(session, dialect):
    session.register("partners", quail.DocumentProvider.from_table(pa.table({
        "id": [f"p{i}" for i in range(5)], "body": ["partner " * 20] * 5,
        "details": [{"title": t} for t in ("long", "x", "東京", "xyz", None)],
    }), id_col="id"))
    if dialect == "builder":
        query = (filtered(session).join(session.docs("partners").alias("p"),
                                       on=col("r.details").field("title").length()
                                       == col("p.details").field("title").length())
                 .ai_filter(prompt("Pair? {0} {1}", col("r.body"), col("p.body")))
                 .select("r.id", "p.id"))
    else:
        query = session.sql('''
            SELECT r.id, p.id FROM docs r JOIN partners p
            ON LENGTH(p.details.title) = LENGTH(r.details.title)
            AND AI_FILTER(PROMPT('Pair? {0} {1}', r.body, p.body))
            WHERE AI_FILTER(PROMPT('q: {0}', r.body))
        ''', dialect=dialect)
    from quail.physical import HashJoin, decode_graph, encode_graph

    plan = query.plan()
    (join,) = [node for node in plan.nodes if isinstance(node, HashJoin)]
    assert join.pair_fraction == 3 / 20
    encoded = json.loads(json.dumps(encode_graph(plan.graph, session.registry.codecs)))
    assert decode_graph(encoded, session.registry.codecs) == plan.graph
    result = _run(query, make_executor(
        {"r": {"q:": [True] * 4}},
        {("r", "p"): lambda *_: True, ("p", "r"): lambda *_: True}))
    assert sorted(result.to_rows()) == [("aa", "p1"), ("bb", "p3"), ("ccc", "p3")]
    estimate = quail.speed_of_light_estimate(query, lambda *_: True)
    assert estimate.join_pair_evaluations == 3


@pytest.mark.parametrize("kind", ["per_batch", "barrier"])
def test_apply_receives_evaluated_column_expressions(session, kind):
    seen = []

    def longer(rows):
        values = rows["r"]
        seen.extend(zip(values.ids["r"].to_pylist(),
                        values.column("length(details.title)").to_pylist()))
        return values.filter(pc.greater(values.column("length(details.title)"), 1))

    query = (filtered(session)
             .apply(longer, columns=[col("r.details").field("title").length()],
                    kind=kind)
             .ai_filter(prompt("after: {0}", col("r.body")))
             .select("r.id", order="as_written"))
    result = _run(query, make_executor(
        {"r": {"q:": [True] * 4, "after:": [True] * 4}},
        functions={"longer": longer}))
    assert seen == [(0, 1), (1, 3), (2, 3), (3, None)]
    assert result.to_rows() == [("bb",), ("ccc",)]


@pytest.mark.parametrize("dialect", ["snowflake", "bq"])
@pytest.mark.parametrize(("selection", "suffix", "names", "expected"), [
    ("r.id AS doc_id, LENGTH(r.text) AS n", "ORDER BY n NULLS LAST, doc_id",
     ["doc_id", "n"], [("aa", 1), ("bb", 3), ("ccc", 3), ("d", None)]),
    ("r.id AS r", "ORDER BY r.id", ["r"],
     [("aa",), ("bb",), ("ccc",), ("d",)]),
    ("DISTINCT LENGTH(r.id) AS r", "ORDER BY LENGTH(r.id)", ["r"],
     [(1,), (2,), (3,)]),
    ("LENGTH(r.text) AS a, LENGTH(r.text) AS b, COUNT(*) AS n",
     "GROUP BY LENGTH(r.text) ORDER BY LENGTH(r.text) NULLS LAST",
     ["a", "b", "n"], [(1, 1, 1), (3, 3, 2), (None, None, 1)]),
    ('r.text AS "r.id"', "ORDER BY r.id", ["r.id"],
     [("z",), ("é東京",), ("abc",), (None,)]),
    ('r.text AS "r.id", COUNT(r.id) AS n',
     "GROUP BY r.text ORDER BY r.text NULLS LAST", ["r.id", "n"],
     [("abc", 1), ("z", 1), ("é東京", 1), (None, 1)]),
    ('COUNT(r.id) AS n, r.text AS "r.id"',
     "GROUP BY r.text ORDER BY r.text NULLS LAST", ["n", "r.id"],
     [(1, "abc"), (1, "z"), (1, "é東京"), (1, None)]),
    ("LENGTH(r.text) AS a, COUNT(*) AS n",
     "GROUP BY a ORDER BY a NULLS LAST", ["a", "n"],
     [(1, 1), (3, 2), (None, 1)]),
])
def test_sql_output_aliases_preserve_values_and_expression_identity(
    session, dialect, selection, suffix, names, expected,
):
    query = session.sql(f"""
        SELECT {selection} FROM docs r
        WHERE AI_FILTER(PROMPT('q: {{0}}', r.body)) {suffix}
    """, dialect=dialect)
    result = run(query)
    assert result.schema.names == names
    assert result.to_rows() == expected


@pytest.mark.parametrize("frontend", ["builder", "snowflake", "bq"])
def test_ai_inputs_use_nested_text_and_cache_each_field_separately(session, frontend):
    table = pa.table({"id": ["a", "b"], "content": [
        {"body": "alpha beta", "summary": "short"},
        {"body": "gamma delta epsilon", "summary": "two words"}],
    })
    session.register("nested", quail.DocumentProvider.from_table(table, "id"))
    stores = []
    for field, alias in (("body", "r"), ("summary", "r"), ("body", "s")):
        if frontend == "builder":
            query = (session.docs("nested").alias(alias)
                     .ai_filter(prompt("q: {0}", col(f"{alias}.content").field(field)))
                     .select(f"{alias}.id"))
        else:
            query = session.sql(f"""
                SELECT {alias}.id FROM nested {alias}
                WHERE AI_FILTER(PROMPT('q: {{0}}', {alias}.content.{field}))
            """, dialect=frontend)
        result = _run(query, make_executor({alias: {"q:": [True, True]}}))
        assert result.to_rows() == [("a",), ("b",)]
        store = query._token_inputs[alias].tokens
        stores.append(store)
        assert [list(tokens) for tokens in store] == [
            row[field].split() for row in table["content"].to_pylist()]
    assert stores[0] is stores[2]
    assert stores[0] is not stores[1]


@pytest.mark.parametrize("keyed", [False, True])
def test_join_preserves_the_right_querys_nested_ai_input(session, keyed):
    table = pa.table({"id": ["a", "b"], "content": [
        {"body": "alpha beta"}, {"body": "gamma delta epsilon"}],
    })
    session.register("nested", quail.DocumentProvider.from_table(table, "id"))
    left_text = col("r.content").field("body")
    right_text = col("p.content").field("body")
    right = (session.docs("nested").alias("p")
             .ai_filter(prompt("q: {0}", right_text)))
    left = session.docs("nested").alias("r")
    pair = prompt("Pair? {0} {1}", left_text, right_text)
    if keyed:
        joined = left.join(right, on=col("r.id") == col("p.id")).ai_filter(pair)
    else:
        joined = left.ai_join(right, pair)
    query = joined.select("r.id", "p.id")
    result = _run(query, make_executor(
        {"p": {"q:": [True, False]}},
        {("r", "p"): lambda *_: True, ("p", "r"): lambda *_: True}))
    assert sorted(result.to_rows()) == (
        [("a", "a")] if keyed else [("a", "a"), ("b", "a")])
    for alias in ("r", "p"):
        assert [list(tokens) for tokens in query._token_inputs[alias].tokens] == [
            row["body"].split() for row in table["content"].to_pylist()]


def test_nested_extraction_offsets_refer_to_the_selected_text():
    texts = ["café 東京", "a short lease"]

    def encode(text):
        return [ord(char) for char in text]

    encode.token_bytes = lambda: {
        ord(char): char.encode() for text in texts for char in text}
    with quail.Session(CONFIG, tokenizer=encode) as session:
        session.register("docs", quail.DocumentProvider.from_table(pa.table({
            "id": [1, 2], "content": [{"body": text} for text in texts],
        }), "id"))
        source = col("d.content").field("body")
        filtered = (session.docs("docs").alias("d")
                    .ai_filter(prompt("q: {0}", source)).select("d.id"))
        filtered.plan()
        filtered.wait_for_tokens()
        previous = filtered._token_inputs["d"].tokens
        assert not previous.has_offsets
        extracted = (session.docs("docs").alias("d")
                     .ai_extract(source, {"place": "What is the place?"}, name="answer")
                     .select("d.id", "answer"))
        request = extracted._prepare_physical()
        (documents,) = request.inputs.values()
        upgraded = extracted._token_inputs["d"].tokens
        assert upgraded.has_offsets and not previous.has_offsets
        for index, text in enumerate(texts):
            assert list(documents.documents[index]) == encode(text)
            assert list(previous[index]) == encode(text)
            assert upgraded[index].character_offsets.tolist() == [
                [position, position + 1] for position in range(len(text))]


@pytest.mark.parametrize("kind", ["per_batch", "barrier"])
def test_apply_can_select_no_rows(session, kind):
    seen = []

    def select_none(rows):
        seen.extend(rows["r"].column("id").to_pylist())
        return rows["r"].take([])

    query = (filtered(session)
             .apply(select_none, columns=[col("r.id")], kind=kind)
             .ai_filter(prompt("after: {0}", col("r.body")))
             .select("r.id", order="as_written"))
    result = _run(query, make_executor(
        {"r": {"q:": [True] * 4, "after:": [True] * 4}},
        functions={"select_none": select_none}))
    assert seen == ["aa", "bb", "ccc", "d"]
    assert result.to_rows() == []
