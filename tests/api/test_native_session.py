"""Rust runs a query's model regions on workers itself, with no Python engine."""

import threading

import _quail_native
from helpers.oracle import Oracle, _serve, finish
from helpers.sessions import CONFIG, TOKENIZER
from test_session_context import ASPECTS, REVIEWS, _same_rows

import quail
from quail.api.engine import Engine, engine_description

QUERIES = [
    "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT r.id, AI.CLASSIFY(r.body, ['good', 'bad']) AS label FROM reviews r "
    "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))",
    "SELECT r.id, a.id FROM reviews r JOIN aspects a "
    "ON AI_FILTER(PROMPT('q2: {0} {1}', r.body, a.aspect))",
    "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) LIMIT 2",
]


def _oracle():
    return Oracle(
        filters=lambda alias, document, position: document % 2 == 0,
        joins=lambda node, assignment: sum(assignment.values()) % 2 == 0,
        labels=lambda spec, ids: spec.labels[sum(ids) % len(spec.labels)])


def _session():
    session = quail.Session(CONFIG, tokenizer=TOKENIZER)
    for name, table in (("reviews", REVIEWS), ("aspects", ASPECTS)):
        session.register(name, quail.DocumentProvider.from_table(table, id_col="id"))
    return session


def test_rust_runs_each_region_on_workers_and_returns_the_engines_rows():
    for text in QUERIES:
        with _session() as session:
            expected = finish(session.sql(text), _oracle()).collect()

        with _session() as session:
            workers = _quail_native.Workers(CONFIG.gpus)
            threading.Thread(
                target=_serve, daemon=True,
                args=(_oracle(), str(workers.paths[0]),
                      engine_description(Engine(session)))).start()
            workers.connect()
            context = quail.SessionContext(session, workers=workers)
            found = context.sql(text)
        assert _same_rows(found, expected, ordered=False), text
        assert found.num_rows > 0, text


def test_rust_explains_a_region_as_the_python_query_does():
    for text in QUERIES:
        with _session() as session:
            expected = session.sql(text).explain()
            (found,) = quail.SessionContext(session)._native.explain_regions(text)
        assert found == expected, text
