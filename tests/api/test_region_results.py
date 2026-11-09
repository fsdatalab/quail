"""Edited-plan scope and per-region ANALYZE results, without a GPU."""

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, oracle_executor
from helpers.sessions import CONFIG, TOKENIZER

import quail
from quail.api.query import RegionQuery

SINGLE = ("SELECT r.id FROM reviews r "
          "WHERE AI_FILTER(PROMPT('q1: {0}', r.body))")
CHAIN = ("WITH kept AS (SELECT r.id, r.body FROM reviews r "
         "WHERE AI_FILTER(PROMPT('q1: {0}', r.body)) ORDER BY r.id LIMIT 2) "
         "SELECT r.id FROM kept r WHERE AI_FILTER(PROMPT('q2: {0}', r.body)) "
         "ORDER BY r.id")


@pytest.fixture
def session():
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register("reviews", quail.DocumentProvider.from_table(pa.table({
            "id": ["r0", "r1", "r2", "r3"],
            "body": [f"document {index} " * 5 for index in range(4)],
        }), id_col="id"))
        yield session


@pytest.mark.parametrize("sql,count", [("SELECT * FROM reviews", 0), (CHAIN, 2)])
def test_edited_plan_scope_is_rejected_before_execution_context(session, monkeypatch,
                                                               sql, count):
    edited = session.sql(SINGLE).plan()
    query = session.sql(sql)
    assert query.num_regions == count
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("invalid edited-plan scope reached execution")

    # Binding is already complete. Even constructing the execution context is
    # too late to reject the scope, before any model callback or GPU work.
    monkeypatch.setattr("quail.api.context.SessionContext", forbidden)
    with pytest.raises(ValueError, match=f"exactly one model region.*has {count}"):
        query.run(plan=edited, physical_executor=forbidden)
    assert calls == []


def _run(query):
    requests = []
    execute = oracle_executor(Oracle(filters=lambda *_: True))

    def executor(request):
        requests.append(request)
        return execute(request)

    return query.run(physical_executor=executor), requests


def _analyze(query, result, monkeypatch):
    seen = []
    original = RegionQuery._explain

    def explain(region, executed, verbose):
        if executed is not None:
            assert executed is result.region_results[region.region_id]
            assert executed.region_id == region.region_id
        seen.append((region.region_id, executed))
        return original(region, executed, verbose)

    monkeypatch.setattr(query, "run", lambda: result)
    monkeypatch.setattr(RegionQuery, "_explain", explain)
    return query.explain(analyze=True), seen


def test_analyze_matches_reverse_execution_order_and_uses_downstream_actual_plan(
        session, monkeypatch):
    query = session.sql(CHAIN)
    regions = query.regions()
    assert len(regions) == 2
    logical_order = [region.region_id for region in regions]
    downstream = regions[0]
    # Inspection has no upstream results yet, so this plan used empty inputs.
    assert sum(len(tokens) for tokens in downstream._document_tokens.values()) == 0
    placeholder = downstream.plan()

    result, requests = _run(query)
    assert len(requests) == 2
    assert result.to_rows() == [("r0",), ("r1",)]
    assert list(result.region_results) == list(reversed(logical_order))
    assert result.explain().count("model region") == 2
    assert [report["region_id"] for report in result.report["regions"]] == logical_order
    executed = result.region_results[downstream.region_id]
    assert executed.physical_plan is not placeholder
    assert executed.physical_plan.graph.to_json() != placeholder.graph.to_json()
    for measured in result.region_results.values():
        assert measured.plan is measured.physical_plan.graph
        assert measured.node_metrics
        assert any(metric.input_rows > 0 for metric in measured.node_metrics.values())

    # Observe the renderer's argument, not just the headings: the estimates and
    # operator graph must be from the nonempty execution, not the placeholder.
    import quail.api.query as query_module

    render = query_module.explain
    rendered_plans = []

    def record(logical, operators, physical, **options):
        assert physical is options["result"].physical_plan
        assert operators == physical.operators
        rendered_plans.append(physical)
        return render(logical, operators, physical, **options)

    monkeypatch.setattr(query_module, "explain", record)
    text, seen = _analyze(query, result, monkeypatch)
    assert [key for key, _ in seen] == logical_order
    assert rendered_plans[0] is executed.physical_plan
    assert text.count("run:") == 2
    assert text.count("est. rows") == 2
    assert "not executed" not in text


def test_identical_sibling_regions_have_distinct_stable_ids(session, monkeypatch):
    query = session.sql(f"{SINGLE} UNION ALL {SINGLE}")
    first, second = query.regions()
    assert first.logical == second.logical
    assert first.operators == second.operators
    assert first.region_id != second.region_id
    result, requests = _run(query)
    assert len(requests) == 2
    assert result.count() == 8
    assert set(result.region_results) == {first.region_id, second.region_id}
    assert len({id(value) for value in result.region_results.values()}) == 2
    # Neither dict insertion order nor completion order determines matching.
    result.region_results = dict(reversed(list(result.region_results.items())))
    text, seen = _analyze(query, result, monkeypatch)
    assert [key for key, _ in seen] == [first.region_id, second.region_id]
    assert text.count("run:") == 2


def test_analyze_labels_a_region_not_read_by_sql_as_unexecuted(session, monkeypatch):
    query = session.sql(CHAIN)
    result, _ = _run(query)
    missing = query.regions()[0].region_id
    # Simulate SQL short-circuiting a region (e.g. a LIMIT above a union).
    del result.region_results[missing]
    text, seen = _analyze(query, result, monkeypatch)
    assert (missing, None) in seen
    assert text.count("not executed (estimated plan):") == 1
    assert text.count("run:") == 1


def test_analyze_rejects_unknown_executed_region_id(session, monkeypatch):
    query = session.sql(SINGLE)
    result, _ = _run(query)
    result.region_results[999] = next(iter(result.region_results.values()))
    monkeypatch.setattr(query, "run", lambda: result)
    with pytest.raises(RuntimeError, match="do not match the query"):
        query.explain(analyze=True)
