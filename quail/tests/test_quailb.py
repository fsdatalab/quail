"""CPU checks for the QUAIL-B query catalog and table schemas."""

import itertools

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.bench import quailb
from quail.bench.quailb import ASPECTS, SETS, queries, register_sets
from quail.planner.plan import EngineConfig, Refusal


def _standin_sets(tmp_path):
    """Write small parquet files with the benchmark table schemas."""
    def write(name, col, values):
        pq.write_table(pa.table({
            "id": [f"{name}{i}" for i in range(len(values))],
            col: values,
        }), tmp_path / f"{name}.parquet")

    write("reviews", "body", [f"review text {i}" for i in range(12)])
    write("aspects", "aspect", ASPECTS)
    write("reports", "report", [f"medical report {i}" for i in range(8)])
    write("terms", "term", [f"reaction {i}" for i in range(6)])
    pq.write_table(pa.table({
        "id": [f"cl{i}" for i in range(6)],
        "claim": [f"claim {i}" for i in range(6)],
        "label": ["SUPPORTS" if i % 2 == 0 else "REFUTES"
                  for i in range(6)],
        "evidence_wiki_url": [f"Page_{i}" for i in range(6)],
    }), tmp_path / "claims.parquet")
    write("evidence", "text", [f"Wikipedia passage {i}" for i in range(6)])
    pq.write_table(pa.table({
        "id": [f"lp{i}" for i in range(6)],
        "destination_context": [f"citation excerpt {i}" for i in range(6)],
        "passage_text": [f"cited passage {i}" for i in range(6)],
        "passage_id": [f"passage-{i}" for i in range(6)],
    }), tmp_path / "citations.parquet")
    return tmp_path


def test_all_queries_compile_and_plan(tmp_path):
    _standin_sets(tmp_path)
    sess = quail.Session(EngineConfig(gpus=1), tokenizer=str.split)
    register_sets(sess, tmp_path)
    qdefs = queries(sess)
    expected = {
        *(f"IMDB-{i}" for i in range(1, 6)),
        *(f"BIO-{i}" for i in range(1, 6)),
        *(f"FEV-{i}" for i in range(1, 7)),
        *(f"LEP-{i}" for i in range(1, 8)),
    }
    assert set(qdefs) == expected
    for qid, (_, build) in qdefs.items():
        query = build()
        plan = query.plan()
        assert not isinstance(plan, Refusal), f"{qid} refused: {plan}"
        assert "physical:" in query.explain(), qid


def test_set_table_matches_design():
    assert SETS == {
        "reviews": 50_000,
        "reports": 2_000,
        "claims": 1_000,
        "citations": 2_000,
    }
    assert len(ASPECTS) == 12


# ---- run_suite's SOL/cost wiring -----------------------------------
#
# An adversarial review (2026-08-24) found that none of the tests
# above - or anywhere else in the suite - ever exercised sol_s,
# sol_efficiency, docs_per_s, or cost_dollars, or the SolViolation
# check meant to guard them. That gap is exactly how a real bug
# shipped: the original `raise AssertionError(...)` for a query
# beating the speed-of-light floor sat inside the same try/except
# that turns any exception into a results-row string, so it never
# actually stopped a run - it just became one more `error` entry,
# silently, and run_suite returned normally. These tests exercise
# run_suite itself (not just the pieces), through its _execute seam,
# so that regression can't come back unnoticed.

def _fake_execute(wall_s=1.0, boot_s=0.0, fresh_tokens=1000):
    """A worker stand-in good enough for any single-filter or
    filter+join query in the catalog: every document passes every
    filter stage, every join tuple matches. Real selectivity doesn't
    matter for these tests - only that run_suite's row-building and
    SolViolation wiring run end to end."""
    def _exec(payload):
        out = dict(filters={}, joins=[], wall_s=wall_s, boot_s=boot_s,
                  fresh_tokens=fresh_tokens)
        survivors = {a: list(range(len(d)))
                    for a, d in payload["docs"].items()}
        for alias, qids in payload["filters"].items():
            n = len(payload["docs"][alias])
            out["filters"][alias] = {d: [1] * len(qids) for d in range(n)}
        for j in payload["joins"]:
            anchor, partners = j["anchor"], j["partners"]
            tuples = list(itertools.product(
                *[survivors[p] for p in partners]))
            rows = {a: [1 for _ in tuples] for a in survivors[anchor]}
            out["joins"].append(dict(
                rows=rows, anchor_index=survivors[anchor],
                partner_index=[list(t) for t in tuples]))
        return out
    return _exec


def test_sol_violation_aborts_run_suite(tmp_path, monkeypatch):
    """The regression test for the bug the adversarial review found:
    a query reporting an impossibly fast wall_s must abort run_suite
    with SolViolation, not disappear into an `error` row while the
    suite keeps going."""
    _standin_sets(tmp_path)
    monkeypatch.setattr(quailb, "build_sets", lambda *a, **k: tmp_path)
    with pytest.raises(quailb.SolViolation):
        quailb.run_suite(str(tmp_path), sf=0.1, only={"IMDB-1"},
                         _execute=_fake_execute(wall_s=0.0001))


def test_run_suite_reports_sol_and_cost_fields(tmp_path, monkeypatch):
    """A normal (non-violating) run reports every new issue #26 field,
    with sane values, and no error."""
    _standin_sets(tmp_path)
    monkeypatch.setattr(quailb, "build_sets", lambda *a, **k: tmp_path)
    suite = quailb.run_suite(str(tmp_path), sf=0.1, only={"IMDB-5"},
                             _execute=_fake_execute(wall_s=30.0, boot_s=2.0))
    row = suite["passes"]["cold"]["queries"][0]
    assert "error" not in row, row
    assert row["query"] == "IMDB-5"
    for key in ("sol_s", "sol_efficiency", "tokens_per_s", "docs_per_s",
               "cost_dollars"):
        assert row[key] is not None, f"{key} missing: {row}"
    assert 0 < row["sol_efficiency"] <= 1.0 + quailb.EFFICIENCY_TOLERANCE
    assert row["cost_dollars"] > 0


def test_docs_per_s_two_sided_query_uses_anchor_only():
    """FEV-5/6 and LEP-7's shape: a filter on both the join's anchor
    and a partner table, so report["stages"] has TWO stage-0 filter
    entries over two different tables. The original _docs_per_s
    summed them - unrelated tables' document counts added together -
    caught by an adversarial review, 2026-08-24."""
    report = dict(stages=[
        {"op": "filter", "alias": "c", "stage": 0, "evaluated": 100},
        {"op": "filter", "alias": "e", "stage": 0, "evaluated": 57},
        {"op": "join", "anchor": "e", "partners": ["c"], "tuples": 1980},
    ])
    assert quailb._docs_per_s(report, 1.82) == pytest.approx(57 / 1.82, abs=0.05)


def test_docs_count_matches_docs_per_s_numerator():
    """_docs_count (split out of _docs_per_s for sol_report.py's raw
    "Docs" column) must agree with _docs_per_s's own numerator for
    every query shape - filter-only, join-only, and two-sided."""
    filter_only = dict(stages=[
        {"op": "filter", "alias": "r", "stage": 0, "evaluated": 500},
        {"op": "filter", "alias": "r", "stage": 1, "evaluated": 480},
    ])
    join_only = dict(stages=[
        {"op": "join", "anchor": "r", "partners": ["a"], "tuples": 2000},
    ])
    two_sided = dict(stages=[
        {"op": "filter", "alias": "c", "stage": 0, "evaluated": 100},
        {"op": "filter", "alias": "e", "stage": 0, "evaluated": 57},
        {"op": "join", "anchor": "e", "partners": ["c"], "tuples": 1980},
    ])
    for report in (filter_only, join_only, two_sided):
        count = quailb._docs_count(report)
        rate = quailb._docs_per_s(report, 10.0)
        assert count is not None
        assert rate == pytest.approx(count / 10.0, abs=0.05)
    assert quailb._docs_count(filter_only) == 500
    assert quailb._docs_count(join_only) == 2000
    assert quailb._docs_count(two_sided) == 57


def test_docs_per_s_zero_wall_s_returns_none():
    """wall_s=0 must not raise ZeroDivisionError - it should report
    unknown throughput, not crash the row."""
    report = dict(stages=[
        {"op": "filter", "alias": "r", "stage": 0, "evaluated": 10},
    ])
    assert quailb._docs_per_s(report, 0) is None
    assert quailb._docs_count(report) == 10   # the count itself is fine
