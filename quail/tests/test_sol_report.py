"""Tests for quail.bench.sol_report: the report table and chart issue
#26 asked for (reports/2026-08-23-sol-throughput-cost.md section 15
called this the one piece not yet built). No GPU, no matplotlib
required except for the one plotting test, which skips if matplotlib
isn't installed - matching reports/make_plots.py's on-demand
dependency."""

import json

import pytest

from quail.bench import sol_report
from quail.bench.quailb import _docs_count


def _row(query, desc, evaluated_stage0, wall_s, sol_s, fresh_tokens,
        alias="r", extra_stages=None):
    """One query's report row, in the exact shape run_suite() builds
    (quailb.py's row = dict(...) in the try block) - a plain filter
    chain's stage 0, so _docs_count reads it as the corpus size."""
    stages = [{"op": "filter", "alias": alias, "stage": 0,
              "evaluated": evaluated_stage0}]
    if extra_stages:
        stages += extra_stages
    efficiency = round(sol_s / wall_s, 4) if wall_s else None
    return dict(query=query, desc=desc, wall_s=wall_s, boot_s=0.0,
               boot_kind="warm", fresh_tokens=fresh_tokens,
               rows=evaluated_stage0, peak_gib=64.0, stages=stages,
               store=None, sol_s=sol_s, sol_efficiency=efficiency,
               tokens_per_s=round(fresh_tokens / wall_s) if wall_s else None,
               docs_per_s=round(evaluated_stage0 / wall_s, 1) if wall_s else None,
               cost_dollars=round(wall_s * 0.001097, 6) if wall_s else None)


def _suite(cold_rows, warm_rows):
    return dict(sf=0.1, lf=1, gpus=1, model="qwen3-4b-fp8",
               passes=dict(cold=dict(queries=cold_rows, pass_wall_s=0.0),
                          warm=dict(queries=warm_rows, pass_wall_s=0.0)))


# ---- build_rows: the normal case ------------------------------------

def test_build_rows_normal_query_uses_cold_size_warm_rate():
    """The documented convention: Docs/Tokens describe the workload
    (cold, no restores); SOL/Efficiency/rates use warm, matching the
    issue's own "(warm)" column labels."""
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    rows = sol_report.build_rows(_suite([cold], [warm]))
    assert len(rows) == 1
    r = rows[0]
    assert r["error"] is None
    assert r["query"] == "IMDB-1"
    assert r["docs"] == 5000                # from cold
    assert r["tokens"] == 1_774_233          # from cold
    assert r["cold_s"] == 14.61
    assert r["warm_s"] == 16.34
    assert r["sol_s"] == 10.195              # from warm (rate_src)
    assert r["efficiency"] == pytest.approx(10.195 / 16.34, abs=1e-4)
    assert r["docs_per_s_warm"] == warm["docs_per_s"]
    assert r["tokens_per_s_warm"] == warm["tokens_per_s"]
    assert r["cost_cold"] == cold["cost_dollars"]
    assert r["cost_warm"] == warm["cost_dollars"]


def test_build_rows_real_validated_numbers():
    """The exact IMDB-2 numbers from the corrected GPU run (reports/
    2026-08-23-sol-throughput-cost.md section 11's table) - this is
    not a synthetic sanity check, it's the real measured data, so the
    row this produces should match what was actually reported."""
    cold = _row("IMDB-2", "join only", 5000, 53.18, 35.5492, 6_124_233,
               extra_stages=[{"op": "join", "anchor": "r",
                              "partners": ["a"], "tuples": 60000}])
    warm = _row("IMDB-2", "join only", 5000, 52.42, 33.0578, 5_690_657,
               extra_stages=[{"op": "join", "anchor": "r",
                              "partners": ["a"], "tuples": 60000}])
    rows = sol_report.build_rows(_suite([cold], [warm]))
    r = rows[0]
    assert r["sol_s"] == 33.0578
    assert r["efficiency"] == pytest.approx(0.6306, abs=0.001)
    assert r["cold_s"] == 53.18
    assert r["warm_s"] == 52.42


# ---- build_rows: partial failures ------------------------------------

def test_build_rows_cold_errored_falls_back_to_warm_for_everything():
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    cold_error = dict(query="IMDB-1", desc="filter only",
                      error="RefusalError: no store configured")
    rows = sol_report.build_rows(_suite([cold_error], [warm]))
    r = rows[0]
    assert r["error"] is None                # not a total failure
    assert r["cold_s"] is None
    assert r["warm_s"] == 16.34
    assert r["docs"] == 5000                 # fell back to warm's size
    assert r["sol_s"] == 10.195
    assert r["cost_cold"] is None
    assert r["cost_warm"] is not None


def test_build_rows_warm_errored_falls_back_to_cold_for_rates():
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm_error = dict(query="IMDB-1", desc="filter only",
                      error="ConnectionError: worker unreachable")
    rows = sol_report.build_rows(_suite([cold], [warm_error]))
    r = rows[0]
    assert r["error"] is None
    assert r["cold_s"] == 14.61
    assert r["warm_s"] is None
    assert r["sol_s"] == 10.195               # fell back to cold's sol_s
    assert r["docs_per_s_warm"] is None        # no warm row, no warm rate
    assert r["tokens_per_s_warm"] is None
    assert r["cost_warm"] is None
    assert r["cost_cold"] is not None


def test_build_rows_both_passes_errored_is_a_full_error_row():
    cold_error = dict(query="IMDB-1", desc="filter only",
                      error="AssertionError: sol violation")
    warm_error = dict(query="IMDB-1", desc="filter only",
                      error="AssertionError: sol violation")
    rows = sol_report.build_rows(_suite([cold_error], [warm_error]))
    r = rows[0]
    assert r["error"] == "AssertionError: sol violation"
    for key in ("docs", "tokens", "cold_s", "warm_s", "sol_s",
               "efficiency", "docs_per_s_warm", "tokens_per_s_warm",
               "cost_cold", "cost_warm"):
        assert r[key] is None, f"{key} should be None on full failure"


def test_build_rows_query_missing_entirely_from_one_pass():
    """A query that only ran in one pass (e.g. --only differed, or a
    pass crashed before reaching it) shouldn't KeyError - it's the
    same as that pass having errored, from this function's view."""
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    rows = sol_report.build_rows(_suite([cold], []))
    assert len(rows) == 1
    r = rows[0]
    assert r["error"] is None
    assert r["cold_s"] == 14.61
    assert r["warm_s"] is None
    assert r["sol_s"] == 10.195


def test_build_rows_empty_suite():
    assert sol_report.build_rows(_suite([], [])) == []


def test_build_rows_preserves_query_order_no_duplicates():
    """Query order should follow cold's list then any warm-only
    additions, each id appearing exactly once even though it's
    looked up in both passes."""
    c1 = _row("A", "d", 10, 1.0, 0.5, 100)
    c2 = _row("B", "d", 10, 1.0, 0.5, 100)
    w1 = _row("A", "d", 10, 1.0, 0.5, 100)
    w2 = _row("B", "d", 10, 1.0, 0.5, 100)
    rows = sol_report.build_rows(_suite([c1, c2], [w1, w2]))
    assert [r["query"] for r in rows] == ["A", "B"]


# ---- _docs_count_of ---------------------------------------------------

def test_docs_count_of_delegates_to_quailb_docs_count():
    row = _row("IMDB-2", "join only", 5000, 53.18, 35.5492, 6_124_233,
              extra_stages=[{"op": "join", "anchor": "r", "partners": ["a"],
                            "tuples": 60000}])
    # this row's filter is on alias "r" and the join anchor is also
    # "r", so _docs_count should read the anchor's stage-0 count (5000)
    assert sol_report._docs_count_of(row) == _docs_count(row) == 5000


def test_docs_count_of_missing_stages_returns_none():
    assert sol_report._docs_count_of(dict(query="X")) is None


# ---- formatting ---------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (None, "—"), (500, "500"), (1774233, "1.8M"), (102400, "102k"),
])
def test_fmt_tokens(n, expected):
    assert sol_report._fmt_tokens(n) == expected


@pytest.mark.parametrize("n,expected", [
    (None, "—"), (0.6306, "63%"), (1.0, "100%"), (0.0, "0%"),
])
def test_fmt_pct(n, expected):
    assert sol_report._fmt_pct(n) == expected


@pytest.mark.parametrize("n,expected", [
    (None, "—"), (0.08062, "$0.0806"), (0.002345, "$0.0023"),
])
def test_fmt_cost(n, expected):
    assert sol_report._fmt_cost(n) == expected


def test_fmt_rate_switches_to_k_at_1000():
    assert sol_report._fmt_rate(999) == "999"
    assert sol_report._fmt_rate(1000) == "1k"
    assert sol_report._fmt_rate(102400) == "102k"
    assert sol_report._fmt_rate(None) == "—"


# ---- render_markdown_table ---------------------------------------------

def test_render_markdown_table_shape_and_content():
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    rows = sol_report.build_rows(_suite([cold], [warm]))
    table = sol_report.render_markdown_table(rows)
    lines = table.splitlines()
    assert len(lines) == 3   # header, separator, one data row
    # every data row (not the separator) has 11 columns -> 12 pipes
    assert lines[0].count("|") == 12
    assert lines[2].count("|") == 12
    assert "IMDB-1" in lines[2]
    assert "62%" in lines[2]   # 10.195 / 16.34 = 0.6239 -> "62%"
    assert "1.8M" in lines[2]


def test_render_markdown_table_error_row_visible_not_dropped():
    cold_error = dict(query="FEV-7", desc="broken", error="Refusal: x")
    warm_error = dict(query="FEV-7", desc="broken", error="Refusal: x")
    rows = sol_report.build_rows(_suite([cold_error], [warm_error]))
    table = sol_report.render_markdown_table(rows)
    assert "FEV-7" in table
    assert "ERROR" in table
    assert "Refusal: x" in table


def test_render_markdown_table_empty_rows():
    table = sol_report.render_markdown_table([])
    assert table.splitlines()[0].startswith("| Query")


def test_render_markdown_table_error_containing_pipe_stays_one_row():
    """An adversarial review (2026-08-25) found that an error message
    containing "|" - a real possibility, KeyError reprs quote dicts -
    added an extra column separator and misaligned every cell after
    it in that row. Every row, including error rows, must have the
    same number of UNESCAPED column separators as the header - a
    backslash-escaped "\\|" is one literal character in a markdown
    table, not a new column, so it must not count as one."""
    import re

    cold_error = dict(query="X-1", desc="broken",
                      error='KeyError: "a|b" not found in {"x": 1}')
    warm_error = dict(query="X-1", desc="broken",
                      error='KeyError: "a|b" not found in {"x": 1}')
    rows = sol_report.build_rows(_suite([cold_error], [warm_error]))
    table = sol_report.render_markdown_table(rows)
    lines = table.splitlines()
    unescaped = lambda s: len(re.findall(r"(?<!\\)\|", s))
    header_columns = unescaped(lines[0])
    for line in lines[1:]:
        assert unescaped(line) == header_columns, line
    assert "a\\|b" in table   # escaped, not silently dropped either


def test_render_markdown_table_docs_none_renders_dash_not_python_none():
    """An adversarial review (2026-08-25) found that a success-path
    row with docs=None (no stage-0 filter and no join - _docs_count's
    fallback logic returns None) printed the literal string "None"
    instead of "—", the only field skipping the None->dash convention
    every other column follows."""
    row = _row("X-1", "no stage-0, no join", 0, 5.0, 1.0, 100)
    row["stages"] = [{"op": "filter", "alias": "r", "stage": 1,
                      "evaluated": 0}]   # no stage 0 present at all
    rows = sol_report.build_rows(_suite([row], [row]))
    assert rows[0]["docs"] is None   # sanity: this really is the None case
    table = sol_report.render_markdown_table(rows)
    assert "None" not in table
    assert "—" in table.splitlines()[2]


# ---- plot_sol_comparison ---------------------------------------------

def test_plot_sol_comparison_writes_a_file(tmp_path):
    pytest.importorskip("matplotlib")
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    rows = sol_report.build_rows(_suite([cold], [warm]))
    out = tmp_path / "sol.png"
    sol_report.plot_sol_comparison(rows, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_sol_comparison_skips_error_rows(tmp_path):
    pytest.importorskip("matplotlib")
    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    cold_error = dict(query="FEV-7", desc="broken", error="x")
    warm_error = dict(query="FEV-7", desc="broken", error="x")
    rows = sol_report.build_rows(_suite([cold, cold_error],
                                        [warm, warm_error]))
    out = tmp_path / "sol.png"
    sol_report.plot_sol_comparison(rows, str(out))   # must not raise
    assert out.exists()


def test_plot_sol_comparison_raises_on_nothing_to_plot(tmp_path):
    pytest.importorskip("matplotlib")
    cold_error = dict(query="FEV-7", desc="broken", error="x")
    warm_error = dict(query="FEV-7", desc="broken", error="x")
    rows = sol_report.build_rows(_suite([cold_error], [warm_error]))
    with pytest.raises(ValueError):
        sol_report.plot_sol_comparison(rows, str(tmp_path / "sol.png"))


# ---- main() / CLI ---------------------------------------------------

def test_main_end_to_end(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    import sys as _sys

    cold = _row("IMDB-1", "filter only", 5000, 14.61, 10.195, 1_774_233)
    warm = _row("IMDB-1", "filter only", 5000, 16.34, 10.195, 1_774_233)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(_suite([cold], [warm])))
    out_png = tmp_path / "out.png"

    argv = _sys.argv
    _sys.argv = ["sol_report", str(suite_path), str(out_png)]
    try:
        sol_report.main()
    finally:
        _sys.argv = argv

    assert out_png.exists()
    captured = capsys.readouterr()
    assert "IMDB-1" in captured.out
    assert "wrote" in captured.out
