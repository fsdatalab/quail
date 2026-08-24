"""Issue #26's report table and SOL-comparison chart: the one piece
of the issue that wasn't built alongside sol_seconds() itself (see
reports/2026-08-23-sol-throughput-cost.md section 15 / the published
walkthrough's "what isn't built yet"). Reads a run_suite() suite
dict (or its saved JSON) and produces the column set the issue
specifies:

    Query  Docs  Tokens  Cold(s)  Warm(s)  SOL(s)  Efficiency
    Docs/s(warm)  Tok/s(warm)  Cost(cold)  Cost(warm)

Docs/Tokens describe the query's workload (cold pass, no restores -
the full corpus every document actually went through); SOL/Efficiency/
Docs per s/Tok per s use the warm pass, matching the issue's own
column labels. Cost is reported for both passes separately, per the
issue's explicit point that cold's boot time is a real cost the user
pays once, not something to blend into a steady-state number.

A query that errored in either pass gets a row with `error` set and
every other field None, rather than crashing the whole table - a
partial suite (a Refusal, a transient network blip) should still
render whatever queries did complete.

Run as a script from the quail/ directory (plotting needs
matplotlib, an on-demand dependency, not a core one - see
reports/make_plots.py for the same pattern):

    uv run --with matplotlib python -m quail.bench.sol_report \\
        results/some_quailb_suite.json reports/plots/sol_report.png
"""

import json
import sys
from pathlib import Path


def build_rows(suite: dict) -> list[dict]:
    """One row per query id present in either pass. Pulls straight
    from what run_suite() already computed - no new arithmetic here,
    just picking which pass's numbers go in which column."""
    cold_by_id = {r["query"]: r for r in suite["passes"]["cold"]["queries"]}
    warm_by_id = {r["query"]: r for r in suite["passes"]["warm"]["queries"]}
    ids = list(dict.fromkeys(list(cold_by_id) + list(warm_by_id)))

    rows = []
    for qid in ids:
        cold = cold_by_id.get(qid)
        warm = warm_by_id.get(qid)
        cold_ok = cold is not None and "error" not in cold
        warm_ok = warm is not None and "error" not in warm

        if not cold_ok and not warm_ok:
            err = (cold or warm or {}).get("error", "missing from both passes")
            rows.append(dict(query=qid, desc=(cold or warm or {}).get("desc"),
                             error=err, docs=None, tokens=None,
                             cold_s=None, warm_s=None, sol_s=None,
                             efficiency=None, docs_per_s_warm=None,
                             tokens_per_s_warm=None, cost_cold=None,
                             cost_warm=None))
            continue

        size_src = cold if cold_ok else warm
        rate_src = warm if warm_ok else cold
        rows.append(dict(
            query=qid,
            desc=size_src.get("desc"),
            error=None,
            docs=_docs_count_of(size_src),
            tokens=size_src.get("fresh_tokens"),
            cold_s=cold.get("wall_s") if cold_ok else None,
            warm_s=warm.get("wall_s") if warm_ok else None,
            sol_s=rate_src.get("sol_s"),
            efficiency=rate_src.get("sol_efficiency"),
            docs_per_s_warm=warm.get("docs_per_s") if warm_ok else None,
            tokens_per_s_warm=warm.get("tokens_per_s") if warm_ok else None,
            cost_cold=cold.get("cost_dollars") if cold_ok else None,
            cost_warm=warm.get("cost_dollars") if warm_ok else None,
        ))
    return rows


def _docs_count_of(row: dict) -> int | None:
    """A query row (from run_suite's per-pass "queries" list) already
    carries "stages" in the same shape quailb._docs_count expects
    (report["stages"]) - reuse that one piece of logic directly
    rather than re-deriving the count from docs_per_s * wall_s, which
    would just reintroduce the rounding docs_per_s already applied."""
    if row.get("stages") is None:
        return None
    from quail.bench.quailb import _docs_count
    return _docs_count(row)


def _fmt_tokens(n) -> str:
    if n is None:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _fmt_s(n) -> str:
    return "—" if n is None else f"{n:.1f}"


def _fmt_pct(n) -> str:
    return "—" if n is None else f"{n:.0%}"


def _fmt_rate(n) -> str:
    if n is None:
        return "—"
    return f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def _fmt_cost(n) -> str:
    return "—" if n is None else f"${n:.4f}"


def render_markdown_table(rows: list[dict]) -> str:
    """The issue's column set, as a markdown table. Error rows print
    the query id and error message in place of the numbers, and
    still take one row - a broken query doesn't disappear from the
    report, it's visible as broken."""
    header = ("| Query | Docs | Tokens | Cold (s) | Warm (s) | SOL (s) "
             "| Efficiency | Docs/s (warm) | Tok/s (warm) | Cost (cold) "
             "| Cost (warm) |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        if r["error"]:
            lines.append(f"| {r['query']} | ERROR: {r['error']} | | | | "
                         f"| | | | | |")
            continue
        lines.append(
            f"| {r['query']} | {r['docs']} | {_fmt_tokens(r['tokens'])} "
            f"| {_fmt_s(r['cold_s'])} | {_fmt_s(r['warm_s'])} "
            f"| {_fmt_s(r['sol_s'])} | {_fmt_pct(r['efficiency'])} "
            f"| {_fmt_rate(r['docs_per_s_warm'])} "
            f"| {_fmt_rate(r['tokens_per_s_warm'])} "
            f"| {_fmt_cost(r['cost_cold'])} | {_fmt_cost(r['cost_warm'])} |")
    return "\n".join(lines)


def plot_sol_comparison(rows: list[dict], out_path: str) -> None:
    """Measured (warm wall_s) vs SOL(s) per query, grouped bars, in
    make_plots.py's palette. Skips error rows - nothing to plot for a
    query that didn't produce a number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ACCENT = "#2979FF"
    GRAY = "#9E9E9E"
    DARK = "#424242"

    plotted = [r for r in rows if not r["error"] and r["warm_s"] is not None
              and r["sol_s"] is not None]
    if not plotted:
        raise ValueError("no rows with both warm_s and sol_s to plot")

    labels = [r["query"] for r in plotted]
    measured = [r["warm_s"] for r in plotted]
    sol = [r["sol_s"] for r in plotted]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
    ax.bar([i - width / 2 for i in x], measured, width, label="Measured (warm)",
          color=GRAY)
    ax.bar([i + width / 2 for i in x], sol, width, label="SOL floor",
          color=ACCENT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("Measured wall time vs. speed-of-light floor", color=DARK)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if len(sys.argv) != 3:
        print("usage: python -m quail.bench.sol_report "
             "<suite.json> <out_plot.png>", file=sys.stderr)
        raise SystemExit(2)
    suite = json.loads(Path(sys.argv[1]).read_text())
    rows = build_rows(suite)
    print(render_markdown_table(rows))
    plot_sol_comparison(rows, sys.argv[2])
    print(f"\n[sol_report] wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
