"""Visualizes the scaling behavior the property tests in
tests/test_budgets_sol_properties.py check numerically: does each SOL
term actually behave like the roofline derivation it implements says
it should (quadratic causal attention, linear elementwise, the
compute/memory crossovers landing where the analytic formulas say).
No GPU, no suite JSON needed - every plot here is generated directly
from budgets.py's formulas at Qwen3-4B/H100 constants.

Run from the quail/ directory:

    uv run --with matplotlib python reports/plot_sol_formula_diagnostics.py

Writes PNGs into reports/plots/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)

plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from plot_colors import BLUE, GRAY, GREEN, DARK, ORANGE

from quail.specs import QWEN3_4B_FP8, H100_SXM
from quail.planner import budgets as b

M, D = QWEN3_4B_FP8, H100_SXM


def fig1_causal_attention_scaling():
    """Causal attention time vs. document length, both axes linear.
    Should be flat-ish (memory-bound, ~linear) below the crossover and
    bend upward (quadratic) above it - test_causal_attention_
    is_quadratic_in_document_length_when_compute_bound checks the
    endpoints of exactly this numerically. Swept to 6x the crossover
    (not the formula's full valid range, which spans many orders of
    magnitude and would compress this entire transition into a sliver
    against the origin on a linear axis) so the bend stays visible."""
    crossover = 24_639   # from attention_crossover-style reasoning,
    #                       matches the write-up's worked derivation
    lengths = np.linspace(1, crossover * 6, 200)
    times = [b._causal_prefill_attention_time(M, D, [int(L)])
            for L in lengths]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(lengths, times, color=BLUE, linewidth=2,
           label="causal attention time")
    ax.axvline(crossover, color=GRAY, linestyle="--", linewidth=1)
    ax.text(crossover * 1.08, max(times) * 0.22,
           f"crossover\n~{crossover:,} tok", fontsize=8, color=DARK)
    ax.set_xlabel("document length (tokens)")
    ax.set_ylabel("seconds (one document)")
    ax.set_title("Causal attention: memory-bound, then quadratic")
    fig.savefig(OUT / "sol_formula_causal_attention_scaling.png")
    plt.close(fig)


def fig2_streaming_attention_scaling():
    """Streaming attention time vs. context length, both axes linear,
    for a few fixed chunk sizes. Each line should be flat-ish then
    linear (never quadratic - that's precisely the shape difference
    from causal attention this whole feature exists to get right,
    section 04 of the write-up). Swept to 40,000 tokens - past where
    every chunk size shown has visibly left the flat, memory-bound
    region - rather than the formula's full valid range."""
    contexts = np.linspace(1, 40_000, 200)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for chunk, color in [(8, GRAY), (40, BLUE), (400, GREEN)]:
        times = [b._shared_context_attention_time(M, D, [(chunk, int(c))])
                for c in contexts]
        ax.plot(contexts, times, color=color, linewidth=2,
               label=f"chunk={chunk}")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("seconds (one streaming item)")
    ax.set_title("Streaming attention: memory-bound, then linear")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "sol_formula_streaming_attention_scaling.png")
    plt.close(fig)


def fig3_projection_and_elementwise():
    """Dense projection (has a compute knee) vs. elementwise (purely
    linear, no knee at all) - side by side to make the "one of these
    formulas has a max(), one doesn't" distinction from section 03 of
    the write-up visible, not just stated. Swept to 6x the compute
    knee, both axes linear, so the knee stays visible instead of
    compressed against the origin."""
    knee = b.compute_knee(M, D)
    chunks = np.linspace(1, knee * 6, 200)
    proj = [b._projection_time(M, D, int(c)) for c in chunks]
    elem = [b.elementwise_time(M, D, int(c)) for c in chunks]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(chunks, proj, color=BLUE, linewidth=2,
           label="dense projections (has a knee)")
    ax.plot(chunks, elem, color=ORANGE, linewidth=2,
           label="elementwise tax (no knee - always linear)")
    ax.axvline(knee, color=GRAY, linestyle="--", linewidth=1)
    ax.text(knee * 1.05, max(proj) * 0.05, f"compute knee\n~{knee:,.0f} tok",
           fontsize=8, color=DARK)
    ax.set_xlabel("chunk (tokens)")
    ax.set_ylabel("seconds")
    ax.set_title("Dense projections have a knee; elementwise never does")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "sol_formula_projection_vs_elementwise.png")
    plt.close(fig)


def fig5_representative_breakdown():
    """The four sol_seconds_breakdown() terms for a few illustrative
    (not measured - see sol_report.py / results/*.json for real
    per-query sol_s totals) workload shapes, stacked, to show how the
    mix shifts with query shape - a filter-heavy workload should lean
    on causal attention + projection, a join-heavy one on streaming
    attention."""
    workloads = {
        "filter-only\n(5000 x 500 tok)": dict(
            causal_doc_lengths=[500] * 5000, streaming_chunks_contexts=[]),
        "3-filter chain\n(5000 docs)": dict(
            causal_doc_lengths=[500] * 5000,
            streaming_chunks_contexts=[(30, 530)] * 5000 * 2),
        "join-heavy\n(5000 anchors x 12 partners)": dict(
            causal_doc_lengths=[500] * 5000,
            streaming_chunks_contexts=[(60, 500)] * 5000 * 12),
    }
    labels = list(workloads)
    terms = ["projection", "elementwise", "causal_attention",
            "streaming_attention"]
    colors = {"projection": BLUE, "elementwise": ORANGE,
             "causal_attention": GREEN, "streaming_attention": GRAY}
    data = {t: [] for t in terms}
    for wl in workloads.values():
        bd = b.sol_seconds_breakdown(M, D, **wl)
        total = sum(bd.values())
        for t in terms:
            data[t].append(bd[t] / total * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(labels))
    for t in terms:
        ax.bar(labels, data[t], bottom=bottom, color=colors[t], label=t)
        bottom += np.array(data[t])
    ax.set_ylabel("% of sol_s")
    ax.set_title("Illustrative workload shapes: where the floor goes")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
             ncol=2)
    fig.savefig(OUT / "sol_formula_component_breakdown.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_causal_attention_scaling()
    fig2_streaming_attention_scaling()
    fig3_projection_and_elementwise()
    fig5_representative_breakdown()
    print(f"[plot_sol_formula_diagnostics] wrote 4 PNGs to {OUT}")
