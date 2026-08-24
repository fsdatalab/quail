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
    """Causal attention time vs. document length, log-log. Should be
    flat-ish (memory-bound, ~linear) below the crossover and steepen
    to a slope-2 line (quadratic) above it - test_causal_attention_
    is_quadratic_in_document_length_when_compute_bound checks the
    endpoints of exactly this numerically."""
    lengths = np.logspace(1, 8, 60)
    times = [b._causal_prefill_attention_time(M, D, [int(L)])
            for L in lengths]
    crossover = 24_639   # from attention_crossover-style reasoning,
    #                       matches the write-up's worked derivation

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(lengths, times, color=BLUE, linewidth=2,
             label="causal attention time")
    ax.axvline(crossover, color=GRAY, linestyle="--", linewidth=1)
    ax.text(crossover * 1.15, min(times) * 3,
           f"crossover\n~{crossover:,} tok", fontsize=8, color=DARK)
    ax.set_xlabel("document length (tokens)")
    ax.set_ylabel("seconds (one document)")
    ax.set_title("Causal attention: memory-bound, then quadratic")
    fig.savefig(OUT / "sol_formula_causal_attention_scaling.png")
    plt.close(fig)


def fig2_streaming_attention_scaling():
    """Streaming attention time vs. context length, for a few fixed
    chunk sizes. Each line should be flat-ish then linear (never
    quadratic - that's precisely the shape difference from causal
    attention this whole feature exists to get right, section 04 of
    the write-up)."""
    contexts = np.logspace(1, 8, 60)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for chunk, color in [(8, GRAY), (40, BLUE), (400, GREEN)]:
        times = [b._shared_context_attention_time(M, D, [(chunk, int(c))])
                for c in contexts]
        ax.loglog(contexts, times, color=color, linewidth=2,
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
    the write-up visible, not just stated."""
    chunks = np.logspace(1, 8, 60)
    proj = [b._projection_time(M, D, int(c)) for c in chunks]
    elem = [b.elementwise_time(M, D, int(c)) for c in chunks]
    knee = b.compute_knee(M, D)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(chunks, proj, color=BLUE, linewidth=2,
             label="dense projections (has a knee)")
    ax.loglog(chunks, elem, color=ORANGE, linewidth=2,
             label="elementwise tax (no knee - always linear)")
    ax.axvline(knee, color=GRAY, linestyle="--", linewidth=1)
    ax.text(knee * 1.15, min(proj) * 3, f"compute knee\n~{knee:,.0f} tok",
           fontsize=8, color=DARK)
    ax.set_xlabel("chunk (tokens)")
    ax.set_ylabel("seconds")
    ax.set_title("Dense projections have a knee; elementwise never does")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "sol_formula_projection_vs_elementwise.png")
    plt.close(fig)


def fig4_subadditivity_gap():
    """The gap between "aggregate everything, one max()" (what
    sol_seconds actually does) and "price every item separately and
    sum" (the wrong way, that would overstate the floor). The gap is
    NOT largest when one item dwarfs the other - a huge item's own
    max() dominates both approaches almost identically then, and the
    tiny item barely registers either way. It's largest when a
    compute-bound item and a memory-bound item are comparably sized:
    then "separate" pays each item's own bottleneck in full (~2x one
    item's cost), while "combined" only pays the larger of the two
    summed sides (~1x) - up to a 50% overstatement. Direct
    visualization of test_streaming_attention_is_subadditive_
    across_items, swept properly instead of in the regime where the
    effect vanishes."""
    compute_item = (50_000_000, 50_000_000)   # deep compute-bound
    t_compute = b._shared_context_attention_time(M, D, [compute_item])

    # memory-bound item: chunk=1 (compute side negligible), context
    # swept so its own time ranges from far below to far above
    # t_compute - the ratio of the two items' own times is what
    # actually controls the subadditivity gap, not either item's
    # absolute size
    contexts = np.logspace(6, 16, 80)
    ratios, gaps = [], []
    for c in contexts:
        mem_item = (1, int(c))
        t_mem = b._shared_context_attention_time(M, D, [mem_item])
        items = [compute_item, mem_item]
        combined = b._shared_context_attention_time(M, D, items)
        separate = t_compute + t_mem
        ratios.append(t_mem / t_compute)
        gaps.append((separate - combined) / separate * 100 if separate else 0.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogx(ratios, gaps, color=GREEN, linewidth=2)
    ax.axvline(1.0, color=GRAY, linestyle="--", linewidth=1)
    ax.text(1.15, max(gaps) * 0.9, "equal-sized items\n(largest gap)",
           fontsize=8, color=DARK)
    ax.set_xlabel("memory-bound item's time ÷ compute-bound item's time")
    ax.set_ylabel("% overstatement from summing separately")
    ax.set_ylim(0, max(gaps) * 1.15)
    ax.set_title("Why sol_seconds aggregates before max(), not after")
    fig.savefig(OUT / "sol_formula_subadditivity_gap.png")
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
    fig4_subadditivity_gap()
    fig5_representative_breakdown()
    print(f"[plot_sol_formula_diagnostics] wrote 5 PNGs to {OUT}")
