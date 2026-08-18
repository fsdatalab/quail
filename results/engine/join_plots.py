"""Generate join-experiment plots from committed result JSONs."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent


def load(name):
    with open(HERE / name) as f:
        return json.load(f)


def nway_query_plan():
    """Query plan tree, bottom-up like a database EXPLAIN, with time per node."""
    d = load("join_nway3.json")
    r = d["result"]
    n_a, n_b, n_c = d["n_a"], d["n_b"], d["n_c"]

    s1 = r["stage1_wall_s"]
    s2 = r["stage2_wall_s"]
    survivors = r["survivors"]
    planted_expected = r["planted_expected_survivors"]
    triples = r["triples"]
    total = r.get("total_wall_s", s1 + s2)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.35", linewidth=1.5)

    def node(x, y, text, color, textcolor="white", fontsize=10):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor,
                bbox=dict(facecolor=color, edgecolor=color, **box_kw))

    def edge(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="#bdc3c7", lw=1.8,
                solid_capstyle="round")

    def time_label(x, y, text, color="#2c3e50"):
        ax.text(x, y, text, ha="left", va="center", fontsize=9.5,
                color=color, style="italic")

    # leaves (bottom)
    node(2.0, 0.5, f"Scan A\n{n_a} docs", "#7f8c8d")
    node(5.0, 0.5, f"Scan B\n{n_b} docs", "#2980b9")
    node(8.0, 0.5, f"Scan C\n{n_c} docs", "#e67e22")

    # stage 1: packed join B x A
    edge(2.0, 0.95, 3.5, 2.05)
    edge(5.0, 0.95, 3.5, 2.05)
    node(3.5, 2.5, f"Packed Join\nB × A\n{n_b * n_a:,} pairs", "#2980b9")
    time_label(5.3, 2.5, f"{s1:.1f} s")

    # filter: drop unmatched, run each survivor once
    edge(3.5, 2.95, 3.5, 3.75)
    node(3.5, 4.2,
         f"Drop B with 0 matches\nrun each survivor once\n"
         f"{survivors}/{n_b} B remain, prefix KV cached",
         "#2c3e50", fontsize=9)
    time_label(5.3, 4.4, f"~0 s")
    time_label(5.3, 4.0,
               f"(expected {planted_expected}/{n_b})", color="#95a5a6")

    # stage 2: packed join B x C, reuse kept KV
    edge(3.5, 4.65, 5.0, 5.45)
    edge(8.0, 0.95, 5.0, 5.45)
    node(5.0, 5.9,
         f"Packed Join\nB × C  (reads cached KV)\n"
         f"{r['stage2_pairs']:,} pairs",
         "#e67e22")
    time_label(6.9, 5.9, f"{s2:.1f} s")

    # assemble
    edge(5.0, 6.4, 5.0, 6.85)
    node(5.0, 7.3, f"Assemble\n{triples:,} triples", "#27ae60")
    time_label(6.9, 7.3, "~0 s")

    # total
    ax.text(5.0, 8.2, f"Total: {total:.1f} s", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2c3e50")

    ax.set_title("3-Way Join Query Plan", fontsize=14, pad=8)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3_plan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  join_nway3_plan.png  (query plan, {total:.1f} s total)")


def nway_time_bar():
    """Horizontal stacked bar: time fraction per stage."""
    d = load("join_nway3.json")
    r = d["result"]

    s1 = r["stage1_wall_s"]
    s2 = r["stage2_wall_s"]
    total = r.get("total_wall_s", s1 + s2)

    fig, ax = plt.subplots(figsize=(8, 2.5))

    bar_h = 0.5
    y = 0

    ax.barh(y, s1, height=bar_h, left=0, color="#2980b9",
            edgecolor="white", linewidth=0.8)
    ax.barh(y, s2, height=bar_h, left=s1, color="#e67e22",
            edgecolor="white", linewidth=0.8)

    gpu = s1 + s2
    ax.text(s1 / 2, y, f"Stage 1: B×A\n{s1:.1f} s ({s1/gpu:.0%})",
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")
    ax.text(s1 + s2 / 2, y,
            f"Stage 2: B×C\n{s2:.1f} s ({s2/gpu:.0%})",
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")

    ax.set_xlim(0, max(total, gpu) * 1.08)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("GPU seconds per stage", fontsize=11)
    ax.set_title(f"3-Way Join Time Breakdown  ({total:.1f} s total)",
                 fontsize=13, pad=8)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  join_nway3_time.png  ({s1:.1f} + {s2:.1f} = {total:.1f} s)")


def nway_vs_stock():
    """Measured staged join against estimated grouped stock vLLM.

    The stock bar is arithmetic from committed measurements - never
    run. Its configuration mirrors the 2-way stock arm exactly: one
    synchronous request per (anchor, partner) pair, submitted
    grouped by B document so the engine's prefix cache hits, prefix
    caching on, admission from the same budget formula. All 100 B
    prefixes (368k tokens) fit the ~978k-token pool, so each prefix
    computes once and nothing is evicted - grouping gives stock the
    same computed-once token accounting we have.

    The four components, each anchored to a measurement:
      fresh   7.025M tokens (368k B prefixes once + 3.104M A
              suffixes + 3.553M C suffixes) at 70.8k tok/s -> 99.2 s
              [the rate is the 2-way stock arm's own measured
               fresh-compute rate: 433 s wall - 234 s host - 95 s
               reads = 104 s over 7.36M fresh tokens]
      reads   every request's suffix attends over B's 3,684 cached
              prefix tokens through the paged pool: 20,000 x 3,684
              x 125 ns/cached-token                         -> 9.2 s
              [125 ns from the c2 calibration cells]
      host    the engine ingests each request's FULL prompt: 80.33M
              prompt tokens x (234 s / 770M)               -> 24.4 s
              [finding 1's measured host term, scaled by tokens
               ingested; scaling by requests instead gives 18 s]
      fixed   20,000 requests x 50.3 us                    -> 1.0 s
                                               total       ~ 134 s
    """
    d = load("join_nway3.json")
    r = d["result"]
    s1, s2 = r["stage1_wall_s"], r["stage2_wall_s"]
    total = r["total_wall_s"]
    tails = round(total - s1 - s2, 1)

    comps = [("fresh compute\n99 s", 99.2, "#95a5a6"),
             ("reads", 9.2, "#b8c2c6"),
             ("host\n24 s", 24.4, "#7f8c8d"),
             ("", 1.0, "#ccd4d7")]
    est_total = round(sum(v for _, v, _ in comps))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ink = "#2c3e50"

    labels = [
        "Quail staged join\n(measured)",
        "Stock vLLM, grouped\n(estimated — never run)",
    ]

    # measured bar: stage segments + readout tail, 2px white gaps
    ax.barh(0, s1, height=0.5, left=0, color="#2980b9",
            edgecolor="white", linewidth=2)
    ax.barh(0, s2, height=0.5, left=s1, color="#e67e22",
            edgecolor="white", linewidth=2)
    ax.barh(0, tails, height=0.5, left=s1 + s2, color=ink,
            edgecolor="white", linewidth=2)
    ax.text(s1 / 2, 0, f"stage 1\n{s1:.1f} s", ha="center",
            va="center", fontsize=9.5, color="white",
            fontweight="bold")
    ax.text(s1 + s2 / 2, 0, f"stage 2\n{s2:.1f} s", ha="center",
            va="center", fontsize=9.5, color="white",
            fontweight="bold")
    ax.text(total + 3, 0, f"{total:.1f} s", ha="left", va="center",
            fontsize=11, color=ink, fontweight="bold")

    # estimated bar: component segments, gray + texture =
    # arithmetic, not a measurement
    left = 0.0
    for name, val, color in comps:
        ax.barh(1, val, height=0.5, left=left, color=color,
                edgecolor="white", linewidth=2, hatch="//")
        if name:
            ax.text(left + val / 2, 1, name, ha="center",
                    va="center", fontsize=9, color=ink)
        left += val
    ax.text(left + 3, 1,
            f"~{est_total} s = 99 fresh + 9 reads + 24 host + 1 fixed",
            ha="left", va="center", fontsize=10.5, color=ink,
            fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds (10,000 + 10,000 pairs)", fontsize=11)
    ax.set_xlim(0, 240)
    ax.set_title("3-Way Join: measured vs grouped stock vLLM "
                 "(estimated)", fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3_vs_stock.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  join_nway3_vs_stock.png  ({total:.1f} s measured vs "
          f"~{est_total} s estimated grouped stock)")


if __name__ == "__main__":
    print("Generating join plots...")
    nway_query_plan()
    nway_time_bar()
    nway_vs_stock()
    print("Done.")
