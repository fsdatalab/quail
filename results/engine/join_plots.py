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
    """Measured staged join against two stock-vLLM estimates.

    Both stock numbers are arithmetic from committed measurements -
    never run. Submission strategy for both: one request per
    (anchor, partner) pair, grouped by B document.

    Estimate 1 - prefix caching on (all 100 B prefixes fit the
    ~978k-token pool, so every prefix computes once):
      fresh   7.025M tokens (368k prefix + 3.104M A + 3.553M C)
              at 70.8k tok/s               -> 99.2 s
              (70.8k = the 2-way stock's measured fresh rate:
               433 s wall - 234 s host - 95 s reads over 7.36M)
      reads   20,000 requests x 3,684 cached tokens x 125 ns -> 9.2 s
      host    80.33M prompt tokens x (234 s / 770M)          -> 24.4 s
      fixed   20,000 requests x 50.3 us                      -> 1.0 s
                                                       total ~ 134 s

    Estimate 2 - no prefix reuse (every request prefills its full
    ~4k-token prompt):
      fresh   80.33M tokens; per request T_pre(4,016) =
              9.2 us/tok x 4,016 + 4.93e-10 x 4,016^2 = 44.9 ms
              x 20,000                                  -> 898 s
      host + fixed                                      -> 25.4 s
                                                       total ~ 923 s
    """
    d = load("join_nway3.json")
    r = d["result"]
    s1, s2 = r["stage1_wall_s"], r["stage2_wall_s"]
    total = r["total_wall_s"]
    tails = round(total - s1 - s2, 1)

    est_cached = 134
    est_naive = 923

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ink, muted = "#2c3e50", "#7f8c8d"

    labels = [
        "Quail staged join\n(measured)",
        "Stock vLLM, prefix caching\n(estimated — never run)",
        "Stock vLLM, no prefix reuse\n(estimated — never run)",
    ]
    y = [0, 1, 2]

    # measured bar: stage segments + readout tail, 2px white gaps
    ax.barh(0, s1, height=0.55, left=0, color="#2980b9",
            edgecolor="white", linewidth=2)
    ax.barh(0, s2, height=0.55, left=s1, color="#e67e22",
            edgecolor="white", linewidth=2)
    ax.barh(0, tails, height=0.55, left=s1 + s2, color=ink,
            edgecolor="white", linewidth=2)

    # estimated bars: neutral gray + texture = "not a measurement"
    ax.barh(1, est_cached, height=0.55, color="#aab4b8",
            edgecolor="white", linewidth=2, hatch="//")
    ax.barh(2, est_naive, height=0.55, color="#cdd4d6",
            edgecolor="white", linewidth=2, hatch="//")

    for yi, val, note in [
            (0, total,
             f"{total:.1f} s   (stage 1: {s1:.1f}  +  stage 2: {s2:.1f})"),
            (1, est_cached, f"~{est_cached} s"),
            (2, est_naive,
             f"~{est_naive} s  ({est_naive/total:.0f}x)")]:
        ax.text(val + 12, yi, note, ha="left", va="center",
                fontsize=11, color=ink, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds (10,000 + 10,000 pairs)", fontsize=11)
    ax.set_xlim(0, est_naive * 1.14)
    ax.set_title("3-Way Join: measured against estimated stock vLLM",
                 fontsize=13, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3_vs_stock.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  join_nway3_vs_stock.png  ({total:.1f} s measured vs "
          f"~{est_cached} / ~{est_naive} s estimated)")


if __name__ == "__main__":
    print("Generating join plots...")
    nway_query_plan()
    nway_time_bar()
    nway_vs_stock()
    print("Done.")
