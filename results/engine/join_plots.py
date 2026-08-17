"""Generate join-experiment plots from committed result JSONs."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = pathlib.Path(__file__).parent


def load(name):
    with open(HERE / name) as f:
        return json.load(f)


def wall_chart():
    """Bar chart: wall-clock seconds for each 2-way arm."""
    d = load("join2way.json")
    runs = d["runs"]

    stock_walls = [r["wall"] for r in runs if r["method"] == "stock_grouped"]
    p25k_walls = [r["wall"] for r in runs if r["method"] == "packed_b25305"]
    pstar_walls = [r["wall"] for r in runs if r["method"] == "packed_bstar_cell"]

    stock_mean = sum(stock_walls) / len(stock_walls)
    p25k_mean = sum(p25k_walls) / len(p25k_walls)
    pstar_mean = sum(pstar_walls) / len(pstar_walls)

    labels = [
        "Stock vLLM\n(grouped)",
        "Packed\n(B = 25,305)",
        "Packed\n(B = B* = 421,752)",
    ]
    means = [stock_mean, p25k_mean, pstar_mean]
    colors = ["#7f8c8d", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, means, color=colors, width=0.55, edgecolor="white",
                  linewidth=0.8)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f} s", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    speedup = stock_mean / pstar_mean
    ax.annotate(
        f"{speedup:.1f}x",
        xy=(2, pstar_mean), xytext=(2.42, stock_mean * 0.65),
        fontsize=13, fontweight="bold", color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
        ha="center",
    )

    ax.set_ylabel("Wall-clock seconds", fontsize=12)
    ax.set_title("2-Way Join: 100 Reports x 2,560 Terms (256k pairs)",
                 fontsize=13, pad=12)
    ax.set_ylim(0, stock_mean * 1.25)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "join_wall_times.png", dpi=180)
    plt.close(fig)
    print(f"  join_wall_times.png  ({stock_mean:.0f} / {p25k_mean:.0f} / {pstar_mean:.0f} s)")


def rate_chart():
    """Bar chart: throughput (tokens/s) for each arm."""
    d = load("join2way.json")
    runs = d["runs"]

    stock_rates = [r["tok_s"] for r in runs if r["method"] == "stock_grouped"]
    p25k_rates = [r["tok_s"] for r in runs if r["method"] == "packed_b25305"]
    pstar_rates = [r["tok_s"] for r in runs if r["method"] == "packed_bstar_cell"]

    stock_mean = sum(stock_rates) / len(stock_rates)
    p25k_mean = sum(p25k_rates) / len(p25k_rates)
    pstar_mean = sum(pstar_rates) / len(pstar_rates)

    labels = [
        "Stock vLLM\n(grouped)",
        "Packed\n(B = 25,305)",
        "Packed\n(B = B*)",
    ]
    means = [stock_mean / 1000, p25k_mean / 1000, pstar_mean / 1000]
    colors = ["#7f8c8d", "#2980b9", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, means, color=colors, width=0.55, edgecolor="white",
                  linewidth=0.8)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}k", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Throughput (k tokens/s)", fontsize=12)
    ax.set_title("Effective Throughput by Method", fontsize=13, pad=12)
    ax.set_ylim(0, max(means) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "join_throughput.png", dpi=180)
    plt.close(fig)
    print(f"  join_throughput.png  ({stock_mean/1000:.1f}k / {p25k_mean/1000:.1f}k / {pstar_mean/1000:.1f}k)")


def predicted_vs_measured():
    """Grouped bar chart: predicted vs measured wall times."""
    d = load("join2way.json")
    runs = d["runs"]

    stock_walls = [r["wall"] for r in runs if r["method"] == "stock_grouped"]
    p25k_walls = [r["wall"] for r in runs if r["method"] == "packed_b25305"]
    pstar_walls = [r["wall"] for r in runs if r["method"] == "packed_bstar_cell"]

    predicted = [199, 102, 92]
    measured = [
        sum(stock_walls) / len(stock_walls),
        sum(p25k_walls) / len(p25k_walls),
        sum(pstar_walls) / len(pstar_walls),
    ]
    labels = ["Stock vLLM", "Packed (25,305)", "Packed (B*)"]

    x = range(len(labels))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], predicted, w, label="Predicted (GPU terms)",
                color="#f39c12", edgecolor="white", linewidth=0.8)
    b2 = ax.bar([i + w / 2 for i in x], measured, w, label="Measured",
                color="#2c3e50", edgecolor="white", linewidth=0.8)

    for bar, val in zip(b1, predicted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, color="#f39c12")
    for bar, val in zip(b2, measured):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, color="#2c3e50")

    ax.annotate(
        "Missing host\ningestion term\n(+290 s)",
        xy=(0 + w / 2, measured[0]),
        xytext=(0.65, measured[0] * 0.75),
        fontsize=9, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        ha="center",
    )

    ax.set_ylabel("Wall-clock seconds", fontsize=12)
    ax.set_title("Predicted vs Measured Wall Time", fontsize=13, pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(measured) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "join_predicted_vs_measured.png", dpi=180)
    plt.close(fig)
    print(f"  join_predicted_vs_measured.png")


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
    total = s1 + s2

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
         f"{survivors}/{n_b} B remain, keep KV",
         "#2c3e50", fontsize=9)
    time_label(5.3, 4.4, f"~0 s")
    time_label(5.3, 4.0,
               f"(expected {planted_expected}/{n_b})", color="#95a5a6")

    # stage 2: packed join B x C, reuse kept KV
    edge(3.5, 4.65, 5.0, 5.45)
    edge(8.0, 0.95, 5.0, 5.45)
    node(5.0, 5.9,
         f"Packed Join\nB × C  (reuse KV)\n{r['stage2_pairs']:,} pairs",
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
    total = s1 + s2

    fig, ax = plt.subplots(figsize=(8, 2.5))

    bar_h = 0.5
    y = 0

    ax.barh(y, s1, height=bar_h, left=0, color="#2980b9",
            edgecolor="white", linewidth=0.8)
    ax.barh(y, s2, height=bar_h, left=s1, color="#e67e22",
            edgecolor="white", linewidth=0.8)

    ax.text(s1 / 2, y, f"Stage 1: B×A\n{s1:.1f} s ({s1/total:.0%})",
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")
    ax.text(s1 + s2 / 2, y,
            f"Stage 2: B×C\n{s2:.1f} s ({s2/total:.0%})",
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")

    ax.set_xlim(0, total * 1.08)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Wall-clock seconds", fontsize=11)
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


if __name__ == "__main__":
    print("Generating join plots...")
    wall_chart()
    rate_chart()
    predicted_vs_measured()
    nway_query_plan()
    nway_time_bar()
    print("Done.")
