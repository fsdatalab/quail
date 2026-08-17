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


def nway_chart():
    """Flow diagram for the 3-way staged join showing selectivity."""
    d = load("join_nway3.json")
    r = d["result"]
    n_a, n_b, n_c = d["n_a"], d["n_b"], d["n_c"]

    s1 = r["stage1_wall_s"]
    s2 = r["stage2_wall_s"]
    survivors = r["survivors"]
    planted_expected = r["planted_expected_survivors"]
    triples = r["triples"]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    def box(x, y, text, color, textcolor="white", fontsize=11):
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=textcolor,
                bbox=dict(facecolor=color, edgecolor=color, **box_kw))

    def arrow(x1, y1, x2, y2, label="", color="#2c3e50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.22, label, ha="center", va="bottom",
                    fontsize=9, color=color)

    # relations
    box(1.0, 5.0, f"A\n{n_a} docs", "#7f8c8d")
    box(5.0, 5.0, f"B\n{n_b} docs", "#2980b9")
    box(9.0, 5.0, f"C\n{n_c} docs", "#e67e22")

    # stage 1
    arrow(1.0, 4.55, 3.5, 3.65)
    arrow(5.0, 4.55, 3.5, 3.65)
    box(3.5, 3.3, f"Stage 1: B×A\n{n_b * n_a:,} pairs\n{s1:.1f} s",
        "#2980b9")

    # gate with selectivity
    arrow(3.5, 2.85, 3.5, 2.1, color="#c0392b")
    ax.text(4.35, 2.65, f"gate: {survivors}/{n_b} B survive",
            fontsize=9.5, color="#c0392b", fontweight="bold",
            va="center")
    ax.text(4.35, 2.35,
            f"(planted: {planted_expected}/{n_b} expected;\n"
            f" model too permissive — no skips)",
            fontsize=8, color="#95a5a6", va="center")

    # dedup + kept KV
    box(3.5, 1.7, f"dedup + keep KV", "#2c3e50", fontsize=9.5)
    ax.text(5.2, 1.7,
            f"each surviving B runs\nonce, not per A match",
            fontsize=8, color="#7f8c8d", va="center")

    # stage 2
    arrow(3.5, 1.3, 6.5, 0.65)
    arrow(9.0, 4.55, 6.5, 0.65)
    box(6.5, 0.3, f"Stage 2: B×C\n{r['stage2_pairs']:,} pairs\n{s2:.1f} s",
        "#e67e22")

    # output
    arrow(6.5, -0.15, 6.5, -0.8, color="#27ae60")
    box(6.5, -1.15, f"{triples:,} triples", "#27ae60", fontsize=10)

    # total wall time
    ax.text(9.2, 0.3, f"total\n{s1 + s2:.1f} s", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#2c3e50")

    ax.set_title("3-Way Chain Join  (A-B-C, packed, no engine)",
                 fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  join_nway3.png  ({s1:.1f} + {s2:.1f} s, "
          f"{survivors} survivors)")


if __name__ == "__main__":
    print("Generating join plots...")
    wall_chart()
    rate_chart()
    predicted_vs_measured()
    nway_chart()
    print("Done.")
