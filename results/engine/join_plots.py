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
    """Stacked bar for 3-way staged join."""
    d = load("join_nway3.json")
    r = d["result"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["3-Way Join\n(100 x 100 x 100)"], [r["stage1_wall_s"]],
           color="#2980b9", label="Stage 1 (A-B)", width=0.4)
    ax.bar(["3-Way Join\n(100 x 100 x 100)"], [r["stage2_wall_s"]],
           bottom=[r["stage1_wall_s"]], color="#e67e22",
           label="Stage 2 (B-C)", width=0.4)

    total = r["stage1_wall_s"] + r["stage2_wall_s"]
    ax.text(0, total + 2, f"{total:.1f} s total", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.text(0, r["stage1_wall_s"] / 2, f"{r['stage1_wall_s']:.1f} s",
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold")
    ax.text(0, r["stage1_wall_s"] + r["stage2_wall_s"] / 2,
            f"{r['stage2_wall_s']:.1f} s", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold")

    ax.set_ylabel("Wall-clock seconds", fontsize=12)
    ax.set_title("3-Way Staged Join", fontsize=13, pad=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, total * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "join_nway3.png", dpi=180)
    plt.close(fig)
    print(f"  join_nway3.png  ({r['stage1_wall_s']:.1f} + {r['stage2_wall_s']:.1f} s)")


if __name__ == "__main__":
    print("Generating join plots...")
    wall_chart()
    rate_chart()
    predicted_vs_measured()
    nway_chart()
    print("Done.")
