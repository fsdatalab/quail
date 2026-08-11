"""GPU busy/idle timeline strips from torch profiler traces.

One row per (engine, operator) pair. A colored bar means a GPU kernel
was running; white means the GPU sat idle while the CPU prepared the
next step. Every row names the method that produced it, because a
strip without its method is unreadable in isolation.

Busy fractions are computed from the trace (union of kernel intervals
over the profiled window), not hardcoded. The drawn slice is a short
zoom so the per-step idle gaps are visible; the busy number in the
row label covers the full window.

Usage:
    python plots/make_plot_strips.py --out results/plots/timeline_strips.png
"""

import argparse
import gzip
import json

# blue = stock vLLM, orange = ours; same assignment as every other
# figure in results/plots
BLUE = "#2a78d6"
ORANGE = "#eb6834"

ROWS = [
    ("results/engine/torchprof_4b_filter_rank0.pt.trace.json.gz",
     "stock vLLM + optimized driver", "filter, 4B", BLUE),
    ("results/engine/torchprof_4b_filter_ours_rank0.pt.trace.json.gz",
     "ours: plan scheduler, pipelined filter", "filter, 4B", ORANGE),
    ("results/engine/torchprof_stock_map64_rank0.pt.trace.json.gz",
     "stock vLLM + optimized driver", "open-ended map, 4B, cap 64", BLUE),
    ("results/engine/torchprof_ours_map64_rank0.pt.trace.json.gz",
     "ours: plan scheduler, pipelined map", "open-ended map, 4B, cap 64",
     ORANGE),
]

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def gpu_intervals(path):
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    iv = [(e["ts"], e["ts"] + e["dur"]) for e in events
          if e.get("cat") in GPU_CATS and e.get("dur", 0) > 0]
    iv.sort()
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/timeline_strips.png")
    ap.add_argument("--slice-s", type=float, default=0.8,
                    help="width of the drawn zoom slice in seconds")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ROWS), 1, figsize=(11, 5.6))
    fig.suptitle("GPU busy timelines (torch profiler, in-container): "
                 "filters saturate the GPU, maps starve it", fontsize=12)

    for ax, (path, method, op, color) in zip(axes, ROWS):
        merged = gpu_intervals(path)
        t0, t1 = merged[0][0], merged[-1][1]
        span = (t1 - t0) / 1e6
        busy = sum(b - a for a, b in merged) / 1e6
        # zoom into the middle of the window so warmup does not skew
        # the picture; the busy % in the label is the full window
        s0 = t0 + 0.5 * (t1 - t0)
        s1 = s0 + args.slice_s * 1e6
        bars = [((a - s0) / 1e6, (min(b, s1) - a) / 1e6)
                for a, b in merged if b > s0 and a < s1]
        ax.broken_barh(bars, (0, 1), color=color, lw=0)
        ax.set_xlim(0, args.slice_s)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{method}  |  {op}  -  GPU busy "
                     f"{100 * busy / span:.1f}% of the {span:.0f}s window",
                     fontsize=10, loc="left")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        print(f"{method} | {op}: busy {100 * busy / span:.1f}% "
              f"over {span:.1f}s")

    axes[-1].set_xticks([0, args.slice_s / 2, args.slice_s])
    axes[-1].set_xticklabels(["0", f"{args.slice_s / 2:.1f}s",
                              f"{args.slice_s:.1f}s"], fontsize=8)
    axes[-1].set_xlabel(
        f"a {args.slice_s:.1f}-second slice from the middle of each run; "
        "color = a GPU kernel running, white = GPU idle while the CPU "
        "prepares the next step", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
