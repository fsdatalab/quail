"""Rebuild every figure in results/plots from banked measurements.

One function per figure, each naming the run it came from. Measured
constants live in MEASURED below, so a re-flight updates one table
rather than hunting through plotting code.

    python plots/make_figures.py            # all of them
    python plots/make_figures.py sweep      # one by name
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/plots"
BLUE, ORANGE, GREEN, YELLOW, PINK, GRAY = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a8a85")

MEASURED = dict(
    # --- batch-size sweep: 10k docs, 1 filter, synchronous API,
    # unprofiled, one container (modal_profiling.py::batchsweep) ---
    sweep=[(512, 60_448), (1024, 39_287), (2048, 76_595), (4096, 84_184),
           (8192, 91_222), (16384, 96_715), (25305, 97_005)],
    sweep_outlier=1024,   # reproducibly slow; excluded from the fit
    # --- the step model fitted to that sweep ---
    a_us_per_token=10.7,
    b_ms_per_step=3.1,
    # --- torch profiler, stock 4B filter, B = 25,305, 15-second
    # window: share of GPU-busy time by kernel class ---
    kernel_mix=[("GEMMs\n(MLP + projections)", 6.51, BLUE),
                ("fp8 quantize/scale", 2.03, ORANGE),
                ("normalization", 1.35, GREEN),
                ("elementwise/activation", 0.74, YELLOW),
                ("attention (QK^T, attn x V)", 0.45, PINK)],
    gpu_busy_filter=0.995,
    # --- ncu speed-of-light, GEMM microbenchmark at prefill shapes,
    # --clock-control none (modal_profiling.py::ncubench) ---
    ncu=[("gate+up GEMM\n25305 x 2560 -> 18432", 93.0, 30.3),
         ("down / QKV / out GEMM\n25305 x 9216 -> 2560", 91.5, 28.0),
         ("large fused GEMM\n(combined)", 93.3, 28.5)],
    # --- filter comparison, 10k docs, 5 filters, same container,
    # 3 reps each (modal_filters.py) ---
    stock_walls=[42.831, 42.773, 43.161],
    rewind_walls=[39.994, 39.828, 39.694],
    # stock's reads come from the client counter, which is correct for
    # separate requests. rewind's come from the scheduler step trace:
    # the client counter cannot see a chain's intermediate prefills
    # (a rewind rewrites prompt_token_ids) and reported 1.143 for
    # three operators doing visibly different work.
    stock_reads=1.228,
    rewind_reads=1.197,
    stock_semaphore=2048,
    budget_tokens=749_782,
    mean_request_tokens=366,
    # --- persisted KV, 315k corpus tokens (modal_persist.py) ---
    recompute_s=4.54,
    restore_s=2.33,
    # --- channel probe (modal_pinprobe.py) ---
    channels=[("PCIe Gen5 spec", 64.0, GRAY),
              ("raw copy, alloc-pinned host memory", 55.4, GREEN),
              ("end to end through vLLM's offload connector", 10.2, BLUE),
              ("raw copy, unpinned host memory", 9.7, YELLOW),
              ("vLLM tiering spec (file-backed, unpinnable here)",
               2.7, ORANGE)],
)

FIGS = {}


def figure(name):
    def wrap(fn):
        FIGS[name] = fn
        return fn
    return wrap


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


# ---------------------------------------------------------------- 1

@figure("doc_lengths_histogram")
def doc_lengths_histogram():
    """The corpus: how long the documents are. Needs the tokenizer,
    so it downloads the IMDB parquet and Qwen's vocabulary once."""
    plt = _plt()
    from tokenizers import Tokenizer

    sys.path.insert(0, "experiments")
    from workload import FLAG_SEED, MODEL, N_FILTERS, SELECTIVITY, build_pool

    rng = np.random.default_rng(FLAG_SEED + 100)
    flags = (rng.random((10_000, N_FILTERS))
             < np.array(SELECTIVITY)[None, :]).astype(int)
    docs = build_pool(10_000)
    bodies = [d + "\n\n[FLAGS] " + " ".join(
        f"FLAG_{j+1}={'YES' if f else 'NO'}" for j, f in enumerate(fl))
        for d, fl in zip(docs, flags)]
    tok = Tokenizer.from_pretrained(MODEL.replace("-FP8", ""))
    lengths = np.array([len(tok.encode(b).ids) for b in bodies])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=80, color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(lengths.mean(), color=ORANGE, ls="--", lw=2,
               label=f"mean = {lengths.mean():.0f} tokens")
    ax.axvline(np.median(lengths), color=GREEN, ls="--", lw=2,
               label=f"median = {np.median(lengths):.0f} tokens")
    ax.set_xlabel("document length (tokens, including the flag line)",
                  fontsize=11)
    ax.set_ylabel("documents", fontsize=11)
    ax.set_title("10,000 IMDB reviews, Qwen 4B tokenizer\n"
                 f"min {lengths.min()}, max {lengths.max()}, "
                 f"total {lengths.sum():,} tokens - 3.4x the KV pool",
                 fontsize=12)
    ax.legend(fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    _save(fig, "doc_lengths_histogram")


# ---------------------------------------------------------------- 2

@figure("roofline_4b")
def roofline_4b():
    """The analytical roofline: where each component sits, and where
    the dense projections cross the ridge."""
    plt = _plt()
    from quail.configs import H100_SXM as D
    from quail.configs import QWEN3_4B_FP8 as M
    from quail import roofline as rf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{M.name} on {D.name}: analytical component roofline\n"
                 f"peak {D.R_D/1e12:,.0f} TFLOP/s, bandwidth "
                 f"{D.BW/1e12:.2f} TB/s, ridge {rf.ridge(D):.0f} FLOP/byte",
                 fontsize=12)

    Bs = np.logspace(1, 5, 300)
    for name, color in (("qkv", BLUE), ("output", GREEN),
                        ("gate_up", ORANGE), ("down", PINK)):
        ai = [rf.projection_intensity(M, int(b))[name] for b in Bs]
        ax1.loglog(Bs, ai, color=color, lw=2, label=name)
    ax1.axhline(rf.ridge(D), color="black", ls="--", lw=1.5,
                label=f"ridge {rf.ridge(D):.0f} FLOP/byte")
    knees = rf.projection_knee(M, D)
    lo = min(v for v in knees.values() if v)
    hi = max(v for v in knees.values() if v)
    ax1.axvspan(lo, hi, color=PINK, alpha=0.15)
    ax1.text(hi * 1.4, 20,
             f"dense knees\n{lo:,.0f}-{hi:,.0f} tokens", fontsize=9,
             color=PINK)
    ax1.set_xlabel("tokens scheduled in the step (B)", fontsize=11)
    ax1.set_ylabel("arithmetic intensity (FLOP/byte)", fontsize=11)
    ax1.set_title("Dense projections saturate near 400 tokens",
                  fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(10, 1e5)

    Ss = np.logspace(1.5, 5, 300)
    B_fix = 25_305
    ax2.loglog(Ss, [rf.attention_time(M, D, B_fix, int(s)) * 1e3
                    for s in Ss], color=ORANGE, lw=2,
               label="attention (reads the KV cache)")
    dense_ms = rf.projection_time(M, D, B_fix) * 1e3
    ax2.axhline(dense_ms, color=BLUE, ls="--", lw=2,
                label=f"dense projections ({dense_ms:.0f} ms)")
    cross = rf.attention_crossover_S(M, D, B_fix)
    ax2.axvline(cross, color=GRAY, ls=":", lw=1)
    ax2.axvline(320, color=GREEN, lw=1.5, alpha=0.8)
    ax2.text(340, dense_ms * 3, "our documents\nS = 320", fontsize=9,
             color=GREEN)
    ax2.text(cross * 1.15, dense_ms * 0.25,
             f"attention overtakes\nat S = {cross:,.0f}", fontsize=9,
             color=GRAY)
    ax2.set_xlabel("context length S (tokens per document)", fontsize=11)
    ax2.set_ylabel("ideal time per step (ms)", fontsize=11)
    ax2.set_title(f"At B = {B_fix:,}: when does KV reading take over?",
                  fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "roofline_4b")


# ---------------------------------------------------------------- 3

@figure("batchsweep_regression")
def batchsweep_regression():
    """The empirical step model, fitted to unprofiled walls."""
    plt = _plt()
    data = MEASURED["sweep"]
    B = np.array([d[0] for d in data], float)
    rate = np.array([d[1] for d in data], float)
    keep = B != MEASURED["sweep_outlier"]

    # 1/throughput = a + b/B is linear in 1/B
    b_coef, a_coef = np.polyfit(1.0 / B[keep], 1.0 / rate[keep], 1)
    ceiling, knee = 1.0 / a_coef, b_coef / a_coef

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Batch-size sweep: 10k documents, 1 filter, synchronous "
                 "API, no profiler\n"
                 f"fit 1/throughput = a + b/B gives a = {a_coef*1e6:.1f} "
                 f"us/token, b = {b_coef*1e3:.1f} ms/step",
                 fontsize=11)

    ax = axes[0]
    ax.semilogx(B[keep], rate[keep] / 1e3, "o", color=BLUE, ms=9, zorder=5)
    ax.semilogx(B[~keep], rate[~keep] / 1e3, "o", color=GRAY, ms=9, zorder=5)
    grid = np.logspace(np.log10(300), np.log10(30_000), 200)
    ax.semilogx(grid, 1e-3 / (a_coef + b_coef / grid), "-", color=ORANGE,
                lw=2, label=f"ceiling {ceiling/1e3:.0f}k tok/s")
    ax.axvline(knee, color=GREEN, ls="--", lw=1.5,
               label=f"knee B = {knee:,.0f}")
    ax.axvline(416, color=GRAY, ls=":", lw=1,
               label="analytical ridge B = 416")
    ax.text(MEASURED["sweep_outlier"], rate[~keep][0] / 1e3 - 6,
            "B=1024\noutlier", ha="center", fontsize=8, color=GRAY)
    ax.set_xlabel("max_num_batched_tokens (B)", fontsize=11)
    ax.set_ylabel("throughput (k tokens/s)", fontsize=11)
    ax.set_title("Throughput vs batch size", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.plot(1e3 / B[keep], 1e6 / rate[keep], "o", color=BLUE, ms=9,
            label="used in fit")
    ax.plot(1e3 / B[~keep], 1e6 / rate[~keep], "o", color=GRAY, ms=9,
            label="excluded")
    xs = np.linspace(0, 1e3 / B.min() * 1.1, 100)
    ax.plot(xs, (a_coef + b_coef * xs / 1e3) * 1e6, "-", color=ORANGE, lw=2)
    ax.set_xlabel("1000 / B", fontsize=11)
    ax.set_ylabel("1/throughput (us/token)", fontsize=11)
    ax.set_title("The fit is linear in 1/B", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[2]
    resid = (rate - 1.0 / (a_coef + b_coef / B)) / rate * 100
    ax.bar(range(len(B)), resid,
           color=[GRAY if not k else BLUE for k in keep])
    ax.set_xticks(range(len(B)))
    ax.set_xticklabels([f"{int(b):,}" for b in B], fontsize=8, rotation=45)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("residual (%)", fontsize=11)
    ax.set_title("Residuals: one line does fit", fontsize=11)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "batchsweep_regression")
    return a_coef, b_coef


# ---------------------------------------------------------------- 4

@figure("filter_timeline_detail")
def filter_timeline_detail():
    """CPU against GPU on the filter workload: no decode gaps, and the
    host work hides behind the GPU. Needs the raw chrome trace."""
    plt = _plt()
    path = "results/engine/torchprof_4b_filter_rank0.pt.trace.json.gz"
    if not os.path.exists(path):
        print(f"skip filter_timeline_detail: {path} not on disk "
              "(re-run modal_profiling.py::torchprof)")
        return
    with gzip.open(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    gpu = [(e["ts"], e["dur"]) for e in events
           if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
           and e.get("dur", 0) > 0]
    cpu = [(e["ts"], e["dur"]) for e in events
           if e.get("cat") == "cuda_runtime" and e.get("dur", 0) > 0]
    t0 = min(s for s, _ in gpu)
    t1 = max(s + d for s, d in gpu)
    mid = (t0 + t1) / 2
    win = 50_000.0                      # 50 ms
    gs = [((s - mid) / 1e3, d / 1e3) for s, d in gpu
          if mid <= s + d and s <= mid + win]
    cs = [((s - mid) / 1e3, d / 1e3) for s, d in cpu
          if mid <= s + d and s <= mid + win]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [1, 1, 1.5]})
    fig.suptitle("Filter workload timeline (torch profiler, stock vLLM, "
                 "4B fp8)\na 50 ms slice from the middle of a 15-second "
                 "steady-state window", fontsize=12)
    axes[0].broken_barh(gs, (0, 1), color=BLUE, lw=0)
    axes[0].set_title("GPU: nearly solid - 99.5% busy, and the white gaps "
                      "are 1-3 us between kernels, not decode stalls",
                      fontsize=10, loc="left")
    axes[1].broken_barh(cs, (0, 1), color=ORANGE, lw=0)
    axes[1].set_title("CPU: kernel launches and syncs - concurrent with "
                      "the GPU, and finished long before it",
                      fontsize=10, loc="left")
    for ax in axes[:2]:
        ax.set_xlim(0, win / 1e3)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks([0.5])
    axes[0].set_yticklabels(["GPU kernels"], fontsize=10)
    axes[1].set_yticks([0.5])
    axes[1].set_yticklabels(["CPU CUDA calls"], fontsize=10)

    ax = axes[2]
    zoom = 3.0
    for s, d in gs:
        if s + d > 20 and s < 20 + zoom:
            ax.broken_barh([(s - 20, d)], (1.2, 0.8), color=BLUE,
                           lw=0.3, edgecolor="white")
    for s, d in cs:
        if s + d > 20 and s < 20 + zoom:
            ax.broken_barh([(s - 20, d)], (0, 0.8), color=ORANGE,
                           lw=0.3, edgecolor="white")
    ax.set_xlim(0, zoom)
    ax.set_ylim(-0.2, 2.2)
    ax.set_yticks([0.4, 1.6])
    ax.set_yticklabels(["CPU", "GPU"], fontsize=10)
    ax.set_xlabel("milliseconds", fontsize=11)
    ax.set_title("Zoomed 3 ms: one block is one kernel", fontsize=10,
                 loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "filter_timeline_detail")


# ---------------------------------------------------------------- 5

@figure("component_decomposition")
def component_decomposition():
    """What the per-token cost a is made of, and how the mix shifts
    with batch size."""
    plt = _plt()
    a_tot = MEASURED["a_us_per_token"]
    b_ms = MEASURED["b_ms_per_step"]
    comps = MEASURED["kernel_mix"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"What costs {a_tot} us per token\n"
                 "torch profiler inside the engine core, stock vLLM, "
                 "4B fp8, B = 25,305", fontsize=12)

    names = [c[0] for c in comps]
    vals = [c[1] for c in comps]
    ax1.barh(range(len(names)), vals, color=[c[2] for c in comps],
             height=0.55)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("us per token", fontsize=11)
    for i, v in enumerate(vals):
        ax1.text(v + 0.08, i, f"{v:.2f} us  ({v / a_tot * 100:.0f}%)",
                 va="center", fontsize=10)
    ax1.text(5.6, 0, "ncu: these kernels run at\n91-93% of the compute "
             "ceiling", va="center", ha="center", fontsize=8.5,
             color="white", weight="bold")
    ax1.set_xlim(0, 9.5)
    ax1.set_title("Per-token cost by component", fontsize=11)

    Bs = np.logspace(np.log10(200), np.log10(30_000), 300)
    bottom = np.zeros_like(Bs)
    for name, a_i, color in comps:
        frac = (a_i * Bs) / (a_tot * Bs + b_ms * 1000) * 100
        ax2.fill_between(Bs, bottom, bottom + frac, color=color, alpha=0.8,
                         label=f"{name.splitlines()[0]}: {a_i:.2f} us/tok")
        bottom += frac
    ax2.fill_between(Bs, bottom, 100, color=GRAY, alpha=0.4,
                     label=f"per-step overhead: {b_ms} ms/step")
    ax2.set_xscale("log")
    ax2.set_xlabel("tokens per step (B)", fontsize=11)
    ax2.set_ylabel("share of step time (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("The mix shifts with batch size", fontsize=11)
    ax2.legend(fontsize=8, loc="lower right")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "component_decomposition")


# ---------------------------------------------------------------- 6

@figure("phi_budget_v2")
def phi_budget_v2():
    """The profiler's per-token split beside ncu's verdict on the
    GEMM kernels themselves."""
    plt = _plt()
    a_tot = MEASURED["a_us_per_token"]
    comps = MEASURED["kernel_mix"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Where each token's {a_tot} us goes, and whether the "
                 "kernels are the problem\nQwen 4B fp8, H100 SXM, "
                 "prefill-only filter workload", fontsize=12)

    names = [c[0] for c in comps]
    vals = [c[1] for c in comps]
    ax1.barh(range(len(names)), vals, color=[c[2] for c in comps],
             height=0.55)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("us per token (torch profiler, B = 25,305)", fontsize=10)
    for i, v in enumerate(vals):
        ax1.text(v + 0.08, i, f"{v:.2f} us ({v / a_tot * 100:.0f}%)",
                 va="center", fontsize=10)
    ax1.set_xlim(0, 9.5)
    ax1.set_title("Per-token cost by component", fontsize=11)

    ncu = MEASURED["ncu"]
    y = np.arange(len(ncu))
    ax2.barh(y - 0.18, [n[1] for n in ncu], height=0.35, color=BLUE,
             label="Compute (SM) utilization")
    ax2.barh(y + 0.18, [n[2] for n in ncu], height=0.35, color=GRAY,
             label="DRAM utilization")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n[0] for n in ncu], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 108)
    ax2.set_xlabel("% of hardware peak (ncu, --clock-control none)",
                   fontsize=10)
    ax2.set_title("The GEMM kernels are near the ceiling", fontsize=11)
    for i, (_n, c, d) in enumerate(ncu):
        ax2.text(c + 1, i - 0.18, f"{c:.1f}%", va="center", fontsize=9,
                 color=BLUE)
        ax2.text(d + 1, i + 0.18, f"{d:.0f}%", va="center", fontsize=9,
                 color=GRAY)
    ax2.legend(fontsize=9, loc="lower right")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, "phi_budget_v2")


# ---------------------------------------------------------------- 7

@figure("rewind_vs_stock_4b")
def rewind_vs_stock_4b():
    """The filter result: KV rewind against stock pipelining, both
    admitted fairly."""
    plt = _plt()
    m = MEASURED
    sw, rw = m["stock_walls"], m["rewind_walls"]
    s_mean, r_mean = float(np.mean(sw)), float(np.mean(rw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KV rewind against stock pipelining: 4B, 10k documents, "
                 "5 gated filters\nsame async API, same container, stock "
                 f"concurrency set from pool capacity ({m['stock_semaphore']}"
                 "), 3 reps each", fontsize=11)

    labels = ["stock pipelining\n(separate requests,\nprefix cache)",
              "KV rewind\n(one living request\nper document)"]
    ax1.bar([0, 1], [s_mean, r_mean], color=[BLUE, ORANGE], width=0.5)
    for w in sw:
        ax1.plot(0, w, "o", color="white", ms=5, zorder=5,
                 markeredgecolor=BLUE, markeredgewidth=1.5)
    for w in rw:
        ax1.plot(1, w, "o", color="white", ms=5, zorder=5,
                 markeredgecolor=ORANGE, markeredgewidth=1.5)
    ax1.text(0, s_mean + 1, f"{s_mean:.1f}s", ha="center", fontsize=12,
             weight="bold")
    ax1.text(1, r_mean + 1, f"{r_mean:.1f}s", ha="center", fontsize=12,
             weight="bold")
    ax1.text(0.5, 34, f"{s_mean / r_mean:.2f}x faster", ha="center",
             fontsize=14, weight="bold", color=ORANGE)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("wall time (seconds)", fontsize=11)
    ax1.set_ylim(0, 50)
    ax1.set_title("Wall time", fontsize=11)

    reads = [m["stock_reads"], m["rewind_reads"]]
    ax2.bar([0, 1], reads, color=[BLUE, ORANGE], width=0.5)
    for i, r in enumerate(reads):
        ax2.text(i, r + 0.012, f"{r:.2f}x", ha="center", fontsize=12,
                 weight="bold")
    ax2.axhline(1.0, color=GRAY, ls=":", lw=1,
                label="1.00x = every token read once")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("corpus read multiplier", fontsize=11)
    ax2.set_ylim(0.9, 1.4)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("How many times the corpus is read", fontsize=11)

    fig.text(0.5, 0.015,
             f"fair concurrency: {m['budget_tokens']:,} token budget / "
             f"{m['mean_request_tokens']} tokens per request = "
             f"{m['stock_semaphore']} documents in flight.\n"
             "Reads count prefill tokens against the 3.2M-token corpus, "
             "so both sides sit above 1.00x: the questions are prefilled "
             "too. Stock's extra 0.03x is block alignment.",
             ha="center", fontsize=8.5, color="#666666")
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    _save(fig, "rewind_vs_stock_4b")


# ---------------------------------------------------------------- 8

@figure("persist_threshold")
def persist_threshold():
    """Restore against recompute, and the bandwidth threshold that
    decides which wins."""
    plt = _plt()
    from quail.configs import H100_SXM as D
    from quail.configs import QWEN3_4B_FP8 as M
    from quail.plan.cost import read_rate

    m = MEASURED
    thresh = M.kappa * read_rate(M, D) / 1e9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Is restoring a corpus cheaper than recomputing it?\n"
                 "restore wins when the channel beats kappa x prefill "
                 f"rate = {M.kappa:,} bytes/token x "
                 f"{read_rate(M, D):,.0f} tokens/s = {thresh:.1f} GB/s",
                 fontsize=12)

    ch = m["channels"]
    y = np.arange(len(ch))
    ax1.barh(y, [c[1] for c in ch], color=[c[2] for c in ch], height=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([c[0] for c in ch], fontsize=9)
    ax1.invert_yaxis()
    for i, c in enumerate(ch):
        ax1.text(c[1] + 0.8, i, f"{c[1]} GB/s", va="center", fontsize=9)
    ax1.axvline(thresh, color="black", ls="--", lw=1.5)
    ax1.text(thresh + 1.2, len(ch) - 0.4,
             f"threshold {thresh:.1f} GB/s\n(4B fp8)", fontsize=8.5)
    ax1.set_xlim(0, 78)
    ax1.set_xlabel("host-to-GPU bandwidth (GB/s)", fontsize=11)
    ax1.set_title("The channel, measured four ways", fontsize=11)

    ax2.barh(0, m["recompute_s"], height=0.5, color=BLUE)
    ax2.barh(0.7, m["restore_s"], height=0.5, color=ORANGE)
    ax2.text(m["recompute_s"] + 0.06, 0, f"{m['recompute_s']}s",
             va="center", fontsize=11)
    ax2.text(m["restore_s"] + 0.06, 0.7,
             f"{m['restore_s']}s  "
             f"({m['recompute_s'] / m['restore_s']:.1f}x faster)",
             va="center", fontsize=11, weight="bold", color=ORANGE)
    ax2.set_yticks([0.35])
    ax2.set_yticklabels(["4B\n315k corpus tokens"], fontsize=10)
    ax2.set_xlim(0, m["recompute_s"] * 1.35)
    ax2.set_xlabel("seconds until the corpus KV is usable again",
                   fontsize=11)
    ax2.set_title("The second query over the same corpus", fontsize=11)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color=BLUE, label="recompute (re-read every document)"),
                        Patch(color=ORANGE, label="restore from host memory")],
               fontsize=9, loc="lower right")

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, "persist_threshold")


# ---------------------------------------------------------------- 9

@figure("restore_vs_recompute")
def restore_vs_recompute():
    """The persist result on its own, for the slide that needs only
    the bars."""
    plt = _plt()
    m = MEASURED
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    fig.suptitle("Persisted KV: the second query over the same corpus",
                 fontsize=12)
    ax.set_title("restore copies saved KV back from host memory; "
                 "recompute re-reads every document through prefill",
                 fontsize=10)
    ax.barh(0, m["recompute_s"], height=0.55, color=BLUE)
    ax.barh(0.75, m["restore_s"], height=0.55, color=ORANGE)
    ax.text(m["recompute_s"] + 0.06, 0, f"{m['recompute_s']}s",
            va="center", fontsize=12)
    ax.text(m["restore_s"] + 0.06, 0.75,
            f"{m['restore_s']}s  "
            f"({m['recompute_s'] / m['restore_s']:.1f}x faster)",
            va="center", fontsize=12, weight="bold", color=ORANGE)
    ax.set_yticks([0.37])
    ax.set_yticklabels(["4B, 315k\ncorpus tokens"], fontsize=10)
    ax.set_xlim(0, m["recompute_s"] * 1.3)
    ax.set_xlabel("seconds until the corpus KV is usable again", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "restore_vs_recompute")


# --------------------------------------------------------------- 10

@figure("rewind_schematic")
def rewind_schematic():
    """What a rewind does to one document's blocks."""
    plt = _plt()
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.6, "KV rewind: one living request per document",
            fontsize=14, weight="bold")

    def row(y, label, blocks, note):
        ax.text(0.2, y + 0.95, label, fontsize=10.5)
        x = 0.3
        for w, color, txt in blocks:
            ax.add_patch(Rectangle((x, y), w, 0.62, facecolor=color,
                                   edgecolor="white", lw=1.5))
            if txt:
                ax.text(x + w / 2, y + 0.31, txt, ha="center",
                        va="center", fontsize=9,
                        color="white" if color != "#e8f4ea" else "#333")
            x += w
        ax.text(x + 0.25, y + 0.31, note, fontsize=8.5, va="center",
                color="#555")

    row(5.0, "1. prefill [document + question 1]; the answer token is "
             "sampled from this pass, so no decode step ever runs",
        [(5.0, BLUE, "document KV (300 tokens)"),
         (1.1, GREEN, "Q1"), (0.35, YELLOW, "A")],
        "one request, one prefill")
    row(3.2, "2. the gate reads the answer in-engine; erase back to the "
             "document boundary",
        [(5.0, BLUE, "document KV - untouched"),
         (1.45, "#e8f4ea", "freed")],
        "by token position, not by block")
    row(1.4, "3. append question 2 onto the same living request",
        [(5.0, BLUE, "document KV - still resident"),
         (1.1, PINK, "Q2")],
        "only 46 new tokens prefill")

    ax.text(0.3, 0.55,
            "The block straddling the boundary keeps its valid slots and "
            "loses only its cache entry, so no token is recomputed. A "
            "prefix cache cannot do this: it matches whole 16-token\n"
            "blocks across separate requests, so the straddling block is "
            "recomputed every stage - and its entries can be evicted "
            "between stages, which is the failure a token budget prevents.",
            fontsize=9, color="#555")
    fig.tight_layout()
    _save(fig, "rewind_schematic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*", help="figure names; default all")
    args = ap.parse_args()
    names = args.only or list(FIGS)
    for n in names:
        match = [k for k in FIGS if k.startswith(n)]
        if not match:
            print(f"no figure matching {n!r}; have: {', '.join(FIGS)}")
            continue
        for k in match:
            FIGS[k]()


if __name__ == "__main__":
    main()
