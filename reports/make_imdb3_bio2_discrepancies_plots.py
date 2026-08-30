"""Plots for the IMDB-3 / BIO-2 discrepancy report.

Pull the inputs from the quail-results volume, then pass the work
directory to this script:

    W=<workdir>
    modal volume get quail-results ablations/discrepancy_imdb3.json $W/imdb3.json
    modal volume get quail-results ablations/discrepancy_bio2.json $W/bio2.json
    modal volume get quail-results ablations/discrepancy_stock_imdb3.json $W/stock_imdb3.json
    modal volume get quail-results ablations/discrepancy_stock_bio2.json $W/stock_bio2.json
    modal volume get quail-results ablations/discrepancy_traces/stock_kineto/ $W/traces/stock_kineto/
    modal volume get quail-results benchmarks/quailb/runs/qb_20260829T185407Z_cbb14b36/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families.json $W/quail.json
    modal volume get quail-results stock_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/stock.json
    modal volume get quail-results pipelined_vllm/20260829T185407Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/summary.json $W/pipelined.json
    modal volume get quail-results sol/sol_quailb_sf0.1.json $W/sol.json
    mkdir -p $W/traces
    for t in imdb3_filter_healthy imdb3_filter_churn imdb3_join \
             bio2_join_early bio2_join_late; do
        modal volume get quail-results \
            ablations/discrepancy_traces/$t.chrome.json.gz \
            $W/traces/$t.chrome.json.gz
    done
    uv run --with matplotlib python \
        reports/make_imdb3_bio2_discrepancies_plots.py $W
"""

import gzip
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
plt.style.use(HERE / "quail.mplstyle")
sys.path.insert(0, str(HERE))
from plot_colors import BLUE, DARK, GRAY, ORANGE, RED, TEAL  # noqa: E402


def load(path):
    with path.open() as f:
        return json.load(f)


def by_query(rows):
    return {r["query"]: r for r in rows}


_KERNEL_EVENT = re.compile(
    rb'"cat": "(?:kernel|gpu_memcpy|gpu_memset)".{0,2400}?'
    rb'"ts": (\d+(?:\.\d+)?)[^,}]*, "dur": (\d+(?:\.\d+)?)', re.S)


def _kernel_intervals(path):
    """Stream (start, end) kernel intervals out of a chrome trace.

    Reads the gzip in chunks with an overlap window, so a trace too
    large for json.load (vLLM's traces carry every CPU op) parses in
    constant memory. Regex fields match kineto's fixed field order.
    """
    evs = []
    overlap = 4096
    tail = b""
    with gzip.open(path) as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            buf = tail + chunk
            for m in _KERNEL_EVENT.finditer(buf):
                ts, dur = float(m.group(1)), float(m.group(2))
                evs.append((ts, ts + dur))
            tail = buf[-overlap:]
    # the overlap can hand the same event to two buffers
    return sorted(set(evs))


_PY_EVENT = re.compile(
    rb'"cat": "python_function", "name": "([^"]{1,200})"'
    rb'.{0,600}?"ts": (\d+(?:\.\d+)?), "dur": (\d+(?:\.\d+)?)', re.S)

# stack frames the CPU attribution sums; every listed substring must
# appear in the frame name (one nesting level each, so the buckets
# subtract cleanly)
FRAME_KEYS = (
    ("busy_loop", (b"run_busy_loop",)),
    ("engine_step", (b"_process_engine_step",)),
    ("input_queue", (b"_process_input_queue",)),
    ("schedule", (b"sched/scheduler.py", b"): schedule")),
    ("cache_probe", (b"kv_cache_coordinator",
                     b"find_longest_cache_hit")),
    ("execute_model", (b"gpu/model_runner", b"execute_model")),
    ("preprocess", (b"preprocess_add_request",)),
    ("block_hash", (b"request_block_hasher",)),
)


def frame_seconds(path):
    """On-stack seconds per FRAME_KEYS bucket, streamed like
    _kernel_intervals; the overlap window can hand an event to two
    buffers, so events dedup on (bucket, start)."""
    seen = {key: set() for key, _ in FRAME_KEYS}
    tail = b""
    with gzip.open(path) as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            buf = tail + chunk
            for m in _PY_EVENT.finditer(buf):
                name = m.group(1)
                ts, dur = float(m.group(2)), float(m.group(3))
                for key, needles in FRAME_KEYS:
                    if all(n in name for n in needles):
                        seen[key].add((ts, dur))
            tail = buf[-4096:]
    return {key: sum(d for _, d in evs) / 1e6
            for key, evs in seen.items()}


def kernel_busy(path):
    """Fraction of a trace window covered by kernel execution.

    Kernel intervals are merged before summing, so overlapping
    streams do not double count.
    """
    evs = _kernel_intervals(path)
    span = max(e for _, e in evs) - min(s for s, _ in evs)
    merged = 0.0
    cur_s, cur_e = evs[0]
    for s, e in evs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged += cur_e - cur_s
            cur_s, cur_e = s, e
    merged += cur_e - cur_s
    busy_us = sum(e - s for s, e in evs)
    return merged / span, len(evs), busy_us / len(evs), merged


def fig_imdb3_timeline(imdb3):
    u = imdb3["unprofiled"]
    filt, join = u["phases"]
    ev0 = u["evictions"][0]["t"]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9.2, 6.2),
                                   sharex=True)

    for p, color in ((filt, BLUE), (join, TEAL)):
        top.scatter([c["t"] for c in p["chunks"]],
                    [c["tokens"] for c in p["chunks"]],
                    s=5, color=color, linewidths=0)
    top.set_yscale("log")
    top.axvline(ev0, color=RED, lw=1, ls="--")
    top.axvline(join["t_start"], color=GRAY, lw=1)
    top.set_ylabel("tokens per forward pass (log scale)")
    top.set_title("IMDB-3: forward-pass size and fresh-token rate, "
                  "instrumented rerun")

    pre = [c["tokens"] for c in filt["chunks"] if c["t"] < ev0]
    post = [c["tokens"] for c in filt["chunks"] if c["t"] >= ev0]
    jt = [c["tokens"] for c in join["chunks"]]
    top.annotate(
        f"{len(pre)} chunks,\nmean {sum(pre) // len(pre):,}",
        (ev0 * 0.45, 25_000), ha="center", color=DARK, fontsize=9)
    top.annotate(
        f"first eviction: retained survivor KV\nhas filled the arena "
        f"({len(post):,} chunks after,\nmean {sum(post) // len(post):,}"
        " tokens each)",
        (ev0 + 1.5, 3_500), color=RED, fontsize=9)
    top.annotate(
        f"join: {len(jt)} chunks,\nmean {sum(jt) // len(jt):,}",
        (join["t_start"] + 8, 20_000), ha="center", color=TEAL,
        fontsize=9)

    # piecewise achieved rate: phase tokens over phase wall, split at
    # the first eviction inside the filter
    t_f0 = filt["t_start"]
    t_j0 = join["t_start"]
    end = t_j0 + join["wall_s"]
    segs = [
        (t_f0, ev0, sum(pre) / (ev0 - t_f0), DARK),
        (ev0, t_f0 + filt["wall_s"],
         sum(post) / (t_f0 + filt["wall_s"] - ev0), RED),
        (t_j0, end, join["tokens"] / join["wall_s"], TEAL),
    ]
    for s, e, rate, color in segs:
        bot.plot([s, e], [rate / 1e3] * 2, color=color, lw=2.2)
        bot.annotate(f"{rate / 1e3:,.0f}k tokens/s",
                     ((s + e) / 2, rate / 1e3 + 7), ha="center",
                     color=color, fontsize=9.5)
    bot.axvline(ev0, color=RED, lw=1, ls="--")
    bot.axvline(t_j0, color=GRAY, lw=1)
    bot.set_ylim(0, 145)
    bot.set_ylabel("fresh tokens per second (thousands)")
    bot.set_xlabel("seconds")
    fig.savefig(OUT / "discrepancy_imdb3_timeline.png", dpi=300)
    plt.close(fig)


def fig_imdb3_composition(imdb3, quail_rows, stock_rows, pipe_rows):
    u = imdb3["unprofiled"]
    filt, join = u["phases"]
    i1 = quail_rows["IMDB-1"]["wall_s"]
    i2 = quail_rows["IMDB-2"]["wall_s"]
    pairs3 = 52_560 / 60_000
    stock_steps = stock_rows["IMDB-3"]["steps"]
    pipe_steps = pipe_rows["IMDB-3"]["steps"]
    rows = [
        ("Quail, IMDB-1 + IMDB-2\nscaled to survivors",
         i1, i2 * pairs3),
        ("Quail measured", filt["wall_s"], join["wall_s"]),
        ("Stock vLLM measured",
         stock_steps[0]["wall_s"], stock_steps[1]["wall_s"]),
        ("Pipelined vLLM measured",
         pipe_steps[0]["wall_s"], pipe_steps[1]["wall_s"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    y = list(range(len(rows)))[::-1]
    for yi, (label, f, j) in zip(y, rows):
        ax.barh(yi, f, color=BLUE, height=0.55)
        ax.barh(yi, j, left=f, color=TEAL, height=0.55)
        ax.annotate(f"filter {f:.1f}", (f / 2, yi), ha="center",
                    va="center", color="white", fontsize=9)
        ax.annotate(f"join {j:.1f}", (f + j / 2, yi), ha="center",
                    va="center", color="white", fontsize=9)
        ax.annotate(f"{f + j:.1f} s", (f + j + 1, yi), va="center",
                    color=DARK, fontsize=9.5)
    recorded = quail_rows["IMDB-3"]["wall_s"]
    ax.plot([recorded, recorded], [y[1] - 0.38, y[1] + 0.38],
            color=DARK, lw=1.2)
    ax.annotate(f"recorded family run: {recorded:.1f} s",
                (recorded + 1, y[1] + 0.34), color=DARK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("seconds")
    ax.set_title("IMDB-3: the composition penalty is Quail's filter "
                 "phase")
    fig.savefig(OUT / "discrepancy_imdb3_composition.png", dpi=300)
    plt.close(fig)


def fig_bio2_rates(bio2, sol, stock_rows, pipe_rows, quail_rows):
    fresh = bio2["unprofiled"]["fresh_tokens"]
    sol_s = sol["queries"]["BIO-2"]["models"]["qwen3-4b-fp8"]["sol_s"]
    quail_s = quail_rows["BIO-2"]["wall_s"]
    stock = stock_rows["BIO-2"]["steps"][0]
    pipe = pipe_rows["BIO-2"]["steps"][0]
    rows = [
        ("SoL estimate", fresh / sol_s, DARK),
        ("Quail", fresh / quail_s, BLUE),
        ("Pipelined vLLM",
         stock["fresh_tokens"] / pipe["wall_s"], GRAY),
        ("Stock vLLM",
         stock["fresh_tokens"] / stock["wall_s"], GRAY),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(rows))
    ax.bar(x, [r[1] / 1e3 for r in rows],
           color=[r[2] for r in rows], width=0.55)
    sol_rate = rows[0][1]
    for xi, (label, rate, _) in zip(x, rows):
        ax.annotate(f"{rate / 1e3:,.1f}k", (xi, rate / 1e3 * 1.10),
                    ha="center", color=DARK, fontsize=9.5)
        if label != "SoL estimate":
            ax.annotate(f"{sol_rate / rate:.1f}x below SoL",
                        (xi, rate / 1e3 * 1.45), ha="center",
                        color=DARK, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("fresh tokens per second (thousands, log scale)")
    ax.set_title("BIO-2: the same 10.4-10.5M fresh tokens at three "
                 "speeds")
    fig.savefig(OUT / "discrepancy_bio2_rates.png", dpi=300)
    plt.close(fig)


def _stock_trace(workdir, cell, label):
    files = next(w["files"] for w in cell["windows"]
                 if w["label"] == label)
    name = next(Path(f).name for f in files
                if f.endswith(".pt.trace.json.gz"))
    return workdir / "traces" / "stock_kineto" / name


def fig_window_busy(workdir, stock3, stockb):
    """Kernel busy per window.

    Quail windows profiled kernel activity only and their walls match
    the unprofiled run within 3.5%, so the raw busy fraction stands.
    vLLM's profiler also traces CPU work, which stretched the stock
    windows 1.1x to 2.4x; each stock bar is therefore corrected to
    the unprofiled run: kernel microseconds per request in the
    window, divided by the unprofiled run's wall per request.
    """
    s3f, s3j = stock3["filter"], stock3["join"]
    w3f, w3j = (next(w for w in stock3["windows"] if w["label"] == l)
                for l in ("stock_imdb3_filter", "stock_imdb3_join"))
    wbj = stockb["windows"][0]
    windows = [
        ("Quail filter,\nfirst chunks", BLUE, None,
         workdir / "traces" / "imdb3_filter_healthy.chrome.json.gz"),
        ("Quail filter,\neviction churn", RED, None,
         workdir / "traces" / "imdb3_filter_churn.chrome.json.gz"),
        ("stock filter", GRAY,
         (w3f["requests"], 1e6 * s3f["wall_s"] / 5000),
         _stock_trace(workdir, stock3, "stock_imdb3_filter")),
        ("Quail join", BLUE, None,
         workdir / "traces" / "imdb3_join.chrome.json.gz"),
        ("stock join", GRAY,
         (w3j["requests"], 1e6 * s3j["wall_s"] / s3j["requests"]),
         _stock_trace(workdir, stock3, "stock_imdb3_join")),
        ("Quail join,\nearly", BLUE, None,
         workdir / "traces" / "bio2_join_early.chrome.json.gz"),
        ("Quail join,\nlate", BLUE, None,
         workdir / "traces" / "bio2_join_late.chrome.json.gz"),
        ("stock join", GRAY,
         (wbj["requests"], 1e6 * stockb["wall_s"] / stockb["pairs"]),
         _stock_trace(workdir, stockb, "stock_bio2_join")),
    ]
    rows = []
    for label, color, correct, path in windows:
        busy, n, mean_us, union_us = kernel_busy(path)
        row = dict(label=label, color=color, raw_busy=busy,
                   kernels=n, mean_kernel_us=mean_us)
        if correct is None:
            row["busy"] = busy
        else:
            requests, unprofiled_us = correct
            row["busy"] = union_us / (requests * unprofiled_us)
        rows.append(row)
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    x = range(len(rows))
    ax.bar(x, [r["busy"] for r in rows],
           color=[r["color"] for r in rows], width=0.6)
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['busy']:.0%}", (xi, r["busy"] + 0.03),
                    ha="center", color=DARK, fontsize=10)
        ax.annotate(f"mean kernel\n{r['mean_kernel_us']:.0f} us",
                    (xi, max(r["busy"] - 0.26, 0.04)), ha="center",
                    color="white" if r["busy"] > 0.32 else DARK,
                    fontsize=7.5)
    for start, end, label in ((0, 2, "IMDB-3 filter"),
                              (3, 4, "IMDB-3 join"),
                              (5, 7, "BIO-2 join")):
        ax.annotate(label, ((start + end) / 2, 1.13), ha="center",
                    color=DARK, fontsize=10)
        if end < len(rows) - 1:
            ax.axvline(end + 0.5, color="#dddddd", lw=0.8)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], fontsize=8.5)
    ax.set_ylabel("kernel time over unprofiled wall (fraction)")
    ax.set_title("GPU busy time at kernel grain; stock bars "
                 "corrected for vLLM profiler overhead\n",
                 fontsize=12)
    fig.savefig(OUT / "discrepancy_window_busy.png", dpi=300)
    plt.close(fig)
    return {f"{r['label']} [{i}]":
            {k: (round(v, 3) if isinstance(v, float) else v)
             for k, v in r.items() if k != "color"}
            for i, r in enumerate(rows)}


def fig_regret(imdb3, bio2, stock3, stockb):
    quail = [imdb3["unprofiled"]["regret_tokens"] / 1e6,
             bio2["unprofiled"]["regret_tokens"] / 1e6]
    stock = [stock3["regret_tokens"] / 1e6,
             stockb["regret_tokens"] / 1e6]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = [0, 1]
    w = 0.32
    ax.bar([xi - w / 2 for xi in x], quail, width=w, color=BLUE)
    ax.bar([xi + w / 2 for xi in x], stock, width=w, color=GRAY)
    for xi, v, off, name in ((0, quail[0], -w / 2, "Quail"),
                             (0, stock[0], w / 2, "stock vLLM"),
                             (1, quail[1], -w / 2, "Quail"),
                             (1, stock[1], w / 2, "stock vLLM")):
        ax.annotate(f"{name}\n{v:,.2f}M", (xi + off, v + 0.04),
                    ha="center", color=DARK, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["IMDB-3", "BIO-2"])
    ax.set_ylim(0, max(quail + stock) * 1.3)
    ax.set_ylabel("KV regret (millions of tokens)")
    ax.set_title("KV regret is nearly equal where it exists at all")
    fig.savefig(OUT / "discrepancy_regret.png", dpi=300)
    plt.close(fig)


def _stock_bio2_trace(workdir, stockb):
    name = Path(stockb["windows"][0]["files"][0]).name
    return workdir / "traces" / "stock_kineto" / name


def _merged(intervals):
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def fig_bio2_strips(workdir, stockb):
    """Kernel activity over two-second excerpts, one strip per
    system: every mark is one kernel-execution interval, gaps are
    GPU idle."""
    quail_evs = _kernel_intervals(
        workdir / "traces" / "bio2_join_late.chrome.json.gz")
    stock_evs = _kernel_intervals(_stock_bio2_trace(workdir, stockb))
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 3.4), sharex=True)
    rows = ((axes[0], quail_evs, BLUE, "white", "Quail"),
            (axes[1], stock_evs, RED, DARK, "stock vLLM"))
    excerpt = 2.0
    for ax, evs, color, text_color, label in rows:
        t0 = min(s for s, _ in evs)
        t1 = max(e for _, e in evs)
        mid = t0 + (t1 - t0 - excerpt * 1e6) / 2
        cut = [(s, e) for s, e in _merged(sorted(evs))
               if e > mid and s < mid + excerpt * 1e6]
        ax.broken_barh(
            [((s - mid) / 1e6, (e - s) / 1e6) for s, e in cut],
            (0, 1), color=color, linewidth=0)
        busy = sum(e - s for s, e in _merged(sorted(evs))) / (t1 - t0)
        ax.set_yticks([])
        ax.set_ylabel(None)
        ax.set_ylim(0, 1)
        ax.text(0.012, 0.5, f"{label} - window {busy:.0%} busy",
                transform=ax.transAxes, va="center",
                color=text_color,
                bbox=(None if text_color == "white" else
                      dict(facecolor="white", alpha=0.75,
                           edgecolor="none")))
        ax.set_xlim(0, excerpt)
    axes[1].set_xlabel("seconds into the excerpt")
    axes[0].set_title("BIO-2: filled while the GPU computes, blank "
                      "while it idles")
    fig.tight_layout()
    fig.savefig(OUT / "discrepancy_bio2_strips.png", dpi=300)
    plt.close(fig)
    return dict(quail_busy=round(
        sum(e - s for s, e in _merged(sorted(quail_evs)))
        / (max(e for _, e in quail_evs)
           - min(s for s, _ in quail_evs)), 4))


def fig_bio2_cpu(workdir, stockb):
    """Where stock's window wall goes: the GPU's kernel time in its
    own lane, and the two CPU threads split exactly by stack frame.

    On-stack seconds include GIL waits and, inside execute_model,
    the wait for the GPU; the two threads run concurrently but share
    the GIL. The GPU lane is device time, not thread time.
    """
    path = _stock_bio2_trace(workdir, stockb)
    fr = frame_seconds(path)
    _, _, _, merged_us = kernel_busy(path)
    kernels = merged_us / 1e6
    gpu = [("running kernels", kernels, TEAL)]
    engine = [
        ("execute_model (launch Python, waiting on the GPU)",
         fr["execute_model"], BLUE),
        ("prefix-cache hit probe", fr["cache_probe"], RED),
        ("scheduler, rest", fr["schedule"] - fr["cache_probe"], GRAY),
        ("step rest (outputs)", fr["engine_step"] - fr["schedule"]
         - fr["execute_model"], DARK),
        ("input queue", fr["input_queue"], ORANGE),
    ]
    inputt = [
        ("prefix-cache block hashing", fr["block_hash"], RED),
        ("request construction, rest",
         fr["preprocess"] - fr["block_hash"], GRAY),
    ]
    lanes = (("GPU", gpu), ("engine thread", engine),
             ("input thread", inputt))
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    for y, (title, parts) in zip((2, 1, 0), lanes):
        x = 0.0
        for name, sec, color in parts:
            ax.barh(y, sec, left=x, color=color, height=0.55)
            if sec > 1.6:
                ax.text(x + sec / 2, y, f"{sec:.1f}",
                        ha="center", va="center", color="white")
            x += sec
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for _, _, c in gpu + engine + inputt[:1]]
    labels = ([n for n, _, _ in gpu + engine]
              + [inputt[0][0]])
    ax.legend(handles, labels, loc="lower right", frameon=False,
              ncol=1, fontsize=8)
    ax.set_yticks([2, 1, 0], labels=[t for t, _ in lanes])
    ax.set_xlabel("seconds in the profiled window (GPU: device "
                  "time; threads: on-stack time)")
    ax.set_xlim(0, fr["busy_loop"] * 1.02)
    ax.set_title("Stock BIO-2: the GPU against the two CPU threads")
    fig.tight_layout()
    fig.savefig(OUT / "discrepancy_bio2_cpu.png", dpi=300)
    plt.close(fig)
    return {k: round(v, 3) for k, v in fr.items()} | dict(
        gpu_kernels_s=round(kernels, 3))


def main():
    workdir = Path(sys.argv[1])
    imdb3 = load(workdir / "imdb3.json")
    bio2 = load(workdir / "bio2.json")
    stock3 = load(workdir / "stock_imdb3.json")
    stockb = load(workdir / "stock_bio2.json")
    sol = load(workdir / "sol.json")
    quail_rows = by_query(
        load(workdir / "quail.json")["passes"]["single"]["queries"])
    stock_rows = by_query(load(workdir / "stock.json")["results"][0])
    pipe_rows = by_query(load(workdir / "pipelined.json")["results"][0])

    OUT.mkdir(exist_ok=True)
    fig_imdb3_timeline(imdb3)
    fig_imdb3_composition(imdb3, quail_rows, stock_rows, pipe_rows)
    fig_bio2_rates(bio2, sol, stock_rows, pipe_rows, quail_rows)
    fig_regret(imdb3, bio2, stock3, stockb)
    busy = fig_window_busy(workdir, stock3, stockb)
    strips = fig_bio2_strips(workdir, stockb)
    cpu = fig_bio2_cpu(workdir, stockb)

    # ---- derived numbers the report cites
    u3 = imdb3["unprofiled"]
    filt, join = u3["phases"]
    ev0 = u3["evictions"][0]["t"]
    pre = [c["tokens"] for c in filt["chunks"] if c["t"] < ev0]
    post = [c["tokens"] for c in filt["chunks"] if c["t"] >= ev0]
    i1, i2 = quail_rows["IMDB-1"]["wall_s"], quail_rows["IMDB-2"]["wall_s"]
    parts = i1 + i2 * 52_560 / 60_000
    sol3 = sol["queries"]["IMDB-3"]["models"]["qwen3-4b-fp8"]["sol_s"]
    solb = sol["queries"]["BIO-2"]["models"]["qwen3-4b-fp8"]["sol_s"]
    ub = bio2["unprofiled"]
    print(json.dumps(dict(
        imdb3=dict(
            recorded_wall_s=quail_rows["IMDB-3"]["wall_s"],
            rerun_wall_s=u3["engine_wall_s"],
            parts_prediction_s=round(parts, 2),
            filter=dict(
                wall_s=filt["wall_s"], gpu_event_s=filt["gpu_s"],
                chunks=filt["n_chunks"],
                first_eviction_t=ev0,
                chunks_before=len(pre),
                mean_tokens_before=sum(pre) // len(pre),
                chunks_after=len(post),
                mean_tokens_after=sum(post) // len(post),
                tokens_per_s=round(filt["tokens"] / filt["wall_s"]),
                launch_cpu_s=filt["timing"]["forward_launch"]),
            join=dict(
                wall_s=join["wall_s"], chunks=join["n_chunks"],
                tokens_per_s=round(join["tokens"] / join["wall_s"]),
                kv=join["kv"]),
            evict_calls=len(u3["evictions"]),
            keys_evicted=sum(e["keys_evicted"]
                             for e in u3["evictions"]),
            regret_tokens=u3["regret_tokens"],
            sol_multiple_recorded=round(
                quail_rows["IMDB-3"]["wall_s"] / sol3, 2),
            sol_multiple_parts=round(parts / sol3, 2)),
        bio2=dict(
            recorded_wall_s=quail_rows["BIO-2"]["wall_s"],
            rerun_wall_s=ub["engine_wall_s"],
            chunks=ub["phases"][0]["n_chunks"],
            mean_chunk_tokens=ub["phases"][0]["tokens"]
            // ub["phases"][0]["n_chunks"],
            regret_tokens=ub["regret_tokens"],
            quail_sol_multiple=round(
                quail_rows["BIO-2"]["wall_s"] / solb, 2),
            stock_sol_multiple=round(
                stock_rows["BIO-2"]["steps"][0]["wall_s"] / solb, 2),
            stock_cache_rate=round(
                stock_rows["BIO-2"]["steps"][0]["cached_tokens"]
                / stock_rows["BIO-2"]["steps"][0]["prompt_tokens"],
                4)),
        stock_cell=dict(
            imdb3=dict(
                filter_wall_s=stock3["filter"]["wall_s"],
                recorded_filter_wall_s=round(
                    stock_rows["IMDB-3"]["steps"][0]["wall_s"], 2),
                join_wall_s=stock3["join"]["wall_s"],
                recorded_join_wall_s=round(
                    stock_rows["IMDB-3"]["steps"][1]["wall_s"], 2),
                regret_tokens=stock3["regret_tokens"],
                buckets=stock3["join"]["buckets"]),
            bio2=dict(
                reports_measured=stockb["reports_measured"],
                ms_per_pair=stockb["ms_per_pair"],
                recorded_ms_per_pair=round(
                    1e3 * stock_rows["BIO-2"]["steps"][0]["wall_s"]
                    / stock_rows["BIO-2"]["steps"][0]["n_pairs"], 3),
                regret_tokens=stockb["regret_tokens"],
                buckets=stockb["buckets"])),
        window_busy=busy, bio2_strips=strips,
        bio2_cpu_attribution=cpu), indent=1))


if __name__ == "__main__":
    main()
