"""The step-utilization figure: what the engine did in every step.

Reads a step trace (one JSON record per scheduler step, written by
DocEngineScheduler when DOCENGINE_STEPTRACE names a file) and draws
four panels over the run's timeline:

1. Tokens per step, split prefill against decode, with the step
   budget drawn as the capacity line - how full each round ran.
2. Live sequences and unique documents per step - their ratio is the
   filters in flight per document.
3. KV pool occupancy in tokens, against the pool size and the
   admission budget - how much of HBM the run actually used.
4. The gap to the next step and the packing CPU inside it - where
   wall time went when the card was not computing.

A text summary prints alongside: mean tokens per step, the fraction
of underfilled steps, total packing CPU against the span. Those are
the numbers a flight banks in its ledger entry.

Usage:
    python plots/make_plot_steps.py trace.jsonl --out steps.png \
        --step-budget 25305 --budget 750000
"""

import argparse
import json


def load(path):
    # banked traces are gzipped (.jsonl.gz); sliced cells are plain
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(recs, step_budget):
    toks = [r["tokens"] for r in recs]
    span = recs[-1]["t"] - recs[0]["t"] if len(recs) > 1 else 0.0
    sched = sum(r["sched_ms"] for r in recs) / 1e3
    lines = [
        f"{len(recs)} steps over {span:.2f}s, {sum(toks):,} tokens, "
        f"mean {sum(toks) / max(1, len(toks)):,.0f} tokens/step",
        f"packing CPU {sched:.2f}s ({100 * sched / span:.1f}% of span)"
        if span else f"packing CPU {sched:.2f}s",
    ]
    if step_budget:
        under = sum(1 for t in toks if t < step_budget)
        lines.append(f"{under} steps ({100 * under / len(toks):.0f}%) "
                     f"under the {step_budget:,}-token step budget")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="JSONL step trace")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--step-budget", type=int, default=None,
                    help="max_num_batched_tokens the run booted with")
    ap.add_argument("--budget", type=int, default=None,
                    help="the plan's admission budget in tokens")
    ap.add_argument("--title", default=None,
                    help="condition label: model, boot, profile, "
                         "operator, rep")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load(args.trace)
    assert recs, f"no step records in {args.trace}"
    t0 = recs[0]["t"]
    t = [r["t"] - t0 for r in recs]
    prefill = [r["prefill_tokens"] for r in recs]
    decode = [r["tokens"] - r["prefill_tokens"] for r in recs]
    seqs = [r["seqs"] for r in recs]
    docs = [r["unique_docs"] for r in recs]
    kv_used = [r["kv_used_tokens"] for r in recs]
    kv_total = recs[0]["kv_total_tokens"]
    gaps_ms = [1e3 * (t[i + 1] - t[i]) for i in range(len(t) - 1)] + [0.0]
    sched_ms = [r["sched_ms"] for r in recs]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    if args.title:
        fig.suptitle(args.title, fontsize=11)

    ax = axes[0]
    ax.stackplot(t, prefill, decode, labels=("prefill", "decode"),
                 colors=("#4477aa", "#ee6677"), step="post")
    if args.step_budget:
        ax.axhline(args.step_budget, ls="--", c="k", lw=1,
                   label=f"step budget {args.step_budget:,}")
    ax.set_ylabel("tokens / step")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(t, seqs, c="#4477aa", lw=0.8, label="scheduled this step")
    ax.plot(t, docs, c="#228833", lw=0.8, label="unique documents")
    # queue depths, when the trace carries them (added 2026-08-09):
    # running pinned at the cap with scheduled at half of it is the
    # overlapped-scheduling cohort split made visible
    if "running" in recs[0]:
        ax.plot(t, [r["running"] for r in recs], c="#ee6677", lw=0.8,
                label="running (engine)")
        ax.plot(t, [r["waiting"] for r in recs], c="#ccbb44", lw=0.8,
                label="waiting (engine)")
    mean_ratio = (sum(seqs) / max(1, sum(docs)))
    ax.set_ylabel("count / step")
    ax.legend(loc="upper right", fontsize=8,
              title=f"mean {mean_ratio:.2f} filters/doc")

    ax = axes[2]
    ax.plot(t, kv_used, c="#4477aa", lw=1, label="KV in use")
    ax.axhline(kv_total, ls="-", c="k", lw=1,
               label=f"pool {kv_total:,}")
    if args.budget:
        ax.axhline(args.budget, ls="--", c="#ee6677", lw=1,
                   label=f"admission budget {args.budget:,}")
    ax.set_ylabel("KV tokens")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(t, gaps_ms, c="#bbbbbb", lw=0.6, label="gap to next step")
    ax.plot(t, sched_ms, c="#ee6677", lw=0.8, label="packing CPU")
    ax.set_ylabel("ms")
    ax.set_xlabel("seconds since first step")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)

    for line in summarize(recs, args.step_budget):
        print(line)
    out = args.out or args.trace.rsplit(".", 1)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
