"""Derive the committed touch-pass timeline summary from raw py-spy
speedscope files.

Raw files live on the quail-results volume; pull them first:

    uv run modal volume get quail-results \\
        boot/pyspy_<tag>.speedscope.json <dir>

Then:

    python reports/make_boot_timeline_data.py \\
        --model qwen3-4b-fp8 --dir <dir> \\
        --trial touch0=8.94 --trial touch1=7.51 --trial touch2=8.31

Each --trial is <speedscope tag>=<warm_kernels_s from the boot row>:
the warm phase is the last warm_kernels_s seconds of the MainThread
profile (the profiler stops right after the boot returns). The first
named trial also gets a 0.1 s binned timeline. Output merges into
results/boot_touch_timeline.json under the model key.
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[0] / "results"
OUT = RESULTS / "boot_touch_timeline.json"
BIN = 0.1

# Both entries in the first rule are the CPU waiting for the GPU to
# finish a warm chunk: result() waits on the answer event, and
# submit()'s pinned-buffer allocation is a CUDA call that stalls
# while the device is busy. One category, because they are one thing.
RULES = [
    ("waiting for the GPU (warm chunk running)",
     lambda n, f: (n == "synchronize" and "cuda" in f)
     or (n == "submit" and "executor/loop.py" in f)),
    ("gemm + quant kernel launches",
     lambda n, f: n in ("gemm", "quant", "fp8_gemm_nt",
                        "per_token_group_quant_fp8")
     or ("torch/_ops.py" in f and n == "__call__")),
    ("attention + triton kernel launches",
     lambda n, f: "executor/attention.py" in f or "triton" in f
     or n == "dynamic_func" or "executor/arena.py" in f),
    ("deepgemm cache reads (disk)",
     lambda n, f: n in ("read_text", "read_bytes", "load", "loads",
                        "exists") or "deep_gemm" in f),
    ("packing + admission (cpu)",
     lambda n, f: ("executor/loop.py" in f
                   and n in ("_staged", "pack_chunk"))
     or "executor/pack.py" in f),
]
CATS = [c for c, _ in RULES] + ["other"]


def classify(frames, stack):
    for fi in reversed(stack):          # leaf first
        fr = frames[fi]
        n, f = fr.get("name", ""), str(fr.get("file", ""))
        for cat, rule in RULES:
            if rule(n, f):
                return cat
    return "other"


def derive(path, tail, with_timeline):
    data = json.load(open(path))
    frames = data["shared"]["frames"]
    prof = next(p for p in data["profiles"]
                if "MainThread" in p.get("name", ""))
    t, times = 0.0, []
    for w in prof["weights"]:
        t += w
        times.append(t)
    start = t - tail
    tot = {c: 0.0 for c in CATS}
    nbins = int(tail / BIN) + 1
    mat = {c: [0.0] * nbins for c in CATS} if with_timeline else None
    for stack, w, te in zip(prof["samples"], prof["weights"], times):
        if te < start or not stack:
            continue
        c = classify(frames, stack)
        tot[c] += w
        if mat is not None:
            b = min(int((te - start) / BIN), nbins - 1)
            mat[c][b] += w
    tot = {c: round(v, 3) for c, v in tot.items()}
    timeline = None
    if mat is not None:
        timeline = dict(bin_s=BIN, n_bins=nbins, categories=CATS,
                        matrix={c: [round(v, 4) for v in mat[c]]
                                for c in CATS})
    return tot, timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--trial", action="append", required=True,
                    help="<speedscope tag>=<warm_kernels_s>")
    args = ap.parse_args()

    totals, timeline = {}, None
    for i, spec in enumerate(args.trial):
        tag, tail = spec.split("=")
        tot, tl = derive(
            Path(args.dir) / f"pyspy_{tag}.speedscope.json",
            float(tail), with_timeline=(i == 0))
        totals[tag] = tot
        if tl is not None:
            timeline = dict(trial=tag, **tl)

    merged = dict(cell="boot_touch_timeline", models={})
    if OUT.exists():
        merged = json.load(open(OUT))
        merged.setdefault("models", {})
    merged["source"] = (
        "py-spy MainThread samples, warm-phase tail of touch boots "
        "(profiler attached: it stretches CPU-side launch work more "
        "than GPU waits - use for structure, not walls); raw on the "
        "quail-results volume: /results/boot/pyspy_*.speedscope.json")
    merged["models"][args.model] = dict(
        totals_per_trial=totals, timeline=timeline)
    json.dump(merged, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} [{args.model}]")
    for tag, tot in totals.items():
        top = sorted(tot.items(), key=lambda kv: -kv[1])[:4]
        print(f"  {tag}: " + ", ".join(f"{c}={v}" for c, v in top))


if __name__ == "__main__":
    main()
