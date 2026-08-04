#!/usr/bin/env python3
"""Compare measured H100 makespans against the ideal model for the same
realized workload.

The engine returns, per run: realized document token lengths (document plus
flags line), per-stage prompt token lengths, the model's answers (the
realized outcome matrix the schedule actually followed), per-wave timings,
and cached-token counts. We rebuild the exact instance, run the analytical
builders for the same policy, and report measured seconds, ideal seconds,
their ratio, and the achieved prefill rate against the FP8 ceiling.

Template accounting: block policies add the generated decision token to the
per-stage prompt length. Task-first shares only its task prefix, so its
per-request unshared tail (answer cue plus the generated token) is folded
into the document length for the ideal instance.
"""

import argparse
import gzip
import json
import os

import numpy as np

from docengine.configs import DEVICES, MODELS
from docengine.instance import Instance
from docengine.lb import resource_lb
from docengine.sched.blockwise import (schedule_blockwise,
                                       schedule_taskfirst)

CUE_TOKENS_FALLBACK = 7


def answers_matrix(r):
    X = np.array(r["flags"], dtype=np.int8)      # planted flags
    if r.get("mode", "waves") == "manifest":
        return X          # manifest runs schedule by the planted outcomes
    for key, a in r["answers"].items():
        i, j = map(int, key.split(","))
        X[i][j - 1] = a
    return X


def ideal_for(r):
    model = MODELS["Qwen3-4B-FP8"]
    dev = DEVICES["H100-SXM-80GB"]
    X = answers_matrix(r)
    if r["policy"] == "task":
        cue = (r["p_task"][0] - r["p_tok"][0]
               if r["p_task"][0] > r["p_tok"][0] else CUE_TOKENS_FALLBACK)
        d = tuple(int(x) + cue + 1 for x in r["d_tok"])
        p = tuple(int(x) for x in r["p_task"])
        inst = Instance(model=model, device=dev, d=d, p=p,
                        s=tuple(r["s"]), delta=256)
        recs = schedule_taskfirst(inst, X)
        lb = resource_lb(inst, "task", X)["LB"]
    else:
        d = tuple(int(x) for x in r["d_tok"])
        p = tuple(int(x) + 1 for x in r["p_tok"])
        inst = Instance(model=model, device=dev, d=d, p=p,
                        s=tuple(r["s"]), delta=256)
        recs = schedule_blockwise(inst, X, r["k"])
        lbpol = "fullspec" if r["k"] == r["n"] else "pipe"
        lb = resource_lb(inst, lbpol, X)["LB"]
    return sum(rec["tau"] for rec in recs), lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    opener = (lambda p: gzip.open(p, "rt")) if args.path.endswith(".gz") \
        else open
    data = json.load(opener(args.path))
    rows = []
    print(f"{'n':>2} {'s1':>5} {'policy':>6} {'k':>2} | {'measured':>9} "
          f"{'ideal':>8} {'ratio':>6} | {'tok/s':>8} {'cachehit%':>9} "
          f"{'agree':>6} {'waves':>5}")
    marks = {"manifest": "*", "warm": "~", "client": "+"}
    for r in data["results"]:
        ideal, lb = ideal_for(r)
        total_prompt = sum(w["prompt_tokens"] for w in r["waves"])
        cached = sum(w["cached_tokens"] for w in r["waves"])
        computed = total_prompt - cached
        rate = computed / r["makespan"]
        w1 = r["waves"][0]
        rows.append(dict(n=r["n"], s1=r["s"][0], policy=r["policy"], k=r["k"],
                         mode=r.get("mode", "waves"),
                         measured=r["makespan"], ideal=ideal, lb=lb,
                         ratio=r["makespan"] / ideal, tok_s=rate,
                         cache_hit=cached / max(1, total_prompt),
                         cache_hit_w1=w1["cached_tokens"]
                         / max(1, w1["prompt_tokens"]),
                         agreement=r["answer_agreement"],
                         waves=len(r["waves"])))
        mk = marks.get(r.get("mode", "waves"), "")
        print(f"{r['n']:>2} {r['s'][0]:>5} {(r['policy'] + mk):>7} "
              f"{r['k']:>2} | "
              f"{r['makespan']:>9.2f} {ideal:>8.2f} "
              f"{r['makespan']/ideal:>6.2f} | {rate:>8.0f} "
              f"{100*cached/max(1,total_prompt):>8.1f}% "
              f"{r['answer_agreement']:>6.3f} {len(r['waves']):>5}")
    if args.csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved", args.csv)


if __name__ == "__main__":
    main()
