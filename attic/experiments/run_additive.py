#!/usr/bin/env python3
"""What changes when KV reads and writes are charged as time instead of
overlapped under compute: reprice the validated builder schedules under the
no-overlap rule, across configs, pass rates, and document-length scales.

Sweeps:
  A. Two filters, real lengths: all four model x card pairs, pass rates
     0.1 / 0.5 / 0.9, three policies, both cost rules.
  B. Length scaling: 1x / 2x / 4x / 10x lengths on the loosest and tightest
     pairs at pass rates 0.5 and 0.9, pipeline vs full speculation.
  C. Four filters at 10x lengths: lookahead 1 / 2 / 4 on Qwen3-4B / H100.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

# repo root, so both docengine (installed or not) and attic import
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from docengine.configs import DEVICES, MODELS                      # noqa: E402
from docengine.instance import Instance, sample_outcomes           # noqa: E402
from docengine.sched.blockwise import (schedule_blockwise,          # noqa: E402
                                       schedule_taskfirst)
from docengine.validator.check import validate                      # noqa: E402

from attic.theory.reprice import (reprice_records,                 # noqa: E402
                                  resource_lb_additive)

SEED = 20260801


def one(inst, X, policy, k):
    if policy == "task":
        recs = schedule_taskfirst(inst, X)
        errs = validate(inst, "task", recs, X)
        lbpol = "task"
    else:
        recs = schedule_blockwise(inst, X, k)
        errs = validate(inst, "pipe" if k == 1 else "spec", recs, X, kmax=k)
        lbpol = "fullspec" if k == inst.n else "pipe"
    assert errs == [], (policy, errs[:3])
    out = reprice_records(inst, recs)
    out["lb_add"] = resource_lb_additive(inst, lbpol, X, k=max(k, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="workloads/documents.parquet")
    ap.add_argument("--only", default="ABC")
    args = ap.parse_args()
    d1 = np.array([int(x) for x in pd.read_parquet(args.workload)["d"]])
    rows = []

    if "A" in args.only:
        for mname, dname in (("Qwen3-4B-FP8", "H100-SXM-80GB"),
                             ("Qwen3-4B-FP8", "L40S-48GB"),
                             ("Qwen3-32B-FP8", "H100-SXM-80GB"),
                             ("Qwen3-32B-FP8", "L40S-48GB")):
            for s1 in (0.1, 0.5, 0.9):
                inst = Instance(model=MODELS[mname], device=DEVICES[dname],
                                d=tuple(int(x) for x in d1), p=(50, 50),
                                s=(s1, 0.5), delta=256)
                X = sample_outcomes(inst, np.random.default_rng(
                    SEED + int(100 * s1)))
                for pol, k in (("task", 0), ("pipe", 1), ("fullspec", 2)):
                    r = one(inst, X, pol, k)
                    rows.append(dict(sweep="A", model=mname, device=dname,
                                     scale=1, s=s1, n=2, policy=pol, **r))
                print(f"A {mname} {dname} s={s1}", flush=True)

    if "B" in args.only:
        for mname, dname in (("Qwen3-4B-FP8", "H100-SXM-80GB"),
                             ("Qwen3-32B-FP8", "L40S-48GB")):
            for scale in (1, 2, 4, 10):
                d = tuple(int(x) * scale for x in d1)
                for s1 in (0.5, 0.9):
                    inst = Instance(model=MODELS[mname],
                                    device=DEVICES[dname], d=d, p=(50, 50),
                                    s=(s1, 0.5), delta=256)
                    X = sample_outcomes(inst, np.random.default_rng(
                        SEED + int(100 * s1)))
                    for pol, k in (("pipe", 1), ("fullspec", 2)):
                        r = one(inst, X, pol, k)
                        rows.append(dict(sweep="B", model=mname, device=dname,
                                         scale=scale, s=s1, n=2, policy=pol,
                                         **r))
                print(f"B {mname} scale={scale}", flush=True)

    if "C" in args.only:
        for scale in (1, 10):
            d = tuple(int(x) * scale for x in d1)
            inst = Instance(model=MODELS["Qwen3-4B-FP8"],
                            device=DEVICES["H100-SXM-80GB"], d=d,
                            p=(50,) * 4, s=(0.9,) * 4, delta=256)
            X = sample_outcomes(inst, np.random.default_rng(SEED + 4))
            for k in (1, 2, 4):
                r = one(inst, X, f"k={k}", k)
                rows.append(dict(sweep="C", model="Qwen3-4B-FP8",
                                 device="H100-SXM-80GB", scale=scale, s=0.9,
                                 n=4, policy=f"k={k}", **r))
            print(f"C scale={scale}", flush=True)

    os.makedirs("results", exist_ok=True)
    path = "results/additive_traffic.csv"
    exists = os.path.exists(path) and args.only != "ABC"
    with open(path, "a" if exists else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
