#!/usr/bin/env python3
"""Solve the N=10,000 two-stage schedules: feasible schedules per policy vs
resource lower bounds (paper sec. 10.5 step 4), on all four model x device
configurations, over a selectivity grid, with coupled outcome scenarios.

Usage:
  python experiments/run_n10k.py [--reps 2] [--out results/n10k_two_stage.csv]
          [--manifest-dir results/manifests]
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# repo root, so docengine imports whether or not it is pip-installed
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from docengine.configs import MODELS, DEVICES                     # noqa: E402
from docengine.instance import Instance, sample_outcomes          # noqa: E402
from docengine.lb import resource_lb                              # noqa: E402
from docengine.sched.blockwise import (schedule_blockwise,        # noqa: E402
                                       schedule_taskfirst)
from docengine.validator.check import validate                    # noqa: E402

S_GRID = (0.10, 0.25, 0.50, 0.75, 0.90)
P = (50, 50)
DELTA = 256
OUTCOME_SEED = 20260801


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="workloads/documents.parquet")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="results/n10k_two_stage.csv")
    ap.add_argument("--manifest-dir", default="results/manifests")
    args = ap.parse_args()

    d = tuple(int(x) for x in pd.read_parquet(args.workload)["d"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)
    rows = []
    for s1 in S_GRID:
        for r in range(args.reps):
            rng = np.random.default_rng(OUTCOME_SEED + 1000 * r + int(s1 * 100))
            for mname, model in MODELS.items():
                for dname, device in DEVICES.items():
                    inst = Instance(model=model, device=device, d=d, p=P,
                                    s=(s1, 0.5), delta=DELTA)
                    X = sample_outcomes(inst, rng) if r or True else None
                    plans = {
                        "task": (schedule_taskfirst(inst, X), "task", 1, "task"),
                        "pipe": (schedule_blockwise(inst, X, 1), "pipe", 1, "pipe"),
                        "fullspec": (schedule_blockwise(inst, X, 2), "spec", 2,
                                     "fullspec"),
                    }
                    for pname, (recs, vpolicy, kmax, lb_policy) in plans.items():
                        t0 = time.time()
                        errs = validate(inst, vpolicy, recs, X, kmax=kmax)
                        assert errs == [], (pname, mname, dname, s1, errs[:3])
                        lb = resource_lb(inst, lb_policy, X)
                        tau_tot = sum(rec["tau"] for rec in recs)
                        rows.append(dict(
                            s1=s1, rep=r, model=mname, device=dname,
                            policy=pname, tau=tau_tot, LB=lb["LB"],
                            gap_pct=100.0 * (tau_tot - lb["LB"]) / lb["LB"],
                            batches=len(recs),
                            U_tot=sum(rec["U"] for rec in recs),
                            peak_resident=max(
                                (rec["M_peak"] - model.W_mem) / model.kappa
                                for rec in recs),
                            validate_s=round(time.time() - t0, 2),
                        ))
                        if s1 == 0.50 and r == 0:
                            path = os.path.join(
                                args.manifest_dir,
                                f"{mname}_{dname}_s50_{pname}.jsonl.gz")
                            with gzip.open(path, "wt") as f:
                                for rec in recs:
                                    f.write(json.dumps(rec) + "\n")
            print(f"s1={s1} rep={r} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    summary = df.groupby(["model", "device", "policy", "s1"]).agg(
        tau=("tau", "mean"), LB=("LB", "mean"), gap_pct=("gap_pct", "mean"),
        batches=("batches", "mean")).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
