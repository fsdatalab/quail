#!/usr/bin/env python3
"""Multi-filter chains, partial speculation, and multi-GPU scaling.

Three sweeps, all analytical, all finite schedules validated:

A. Four filters, partial speculation: per-stage pass rate sweep on the
   loosest (4B/H100) and tightest (32B/L40S) model and card pairs, for
   task-first and lookahead k in {1, 2, 4}. Reports the validated builder
   latency, the resource lower bound, and the expected-flow LP target.
B. Filter-count sweep: n = 1..6 at a fixed 0.8 per-stage pass rate on both
   pairs, task-first and lookahead k in {1, 2, n}.
C. Multi-GPU scaling: 1, 2, 4, 8 H100s as data-parallel replicas
   (documents split by balanced token counts; makespan = slowest card).
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docengine.cluster import lb_multi, run_builder_multi          # noqa: E402
from docengine.configs import DEVICES, MODELS                      # noqa: E402
from docengine.instance import Instance, sample_outcomes           # noqa: E402
from docengine.lb import resource_lb                               # noqa: E402
from docengine.optimizer.state_actions import (build_blockwise_lp,  # noqa: E402
                                               build_taskfirst, make_types)
from docengine.optimizer.steady_state_lp import solve_expected_flow  # noqa: E402
from docengine.sched.blockwise import (schedule_blockwise,          # noqa: E402
                                       schedule_taskfirst)
from docengine.validator.check import validate                      # noqa: E402

SEED = 20260801


def vpol(k, n):
    return "pipe" if k == 1 else "spec"


def run_A(d, out):
    rows = []
    types = make_types(d, 0, rep="exact")
    for mname, dname in (("Qwen3-4B-FP8", "H100-SXM-80GB"),
                         ("Qwen3-32B-FP8", "L40S-48GB")):
        for s_stage in (0.5, 0.65, 0.8, 0.9, 0.95):
            n = 4
            inst = Instance(model=MODELS[mname], device=DEVICES[dname], d=d,
                            p=(50,) * n, s=(s_stage,) * n, delta=256)
            X = sample_outcomes(inst, np.random.default_rng(
                SEED + int(1000 * s_stage)))
            plans = [("task", None)] + [(f"k={k}", k) for k in (1, 2, 4)]
            for name, k in plans:
                if k is None:
                    recs = schedule_taskfirst(inst, X)
                    errs = validate(inst, "task", recs, X)
                    lb = resource_lb(inst, "task", X)["LB"]
                    lp = solve_expected_flow(build_taskfirst(inst, types))
                else:
                    recs = schedule_blockwise(inst, X, k)
                    errs = validate(inst, vpol(k, n), recs, X, kmax=k)
                    lbpol = "fullspec" if k == n else "pipe"
                    lb = resource_lb(inst, lbpol, X)["LB"]
                    lp = solve_expected_flow(
                        build_blockwise_lp(inst, types, k, levels=5))
                assert errs == [], (name, errs[:3])
                assert lp.status == "optimal"
                rows.append(dict(model=mname, device=dname, s=s_stage,
                                 policy=name, tau=sum(r["tau"] for r in recs),
                                 LB=lb, lp_target=len(d) / lp.lam,
                                 batches=len(recs)))
            print(f"A: {mname} {dname} s={s_stage} done", flush=True)
    _write(out, rows)


def run_B(d, out):
    rows = []
    for mname, dname in (("Qwen3-4B-FP8", "H100-SXM-80GB"),
                         ("Qwen3-32B-FP8", "L40S-48GB")):
        for n in (1, 2, 3, 4, 5, 6):
            inst = Instance(model=MODELS[mname], device=DEVICES[dname], d=d,
                            p=(50,) * n, s=(0.8,) * n, delta=256)
            X = sample_outcomes(inst, np.random.default_rng(SEED + n))
            ks = sorted({1, min(2, n), n})
            plans = [("task", None)] + [(f"k={k}", k) for k in ks]
            for name, k in plans:
                if k is None:
                    recs = schedule_taskfirst(inst, X)
                    errs = validate(inst, "task", recs, X)
                else:
                    recs = schedule_blockwise(inst, X, k)
                    errs = validate(inst, vpol(k, n), recs, X, kmax=k)
                assert errs == [], (n, name, errs[:3])
                rows.append(dict(model=mname, device=dname, n=n, policy=name,
                                 k=(0 if k is None else k),
                                 tau=sum(r["tau"] for r in recs),
                                 batches=len(recs)))
            print(f"B: {mname} n={n} done", flush=True)
    _write(out, rows)


def run_C(d, out):
    rows = []
    cases = [("Qwen3-4B-FP8", 2, 0.5), ("Qwen3-32B-FP8", 4, 0.8)]
    for mname, n, s_stage in cases:
        inst = Instance(model=MODELS[mname], device=DEVICES["H100-SXM-80GB"],
                        d=d, p=(50,) * n, s=(s_stage,) * n, delta=256)
        X = sample_outcomes(inst, np.random.default_rng(SEED + 7 * n))
        builders = [
            ("task", lambda sub, Xs: schedule_taskfirst(sub, Xs), "task", 1),
            ("k=1", lambda sub, Xs: schedule_blockwise(sub, Xs, 1), "pipe", 1),
            (f"k={n}", lambda sub, Xs: schedule_blockwise(sub, Xs, n),
             "spec", n),
        ]
        for G in (1, 2, 4, 8):
            for name, fn, vp, kmax in builders:
                mk, per = run_builder_multi(
                    inst, X, G, fn,
                    validator=lambda s_, r_, x_: validate(s_, vp, r_, x_,
                                                          kmax=kmax))
                lb = lb_multi(inst, "task" if name == "task" else
                              ("fullspec" if kmax == n and n > 1 else "pipe"),
                              X, G)
                rows.append(dict(model=mname, n=n, s=s_stage, gpus=G,
                                 policy=name, makespan=mk, LB=lb,
                                 tokens_max=max(p["tokens"] for p in per),
                                 tokens_min=min(p["tokens"] for p in per)))
            print(f"C: {mname} n={n} G={G} done", flush=True)
    _write(out, rows)


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="workloads/documents.parquet")
    ap.add_argument("--only", default="ABC")
    args = ap.parse_args()
    d = tuple(int(x) for x in pd.read_parquet(args.workload)["d"])
    if "A" in args.only:
        run_A(d, "results/multi_filters.csv")
    if "B" in args.only:
        run_B(d, "results/n_sweep.csv")
    if "C" in args.only:
        run_C(d, "results/multi_gpu.csv")
    for f in ("multi_filters", "n_sweep", "multi_gpu"):
        p = f"results/{f}.csv"
        if os.path.exists(p):
            print(f"\n== {f} ==")
            print(pd.read_csv(p).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
