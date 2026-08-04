#!/usr/bin/env python3
"""Solve the expected-flow LP per method and convert its rates into a
validated finite 10,000-document execution (revised paper secs. 'The
steady-state expected-flow LP' and 'From the LP to a finite execution').

Reports, per model x device x first-stage selectivity x method:
  N/lambda*   the asymptotic fluid LP latency target (restricted model,
              exact empirical length types),
  T_construct the validated finite replay latency, with fill/core/repair/
              drain whole-batch phase shares,
  LB_res      the certified resource lower bound,
and the signed replay deviation T_construct - N/lambda*.
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

from docengine.configs import DEVICES, MODELS                     # noqa: E402
from docengine.instance import Instance, sample_outcomes          # noqa: E402
from docengine.lb import resource_lb                              # noqa: E402
from docengine.validator.check import validate                    # noqa: E402

from attic.theory.optimizer.state_actions import (build_fullspec,  # noqa: E402
                                                  build_pipeline,
                                                  build_taskfirst,
                                                  make_types)
from attic.theory.optimizer.steady_state_lp import solve_expected_flow  # noqa: E402
from attic.theory.replay import replay                            # noqa: E402

S_GRID = (0.10, 0.50, 0.90)
P = (50, 50)
OUTCOME_SEED = 20260801


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="workloads/documents.parquet")
    ap.add_argument("--out", default="results/lp_two_stage.csv")
    args = ap.parse_args()

    d = tuple(int(x) for x in pd.read_parquet(args.workload)["d"])
    N = len(d)
    types = make_types(d, 0, rep="exact")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for s1 in S_GRID:
        for mname, model in MODELS.items():
            for dname, device in DEVICES.items():
                inst = Instance(model=model, device=device, d=d, p=P,
                                s=(s1, 0.5), delta=256)
                rng = np.random.default_rng(OUTCOME_SEED + int(s1 * 100))
                X = sample_outcomes(inst, rng)
                builders = {
                    "task": (lambda: build_taskfirst(inst, types), "task", 1, "task"),
                    "pipe": (lambda: build_pipeline(inst, types), "pipe", 1, "pipe"),
                    "fullspec": (lambda: build_fullspec(inst, types), "spec", 2,
                                 "fullspec"),
                }
                for name, (build, vpolicy, kmax, lbpol) in builders.items():
                    mm = build()
                    lp = solve_expected_flow(mm)
                    assert lp.status == "optimal", (name, lp.status)
                    recs, phases = replay(inst, X, mm, lp)
                    errs = validate(inst, vpolicy, recs, X, kmax=kmax)
                    assert errs == [], (name, mname, dname, s1, errs[:3])
                    tau_by = {ph: sum(r["tau"] for r in recs
                                      if r["t"] in set(ids))
                              for ph, ids in phases.items()}
                    t_construct = sum(r["tau"] for r in recs)
                    lb = resource_lb(inst, lbpol, X)["LB"]
                    rows.append(dict(
                        s1=s1, model=mname, device=dname, method=name,
                        lp_target=N / lp.lam, t_construct=t_construct,
                        deviation=t_construct - N / lp.lam,
                        LB=lb, batches=len(recs),
                        fill=tau_by["fill"], core=tau_by["core"],
                        repair=tau_by["repair"], drain=tau_by["drain"],
                        lp_actions=lp.meta["n_actions"],
                        gpu_time_residual=lp.residuals["gpu_time"],
                    ))
                print(f"s1={s1} {mname} {dname} done", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["lp_target", "t_construct", "deviation", "LB", "batches"]
    print(df.set_index(["model", "device", "method", "s1"])[cols]
          .round(3).to_string())


if __name__ == "__main__":
    main()
