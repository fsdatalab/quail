#!/usr/bin/env python3
"""Small-N exact study on REAL model/device numbers (Qwen3-4B / H100, real
document lengths from the workload).

This is the regime where scheduling actually matters under tau_0: batches are
small, so the per-batch weight-read floor W_run/BW is comparable to dense
time, and crossing an outcome gate can cost a real batch. The naive
prediction -- speculation wins iff N*p2 < U* = R_D*W_run/(2P*BW) (~295
tokens on H100), i.e. N* ~ 6 -- is REFUTED by the exact DP: the optimal
pipeline STAGGERS documents, holding some back so that survivors' F2
branches ride inside a later document-prefill batch, which hides the outcome
gate behind useful dense work at zero cost. Speculation therefore wins only
when nothing is left to overlap with (N=2 here; generally the query tail).
The constructors do not stagger and run 12-24% above exact for N <= 8.

Exact DP (offline Dijkstra, atomic prefill) for N <= 4; the validated
constructors extend the sweep to N=64. Prints tau in milliseconds.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docengine.configs import DEVICES, MODELS                     # noqa: E402
from docengine.reference.offline import solve_offline                 # noqa: E402
from docengine.instance import Instance, sample_outcomes          # noqa: E402
from docengine.sched.blockwise import (schedule_blockwise,        # noqa: E402
                                       schedule_taskfirst)
from docengine.validator.check import validate                    # noqa: E402

MODEL, DEVICE = MODELS["Qwen3-4B-FP8"], DEVICES["H100-SXM-80GB"]
S1 = 0.5


def make_inst(d):
    return Instance(model=MODEL, device=DEVICE, d=tuple(d), p=(50, 50),
                    s=(S1, 0.5), delta=max(d))     # atomic prefill


def main():
    lengths = pd.read_parquet("workloads/documents.parquet")["d"].to_numpy()
    rng = np.random.default_rng(123)
    pool = rng.choice(lengths, size=64, replace=False).tolist()

    print(f"docs (first 8): {pool[:8]}  |  U* = R_D*W_run/(2P*BW) = "
          f"{DEVICE.R_D * MODEL.W_run / (2 * MODEL.P * DEVICE.BW):.0f} tokens")
    print("\nEXACT offline DP (atomic prefill), tau in ms:")
    print(f"{'N':>3} {'task':>9} {'pipe':>9} {'fullspec':>9}  winner")
    for N in (2, 3, 4):
        d = pool[:N]
        it = make_inst(d)
        X = sample_outcomes(it, np.random.default_rng(1000 + N))
        vals = {}
        for pol, kmax in (("task", 1), ("pipe", 1), ("fullspec", 2)):
            t0 = time.time()
            v, _ = solve_offline(it, pol, X, kmax=kmax, evict_mode="none")
            vals[pol] = v
            assert time.time() - t0 < 300
        win = min(vals, key=vals.get)
        print(f"{N:>3} {vals['task']*1e3:>9.3f} {vals['pipe']*1e3:>9.3f} "
              f"{vals['fullspec']*1e3:>9.3f}  {win}")

    print("\nConstructors (same instances), tau in ms:")
    print(f"{'N':>3} {'task':>9} {'pipe':>9} {'fullspec':>9}  winner")
    for N in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        d = pool[:N]
        it = make_inst(d)
        X = sample_outcomes(it, np.random.default_rng(1000 + N))
        vals = {}
        for pol, k, vp, kmax in (("task", None, "task", 1),
                                 ("pipe", 1, "pipe", 1),
                                 ("fullspec", 2, "spec", 2)):
            recs = (schedule_taskfirst(it, X) if k is None
                    else schedule_blockwise(it, X, k))
            errs = validate(it, vp, recs, X, kmax=kmax)
            assert errs == [], errs[:3]
            vals[pol] = sum(r["tau"] for r in recs)
        win = min(vals, key=vals.get)
        print(f"{N:>3} {vals['task']*1e3:>9.3f} {vals['pipe']*1e3:>9.3f} "
              f"{vals['fullspec']*1e3:>9.3f}  {win}")


if __name__ == "__main__":
    main()
