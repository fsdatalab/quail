#!/usr/bin/env python3
"""The analytical crossover map for reasoning filters (phase A).

Sweeps thinking length across the standard configurations and prints,
per cell: the three policy values from the Bellman recurrences, the
optimal composition, and the pipeline's steady throughput ceiling.
Calibrated wall-clock uses the measured 80,000 tokens per second in
place of the 275,000 ceiling (phi = 80/275); both are written to the
CSV, the table prints calibrated seconds.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docengine.configs import DEVICES, MODELS                  # noqa: E402
from docengine.reasoning.lp import lp_throughput               # noqa: E402
from docengine.reasoning.model import (RInstance,              # noqa: E402
                                       best_composition,
                                       value_blockwise, value_taskfirst)

PHI = 80_000.0 / 275_000.0
CONFIGS = [(2, 0.5), (3, 0.9), (4, 0.8), (4, 0.95)]
THINK = [0, 32, 128, 512]


def make(n, s, think, calib):
    return RInstance(model=MODELS["Qwen3-4B-FP8"],
                     device=DEVICES["H100-SXM-80GB"],
                     N=10000, d=313.0, s=(s,) * n, p=(25,) * n,
                     g=(think + 1,) * n, calib=calib)


def main():
    rows = []
    print(f"{'cfg':>10} {'think':>6} | {'task':>8} {'pipeline':>9} "
          f"{'fullspec':>9} {'best':>8} {'K*':>10} | {'lam pipe':>9} "
          f"{'binding':>9}")
    for n, s in CONFIGS:
        for think in THINK:
            it = make(n, s, think, PHI)
            t_task = value_taskfirst(it)["T"]
            t_pipe = value_blockwise(it, (1,) * n)["T"]
            t_spec = value_blockwise(it, (n,))["T"]
            K, vb = best_composition(it)
            lp = lp_throughput(it, (1,) * n)
            rows.append(dict(n=n, s=s, think=think, task=t_task,
                             pipeline=t_pipe, fullspec=t_spec,
                             best=vb["T"], K="".join(map(str, K)),
                             block_docs=vb["block_docs"],
                             lam_pipe=lp["lam"], binding=lp["binding"],
                             ideal_task=value_taskfirst(
                                 make(n, s, think, 1.0))["T"],
                             ideal_pipe=value_blockwise(
                                 make(n, s, think, 1.0), (1,) * n)["T"]))
            print(f"n={n} s={s:<4} {think:>6} | {t_task:>8.1f} "
                  f"{t_pipe:>9.1f} {t_spec:>9.1f} {vb['T']:>8.1f} "
                  f"{str(K):>10} | {lp['lam']:>9.0f} {lp['binding']:>9}")
    os.makedirs("results", exist_ok=True)
    with open("results/reasoning_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved results/reasoning_sweep.csv")


if __name__ == "__main__":
    main()
