#!/usr/bin/env python3
"""Phase B analytical map: long documents, where memory binds.

Two workloads with the corpus mass held near three million tokens:
one hundred documents of thirty thousand tokens, and thirty documents
of one hundred thousand. Filters are answer-only (g=1) or forced
thinking (g=257, the cost instrument: generation length is controlled
exactly, the answer is the first token). Fluid values and throughput
for the fixed policies and the best composition, plus the exact
adaptive-versus-fixed preview at a contested memory ratio (N=4
documents, capacity 2.5 document footprints). Calibrated seconds."""

import csv
import os

from docengine.configs import DEVICES, MODELS
from docengine.reasoning.exact import solve_online
from docengine.reasoning.lp import lp_throughput
from docengine.reasoning.model import (RInstance,
                                       best_composition,
                                       value_blockwise, value_taskfirst)

PHI = 80_000.0 / 275_000.0


def make(N, d, n, s, g, calib=PHI, cap=None):
    return RInstance(model=MODELS["Qwen3-4B-FP8"],
                     device=DEVICES["H100-SXM-80GB"],
                     N=N, d=float(d), s=(s,) * n, p=(25,) * n, g=(g,) * n,
                     calib=calib, cap_tokens=cap)


def main():
    rows = []
    print(f"{'workload':>16} {'g':>4} | {'task':>7} {'pipe':>7} "
          f"{'spec':>7} {'best':>7} {'K*':>7} {'B':>6} | {'lam pipe':>8}")
    for N, d in ((100, 30_000), (30, 100_000)):
        for g in (1, 257):
            it = make(N, d, 2, 0.7, g)
            t_task = value_taskfirst(it)["T"]
            t_pipe = value_blockwise(it, (1, 1))["T"]
            t_spec = value_blockwise(it, (2,))["T"]
            K, vb = best_composition(it)
            lam = lp_throughput(it, (1, 1))
            rows.append(dict(N=N, d=d, g=g, task=t_task, pipe=t_pipe,
                             spec=t_spec, best=vb["T"],
                             K="".join(map(str, K)),
                             block_docs=vb["block_docs"],
                             lam_pipe=lam["lam"], binding=lam["binding"]))
            print(f"{N:>5} x {d:>8} {g:>4} | {t_task:>7.1f} {t_pipe:>7.1f} "
                  f"{t_spec:>7.1f} {vb['T']:>7.1f} {str(K):>7} "
                  f"{vb['block_docs']:>6.1f} | {lam['lam']:>8.2f}")

    print("\nexact adaptive preview, N=4 docs of 30k, capacity 2.5 "
          "footprints:")
    for g in (1, 33, 257):
        foot = 30_000 + 2 * (25 + g)
        it = make(4, 30_000, 2, 0.7, g, calib=1.0,
                  cap=int(2.5 * foot))
        vals = dict(pipe=solve_online(it, (1, 1)),
                    spec=solve_online(it, (2,)),
                    opt=solve_online(it, "opt"))
        gain = 1.0 - vals["opt"] / min(vals["pipe"], vals["spec"])
        rows.append(dict(N=4, d=30_000, g=g, task=0, pipe=vals["pipe"],
                         spec=vals["spec"], best=vals["opt"], K="opt",
                         block_docs=0, lam_pipe=0, binding="exact"))
        print(f"  g={g:>4}: pipe {vals['pipe']:.2f}  spec "
              f"{vals['spec']:.2f}  adaptive {vals['opt']:.2f}  "
              f"gain {100 * gain:.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/longdoc_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved results/longdoc_map.csv")


if __name__ == "__main__":
    main()
