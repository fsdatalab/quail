#!/usr/bin/env python3
"""Exact small-N study for reasoning filters: fixed policies against
the adaptive optimum, online against clairvoyant, ample against tight
memory. N=3 documents, three filters; values in seconds (ideal
compute pricing, phi=1)."""

import csv
import itertools
import os

import numpy as np

from docengine.configs import DEVICES, MODELS
from docengine.reasoning.exact import solve_offline, solve_online
from docengine.reasoning.model import RInstance


def make(s, g, cap):
    return RInstance(model=MODELS["Qwen3-4B-FP8"],
                     device=DEVICES["H100-SXM-80GB"],
                     N=3, d=300.0, s=(s,) * 3, p=(25,) * 3, g=(g,) * 3,
                     cap_tokens=cap)


def e_offline(inst):
    """Exact expectation of the clairvoyant optimum over all 512
    outcome scenarios."""
    total = 0.0
    for bits in itertools.product((0, 1), repeat=9):
        X = np.array(bits, dtype=int).reshape(3, 3)
        prob = 1.0
        for i in range(3):
            for j in range(3):
                prob *= inst.s[j] if X[i][j] else (1 - inst.s[j])
        total += prob * solve_offline(inst, X, "opt")
    return total


def main():
    rows = []
    print(f"{'s':>4} {'g':>4} {'cap':>6} | {'task':>7} {'pipe':>7} "
          f"{'spec':>7} {'opt':>7} | {'adaptive gain':>13}")
    for s in (0.5, 0.9):
        for g in (1, 8, 32):
            for cap in (None, 900):
                it = make(s, g, cap)
                vals = dict(task=solve_online(it, "task"),
                            pipe=solve_online(it, (1, 1, 1)),
                            spec=solve_online(it, (3,)),
                            opt=solve_online(it, "opt"))
                gain = 1.0 - vals["opt"] / min(vals["pipe"], vals["spec"])
                rows.append(dict(s=s, g=g, cap=cap or 0, **vals,
                                 adaptive_gain=gain))
                print(f"{s:>4} {g:>4} {cap or 'ample':>6} | "
                      f"{vals['task']:>7.3f} {vals['pipe']:>7.3f} "
                      f"{vals['spec']:>7.3f} {vals['opt']:>7.3f} | "
                      f"{100 * gain:>12.1f}%")
    for g in (1, 32):
        it = make(0.9, g, None)
        eo = e_offline(it)
        on = solve_online(it, "opt")
        print(f"clairvoyance s=0.9 g={g}: E[offline]={eo:.3f} "
              f"online={on:.3f} gap={100 * (on / eo - 1):.1f}%")
        rows.append(dict(s=0.9, g=g, cap=0, task=0, pipe=0, spec=0,
                         opt=on, adaptive_gain=on / eo - 1))
    os.makedirs("results", exist_ok=True)
    with open("results/reasoning_exact.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved results/reasoning_exact.csv")


if __name__ == "__main__":
    main()
