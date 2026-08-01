"""Multiple GPUs as data-parallel replicas.

Each GPU holds a full copy of the model weights (both models fit on one
H100 or L40S at FP8), keeps its own KV cache, and processes its own share
of the documents, so there is no cross-GPU communication term in the ideal
model. Documents are split up front by longest-processing-time assignment
on token counts, which balances the shares; any residual imbalance shows up
honestly in the makespan, which is the maximum over the per-GPU schedules.
The fluid throughput of G replicas is exactly G times the single-GPU rate;
the finite schedules below are built, validated, and priced per GPU.
"""

from typing import Callable, List

import numpy as np

from .instance import Instance
from .lb import resource_lb


def partition_docs(d, G: int) -> List[List[int]]:
    """Longest-processing-time split of document indices into G balanced
    shares by token count."""
    order = sorted(range(len(d)), key=lambda i: -d[i])
    shares = [[] for _ in range(G)]
    loads = [0] * G
    for i in order:
        g = int(np.argmin(loads))
        shares[g].append(i)
        loads[g] += d[i]
    return shares


def sub_instance(inst: Instance, subset: List[int]) -> Instance:
    return Instance(model=inst.model, device=inst.device,
                    d=tuple(inst.d[i] for i in subset), p=inst.p, s=inst.s,
                    delta=inst.delta, max_new_tokens=inst.max_new_tokens,
                    max_seqs=inst.max_seqs)


def run_builder_multi(inst: Instance, X, G: int,
                      builder: Callable, validator: Callable = None):
    """Build one schedule per GPU share; return (makespan, per-GPU details).

    builder(sub_inst, X_sub) -> manifest records.
    validator(sub_inst, records, X_sub) -> error list, checked when given.
    """
    shares = partition_docs(inst.d, G)
    per_gpu = []
    for g, subset in enumerate(shares):
        sub = sub_instance(inst, subset)
        X_sub = np.asarray(X)[subset]
        recs = builder(sub, X_sub)
        if validator is not None:
            errs = validator(sub, recs, X_sub)
            assert errs == [], (g, errs[:3])
        per_gpu.append(dict(gpu=g, docs=len(subset),
                            tokens=sum(sub.d),
                            tau=sum(r["tau"] for r in recs),
                            batches=len(recs)))
    makespan = max(p["tau"] for p in per_gpu)
    return makespan, per_gpu


def lb_multi(inst: Instance, policy: str, X, G: int) -> float:
    """Per-share resource bound; the cluster bound is the max share bound."""
    shares = partition_docs(inst.d, G)
    worst = 0.0
    for subset in shares:
        sub = sub_instance(inst, subset)
        worst = max(worst, resource_lb(sub, policy,
                                       np.asarray(X)[subset])["LB"])
    return worst
