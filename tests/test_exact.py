"""Checks for the live cost-model and instance layers, plus the toy
model/device helpers shared by the test files. The verification battery
for the exact reference solvers lives in attic/tests/test_exact_reference.py
(same toy helpers, duplicated there so neither suite reaches across
directories)."""

import random

import numpy as np

from docengine.configs import DeviceConfig, ModelConfig
from docengine.costmodel import a_pairs
from docengine.instance import Instance, survival

# A tiny synthetic model/device so that memory pressure and weight reads are
# exercised at hand-checkable scale. kappa = 2*2*1*2*1 = 8 bytes/token.
TOY_MODEL = ModelConfig(name="toy", P=1e6, L=2, h=4, n_q=2, n_kv=1, d_h=2,
                        L_ctx=64, q_kv=1, W_mem=1_000.0, W_run=1_000.0)


def toy_device(kv_tokens: int) -> DeviceConfig:
    """Device whose free-KV capacity is exactly kv_tokens."""
    M = TOY_MODEL.W_mem + 100.0 + kv_tokens * TOY_MODEL.kappa
    return DeviceConfig(name=f"toy-{kv_tokens}", M=M, BW=1e6, R_D=1e9, R_A=1e9,
                        S=100.0)


def inst(d, p, s, kv_tokens=10_000, delta=1, cap=None, model=TOY_MODEL):
    return Instance(model=model, device=toy_device(kv_tokens), d=tuple(d),
                    p=tuple(p), s=tuple(s), delta=delta, max_new_tokens=cap)


def test_chunk_attention_invariance():
    rng = random.Random(0)
    for _ in range(200):
        d = rng.randint(1, 40)
        cuts = sorted(rng.sample(range(1, d), rng.randint(0, min(5, d - 1))))
        parts, prev = [], 0
        for c in cuts + [d]:
            parts.append(c - prev)
            prev = c
        total, c = 0, 0
        for q in parts:
            total += a_pairs(c, q)
            c += q
        assert total == d * (d + 1) // 2  # eq. (16)


def test_survival_matrix():
    X = np.array([[1, 1, 0], [0, 1, 1]])
    Y = survival(X)
    assert Y.tolist() == [[1, 1, 1], [1, 0, 0]]
