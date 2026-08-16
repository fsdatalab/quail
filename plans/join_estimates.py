"""Every number in plans/join_plan.md, derived. Run:

    python plans/join_estimates.py

Measured constants are imported from quail/plan/cost.py (the one
table). Three inputs are not in cost.py and carry their provenance
here: the packed rate (plans/packed_forward.md, the round-6 ladder),
the raw cached-read prices per suffix width (plans/cost_model.md
Section 4), and the KV pool size (README, the shipped boot). The
document lengths are assumptions until step 1 of the plan tokenizes
the real data; change them here and the plan's numbers move with
them.

Attention accounting: a prefill of h tokens costs a2*h^2 on top of
linear work, and causal pairs(h) = h(h+1)/2 ~ h^2/2, so the price
per attention pair is 2*a2. Both sustained rates were measured on
the calibration corpus, whose profile is CAL_SQ_PER_TOKEN squared
tokens per token — the rates already embed that much attention per
token, so each estimate charges only its excess over the profile
(the same centering rule cost.py uses).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quail.plan.cost import (
    ALPHA2_S_PER_TOKEN2 as A2,
    CAL_SQ_PER_TOKEN as SQ_CAL,
    STEP_BETA_N_S as BETA_N,
    STEP_FIXED_S,
    STEP_TOKEN_S as A_ENG,
)

A_PKD = 1.0 / 121045.0        # s/token, packed pipeline (packed_forward.md)
S = 25305                     # step/chunk token budget (batch sweep best)
POOL = 946800                 # KV pool tokens at the shipped boot (README)
READ_NS = {16: 82.0, 32: 104.0, 64: 141.0}   # ns per cached token at
#                               suffix width c (cost_model.md 4, raw slopes)
THRASH = 1.87                 # measured wall multiplier, default admission
#                               at 3.4x pool pressure (README filter result)

NL, NR = 8103, 3718           # reports, terms (FDJ paper, Table 1)
TL, TR = 1000, 8              # assumed doc tokens - step 1 replaces these
p, q = 40, 50                 # assumed preamble / question tail

P = NL * NR
f, s = p + TL, TR + q


def read_price(c):
    """Per-cached-token price at suffix width c, interpolated between
    the measured widths. Outside 16..64 there is no measurement; the
    plan flags c or h outside the calibrated grid wherever used."""
    ks = sorted(READ_NS)
    lo = max(k for k in ks if k <= c)
    hi = min(k for k in ks if k >= c)
    if lo == hi:
        return READ_NS[lo] * 1e-9
    w = (c - lo) / (hi - lo)
    return (READ_NS[lo] * (1 - w) + READ_NS[hi] * w) * 1e-9


def attn_excess_s(tokens, sq_per_token):
    return tokens * (sq_per_token - SQ_CAL) * A2


def hrs(x):
    return x / 3600.0


print(f"pairs P = {NL} x {NR} = {P:,}")
print(f"f = {p}+{TL} = {f}; s = {TR}+{q} = {s}")
print(f"terms side, fully resident = NR*(p+TR) = {NR * (p + TR):,} tokens "
      f"= {NR * (p + TR) / POOL:.0%} of the pool")

k = (S - f) // s
m = math.ceil(NR / k)
print(f"\nchunk geometry: k = ({S}-{f})//{s} = {k} suffixes/chunk; "
      f"m = ceil({NR}/{k}) = {m} chunks/report; waste bound f/S = {f / S:.1%}")

# A1 - stock vLLM, one request per pair, arbitrary order: no reuse.
h_pair = p + TL + TR + q
fresh = P * h_pair
t_lin = fresh * A_ENG
t_attn = attn_excess_s(fresh, h_pair)
t_req = P * BETA_N
t_step = fresh / S * STEP_FIXED_S
a1 = t_lin + t_attn + t_req + t_step
print(f"\nA1 stock, arbitrary order: fresh = P*{h_pair} = {fresh / 1e9:.2f}B")
print(f"  linear {hrs(t_lin):.1f} h + attn excess {hrs(t_attn):.1f} h + "
      f"requests {hrs(t_req):.2f} h + steps {hrs(t_step):.2f} h = "
      f"{hrs(a1):.1f} h; x{THRASH} thrash = {hrs(a1 * THRASH):.0f} h")
print(f"  prefix working set NL*f = {NL * f / 1e6:.1f}M tokens = "
      f"{NL * f / POOL:.1f}x pool (why thrash applies)")

# B - engine chain mode. A2-grouped-stock = B + per-pair requests + the
# 16-token boundary block recomputed per pair.
fresh_b = NL * f + P * s
t_lin = fresh_b * A_ENG
t_attn = attn_excess_s(NL * f, f) + attn_excess_s(P * s, s)
rd = read_price(s)
t_read = P * f * rd
t_step = fresh_b / S * STEP_FIXED_S
b = t_lin + t_attn + t_read + t_step + NL * BETA_N
print(f"\nB engine chains: fresh = NL*f + P*s = {NL * f / 1e6:.1f}M + "
      f"{P * s / 1e9:.3f}B = {fresh_b / 1e9:.3f}B -> {hrs(t_lin):.2f} h")
print(f"  reads P*f*{rd * 1e9:.0f}ns (interp c=32:104 / c=64:141 at c={s}) "
      f"= {hrs(t_read):.2f} h  [c=32/c=64 endpoints: "
      f"{hrs(P * f * READ_NS[32] * 1e-9):.2f}/{hrs(P * f * READ_NS[64] * 1e-9):.2f} h; "
      f"h={f} is below the 2,048-min calibrated grid]")
print(f"  attn centering {t_attn:+.0f} s; steps {t_step:.0f} s")
print(f"  B = {hrs(b):.1f} h")

boundary = P * 8
a2_plan = b + P * BETA_N + boundary * (A_ENG + STEP_FIXED_S / S)
print(f"A2 grouped stock = B + requests {hrs(P * BETA_N):.2f} h + boundary "
      f"{boundary / 1e6:.0f}M tok {hrs(boundary * A_ENG):.2f} h = "
      f"{hrs(a2_plan):.1f} h")

# C - packed forward pass, prefix recomputed per chunk. Attention is
# charged in full from pair counts (cross, within-suffix, prefix-self)
# because the chunk shape is nothing like the calibration corpus.
fresh_c = P * s + NL * m * f
pairs = (P * s * f
         + P * s * (s + 1) // 2
         + NL * m * f * (f + 1) // 2)
t_non = fresh_c * (A_PKD - SQ_CAL * A2)
t_att = 2 * pairs * A2
c = t_non + t_att
print(f"\nC packed, recompute: fresh = P*s + NL*m*f = {P * s / 1e9:.3f}B + "
      f"{NL * m * f / 1e6:.1f}M = {fresh_c / 1e9:.3f}B")
print(f"  attention pairs {pairs / 1e12:.2f}e12 (cross {P * s * f / 1e12:.2f}e12)"
      f" -> non-attn {hrs(t_non):.2f} h + attn {hrs(t_att):.2f} h = "
      f"{hrs(c):.2f} h; cross-attn share {t_att / c:.0%}")
print(f"  5% pair sample (confirming cell): {c * 0.05 / 60:.0f} min")

# The flip - terms anchored - to show the anchor rule's stakes.
f2, s2 = p + TR, TL + q
k2 = (S - f2) // s2
m2 = math.ceil(NL / k2)
fresh_f = P * s2 + NR * m2 * f2
pairs_f = (P * (s2 * f2 + s2 * (s2 + 1) // 2)
           + NR * m2 * f2 * (f2 + 1) // 2)
flip = fresh_f * (A_PKD - SQ_CAL * A2) + 2 * pairs_f * A2
print(f"\nflip (terms anchored, packed): fresh {fresh_f / 1e9:.1f}B -> "
      f"{hrs(flip):.0f} h")

w_sat = math.ceil(2 * S / s)
w_mem = int(0.8 * POOL // (f + s + 1))
print(f"\nengine fallback sizing: W_sat = {w_sat}, W_mem = {w_mem} "
      f"(memory-bound below saturation -> drop step budget toward 16k)")
