"""Every number in plans/join_plan.md, derived. Run:

    python plans/join_estimates.py

Inputs, in provenance order:

  quail/plan/cost.py
      The fitted constants (fit.py over calibrate_all.json). Imported,
      not re-derived - that file is the canonical table.
  results/engine/calibrate_all.json
      Raw c2 cells -> the cached-read price per suffix width, by the
      protocol cost_model.md Section 4 states: at width c and N = 32,
      subtract the h=8,192 cell's engine-timer median from the
      h=16,384 cell's and divide by the 262,144 additional cached
      tokens. (Reproduces the documented 101 ns at c=32.) Also the
      boot row's pool size.
  results/engine/single_filter_forward_vllm_kernels.json
      The packed pipeline rate (the custom-kernel ladder rung) and
      the chunk token budget the ladder ran at.
  results/engine/filter_cells.json
      The shipped admission budget in tokens.

One number has NO committed artifact: the default-admission thrash
multiplier (README records 80.1 s against the fair 42.9 s, x1.87,
but those cells were never landed in results/). It is carried here
as prose-sourced and the plan's 5% sample run measures it fresh.

The block marked ASSUMPTIONS is the only hand-typed input: dataset
sizes from the FDJ paper's Table 1, and document/prompt lengths that
step 1 of the plan replaces with tokenized values.

Attention accounting: a prefill of h tokens costs a2*h^2 on top of
linear work, and causal pairs(h) ~ h^2/2, so the price per attention
pair is 2*a2. Both sustained rates were measured on the calibration
corpus (profile CAL_SQ_PER_TOKEN squared tokens per token), so each
estimate charges only its excess over that profile - cost.py's
centering rule.
"""

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quail.configs import H100_SXM, QWEN3_4B_FP8
from quail.plan.cost import (
    ACT_BYTES_PER_HIDDEN,
    ALPHA2_S_PER_TOKEN2 as A2,
    BOOT_POOL_FRACTION,
    CAL_SQ_PER_TOKEN as SQ_CAL,
    STEP_BETA_N_S as BETA_N,
    STEP_FIXED_S,
    STEP_TOKEN_S as A_ENG,
)

# ---------------------------------------------------------- artifacts

cal = json.load(open(ROOT / "results/engine/calibrate_all.json"))
ladder = json.load(
    open(ROOT / "results/engine/single_filter_forward_vllm_kernels.json"))
filters = json.load(open(ROOT / "results/engine/filter_cells.json"))

POOL = next(r["pool_tokens"] for r in cal if r.get("meta") == "boot")
BUDGET = filters["budget_tokens"]
A_PKD = 1.0 / ladder["mean_tokens_per_second"]["packed_custom_qk"]

# The chunk budget B* is a parameter, not a constant. Memory caps it
# (the packed pass holds only activations plus any kept prefixes -
# no KV cache), the measured flat region floors it at ~4,096, and
# the default is the sweep's largest measured point. Bigger B cuts
# prefix recomputes, but the sweep recorded per-token GEMM cost
# rising with B (L2 pressure, the open problem in the README), so
# values past the measured range are probe cells, not assumptions.
MODEL, DEVICE = QWEN3_4B_FP8, H100_SXM
ACT = ACT_BYTES_PER_HIDDEN * MODEL.h        # peak activation bytes/token


def chunk_cap(kept_prefix_tokens=0):
    free = (DEVICE.M * BOOT_POOL_FRACTION - MODEL.W_mem
            - kept_prefix_tokens * MODEL.kappa)
    return int(free // ACT)


B_MEAS = ladder["batch_tokens"]             # largest measured sweep point

# B* is derived, not chosen: the memory cap over a declared slack.
# The slack covers the two soft spots in the cap - the activation
# estimate is architectural, not measured (cost.py's own caveat),
# and the FlashInfer path costs 147 KB/token instead of 83. All
# arms run at B*: packed chunks and the stock boot's
# max_num_batched_tokens alike. B* sits far past the measured
# sweep, so every wall below is conditional on the rates holding
# there; one reference cell at B_MEAS rides along to tell a rate
# change at large B apart from a slow kernel.
SLACK = 2
S = chunk_cap(0) // SLACK

c2 = {r["cell"]: r for r in cal if r.get("family") == "c2"}
READ_NS = {}
for c in (16, 32, 64):
    lo = c2[f"c2_c{c}_h8192_n32"]["exec_ms_median"]
    hi = c2[f"c2_c{c}_h16384_n32"]["exec_ms_median"]
    READ_NS[c] = (hi - lo) * 1e6 / (32 * 8192)

STOCK_FAIR_WALL = statistics.median(
    c["wall"] for c in filters["cells"] if c["arm"] == "stock")
THRASH = 80.1 / 42.9   # README prose only - no committed cells; see header

# -------------------------------------------------------- ASSUMPTIONS

NL, NR = 8103, 3718    # reports x terms (FDJ paper, Table 1)
TL, TR = 1000, 8       # document tokens - step 1 tokenizes the real data
p, q = 40, 50          # preamble / question tail - step 1 measures the prompt

# --------------------------------------------------------------------

P = NL * NR
f, s = p + TL, TR + q


def read_price(c):
    """Per-cached-token price at suffix width c, interpolated between
    the measured widths. c or h outside the calibrated grid is flagged
    where used."""
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


print("derived from artifacts: pool "
      f"{POOL:,} tok (calibrate boot); admission budget {BUDGET:,} tok "
      f"(filter run); packed rate {1 / A_PKD:,.0f} tok/s (ladder)")
print(f"chunk budget B* = cap/{SLACK} = {S:,} tok, all arms (cap "
      f"{chunk_cap():,} at {ACT / 1e3:.0f} KB/token activations; largest "
      f"measured point {B_MEAS:,}, kept as the reference cell)")
print("cached-read prices from c2 cells (ns/cached token): "
      + ", ".join(f"c={c}: {v:.0f}" for c, v in READ_NS.items())
      + f"; fair stock wall {STOCK_FAIR_WALL:.1f} s")

print(f"\npairs P = {NL} x {NR} = {P:,}")
print(f"f = {p}+{TL} = {f}; s = {TR}+{q} = {s}")
print(f"terms side, fully resident = NR*(p+TR) = {NR * (p + TR):,} tokens "
      f"= {NR * (p + TR) / POOL:.0%} of the pool")

k = (S - f) // s
m = math.ceil(NR / k)
print(f"chunk geometry: k = ({S}-{f})//{s} = {k} suffixes/chunk; "
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
      f"{hrs(a1):.1f} h; x{THRASH:.2f} thrash (README-sourced, not "
      f"remeasured here) = {hrs(a1 * THRASH):.0f} h")
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
print(f"  reads P*f*{rd * 1e9:.0f}ns (c2 cells interpolated to c={s}) = "
      f"{hrs(t_read):.2f} h  [c=32/c=64 cell prices: "
      f"{hrs(P * f * READ_NS[32] * 1e-9):.2f}/"
      f"{hrs(P * f * READ_NS[64] * 1e-9):.2f} h; h={f} is below the "
      f"2,048-minimum calibrated grid]")
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
c_wall = t_non + t_att
print(f"\nC packed, recompute: fresh = P*s + NL*m*f = {P * s / 1e9:.3f}B + "
      f"{NL * m * f / 1e6:.1f}M = {fresh_c / 1e9:.3f}B")
print(f"  attention pairs {pairs / 1e12:.2f}e12 (cross {P * s * f / 1e12:.2f}e12)"
      f" -> non-attn {hrs(t_non):.2f} h + attn {hrs(t_att):.2f} h = "
      f"{hrs(c_wall):.2f} h; cross-attn share {t_att / c_wall:.0%}; "
      f"effective rate {fresh_c / c_wall:,.0f} tok/s")
print(f"  5% pair sample (confirming cell): {c_wall * 0.05 / 60:.0f} min")

# Reference cell at the largest measured point: same math at B_MEAS,
# to separate "rate fell at 85k" from "our kernel is slow" if the
# main arm misses its prediction.
for B in (B_MEAS,):
    kB = (B - f) // s
    mB = math.ceil(NR / kB)
    fresh_v = P * s + NL * mB * f
    pairs_v = (P * (s * f + s * (s + 1) // 2)
               + NL * mB * (f * (f + 1) // 2))
    w = fresh_v * (A_PKD - SQ_CAL * A2) + 2 * pairs_v * A2
    print(f"  reference cell at B*={B:,} (largest measured): k={kB:,}, "
          f"m={mB}, recompute {NL * mB * f / fresh_v:.1%} -> {hrs(w):.2f} h")

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

ENGINE_STEP = 16_384   # the engine fallback's own step budget (flat
#                        region; the 85k chunk budget is a packed setting)
w_sat = math.ceil(2 * ENGINE_STEP / s)
w_mem = int(BUDGET // (f + s + 1))
print(f"\nengine fallback sizing at a {ENGINE_STEP:,} step budget: "
      f"W_sat = {w_sat}, W_mem = {w_mem}")

# The prototype sample: 100 reports, ALL terms - sampling only the
# anchor side keeps the per-report chunk geometry (m stays 9), so the
# full-scale walls are the sample walls times NL/100.
NLS = 100
PS = NLS * NR
fresh_cs = PS * s + NLS * m * f
pairs_cs = (PS * (s * f + s * (s + 1) // 2)
            + NLS * m * f * (f + 1) // 2)
c_s = fresh_cs * (A_PKD - SQ_CAL * A2) + 2 * pairs_cs * A2
fresh_as = NLS * f + PS * s
a2_s = (fresh_as * A_ENG
        + attn_excess_s(NLS * f, f) + attn_excess_s(PS * s, s)
        + PS * f * rd + PS * BETA_N
        + PS * 8 * (A_ENG + STEP_FIXED_S / S)
        + fresh_as / S * STEP_FIXED_S)
print(f"\nprototype sample ({NLS} reports x all {NR} terms = {PS:,} pairs; "
      f"extrapolation x{NL / NLS:.0f}):")
print(f"  packed: {fresh_cs / 1e6:.1f}M fresh -> {c_s / 60:.1f} min")
print(f"  grouped stock, client-side admission: {fresh_as / 1e6:.1f}M fresh "
      f"+ reads/requests/boundary -> {a2_s / 60:.1f} min")
