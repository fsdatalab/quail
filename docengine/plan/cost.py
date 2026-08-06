"""Algorithm 1 of paper/PAPER.md section 5.1: the calibrated cost estimator.

COST(p) predicts the makespan of one candidate plan as a sum of five
terms plus a fixed overhead:

    T_in + T_quest + T_reread + T_dec + T_mem + c0

predict_makespan is the paper's line 9; each term is a small helper
here (t_in .. t_mem) so a caller can price one term alone. The planner
(plan_query in docengine/plan/planner.py) consumes this module for its
prediction. The reasoning layer (docengine/reasoning/model.py) prices
policies stepwise from the same primitives - dense_seconds,
decode_step_seconds, fluid_block_seconds - so no rate law in the
package is derived twice.
"""

# Every calibrated constant of the estimator, in one table. Update
# them here and nowhere else. Anchors are from the re-baseline flight
# on the CUDA 13 image (results/engine/speed_limit.json: best cell
# 97,889 tok/s, cross-checked by xengine.json's 97,220 vLLM and
# 97,393 SGLang at matching config). Constraint the code cannot show:
# the achieved rate is a property of the image, not the engine - the
# old slim image sustained 80,556 - and accuracy moves with the same
# substrate (see the attribution table in notes/RESULTS.md), so a
# stack change requires re-anchoring BOTH numbers together.
PHI = 97_000 / 275_000        # serving rate over spec ceiling, 4B anchor
ENGINE_OVERHEAD_S = 3.2       # c0: per-query software residue at 10k docs.
                              # NOT re-anchored: the flight measured up to
                              # 45 percent wall spread across containers
                              # (host variance), so c0 waits for a
                              # host-controlled protocol.
BOOT_POOL_FRACTION = 0.92     # gpu_memory_utilization we ship
POOL_HEADROOM = 0.80          # admission budget stays under the pool by this
ROUND_TOKENS = 2048           # question work that keeps one round busy:
#                               the observed practical round size at
#                               corpus scale (step recorder, spec_smoke).
#                               The hybrid switch errs conservative if
#                               real rounds run bigger. Measured under
#                               the old fixed 2,048 step budget; the
#                               budget is now plan-derived (usually
#                               larger), so this constant awaits
#                               re-measurement from the step recorder.
FORK_SEQ_S = 70e-6            # scheduler CPU per forked sibling, measured
#                               (the ledger's fork section, spec_smoke
#                               2026-08-05: the wall gap that survived
#                               removing all recompute)
ACT_BYTES_PER_HIDDEN = 32     # peak per-token activation bytes per hidden
#                               dim (the MLP gate and up intermediates
#                               dominate; ~82 KB/token at 4B, ~164 KB at
#                               32B). An architecture estimate, not yet
#                               measured - like the C5 weight sizes it
#                               awaits a profiled number.
STEP_TOKENS_MIN = 2048        # smallest step budget worth booting: below
#                               this the weight pass stops amortizing
STEP_TOKENS_MAX = 32768       # the sweep's largest tested budget; past
#                               it nothing is measured
STEP_POOL_FRACTION = 0.03     # the activation reservation a step budget
#                               may take from the KV pool: steps are
#                               compute-bound past ~105 tokens, so a
#                               bigger budget buys only amortization
#                               (measured 0.8 percent, defaults against
#                               the 32k best cell) and must not charge
#                               a thin pool for it


# ------------------------------------------------------- rate primitives

def dense_seconds(model, tokens, compute_rate):
    """Seconds of dense forward-pass compute over `tokens`: 2P FLOPs
    per token at `compute_rate` FLOP/s (the spec ceiling R_D, or
    PHI * R_D when priced at the calibrated serving rate)."""
    return 2.0 * model.P * tokens / compute_rate


def decode_step_seconds(model, device, concurrent, context_tokens,
                        compute_rate):
    """Seconds for one decode step advancing `concurrent` calls at mean
    context `context_tokens`: the larger of the step's dense compute
    and its memory traffic (one weight pass plus one KV read per
    call). This is how the reasoning layer prices decode stepwise."""
    comp = 2.0 * model.P * concurrent / compute_rate
    bw = (model.W_run + model.kappa * context_tokens * concurrent) \
        / device.BW
    return max(comp, bw)


def fluid_block_seconds(model, device, tokens, compute_rate):
    """Fluid price of pushing `tokens` through one ungated block: one
    forward pass plus one KV write per token. Backs BLOCKCOST of
    Algorithm 3 (group_stages in docengine/reasoning/model.py)."""
    return dense_seconds(model, tokens, compute_rate) \
        + model.kappa * tokens / device.BW


def read_rate(model, device):
    """R, tokens per second: PHI applied to the device's spec ceiling
    at the dense cost of 2P FLOPs per token, so reading T tokens costs
    dense_seconds(model, T, PHI * device.R_D). Anchors: 80,000
    measured at 4B on the H100 (the ratio PHI); 10,800 measured at
    32B against 9,300 predicted, fifteen percent under."""
    return PHI * device.R_D / (2 * model.P)


def decode_rate(model, device, concurrent, context_tokens):
    """R_dec, generated tokens per second across `concurrent` calls:
    `concurrent` tokens per step at the step price of
    decode_step_seconds, at the calibrated compute rate PHI * R_D."""
    step = decode_step_seconds(model, device, concurrent, context_tokens,
                               PHI * device.R_D)
    return concurrent / step


# ------------------------------------------------- Algorithm 1, by line

def stage_survivals(n_filters, selectivity):
    """Lines 1-2: A_j = surv(j-1) for j = 1..n, the fraction of
    documents that issue stage j when every stage is gated. A scalar
    selectivity is one pass rate for every stage; a sequence gives
    per-stage rates. Assumption 3: the fraction applies uniformly to
    the corpus token mix."""
    if hasattr(selectivity, "__len__"):
        out, s = [1.0], 1.0
        for x in selectivity[:n_filters - 1]:
            s *= x
            out.append(s)
        return tuple(out)
    return tuple(selectivity ** j for j in range(n_filters))


def t_in(model, device, shard_tokens, access="read", store_read_bw=None):
    """Line 4, the input pass over the heaviest shard H: read every
    token once at R, or load the persisted KV bytes from a warm store
    at the store's bandwidth - whichever access the plan selected
    (Algorithm 2 line 13; the restore side pays only the read because
    the write was paid at ingest)."""
    if access == "restore":
        return shard_tokens * model.kappa / store_read_bw
    return shard_tokens / read_rate(model, device)


def t_quest(model, device, n_docs, workers, n_filters,
            question_tokens=46, preamble_tokens=33, selectivity=1.0):
    """Line 5, question work per worker: stage 1 pays its full
    question; each later stage pays only the tail past the shared
    preamble, thinned by survival. question_tokens may be one shared
    length or a per-stage sequence. The tail is clamped at one token:
    a reached stage always appends at least its answer cue."""
    A = stage_survivals(n_filters, selectivity)
    lens = question_tokens if hasattr(question_tokens, "__len__") \
        else (question_tokens,) * n_filters
    q = A[0] * lens[0]
    for j in range(1, n_filters):
        q += A[j] * max(1, lens[j] - preamble_tokens)
    return (n_docs / workers) * q / read_rate(model, device)


def t_reread(model, device, reread_tokens):
    """Line 6: X document tokens recomputed because retention dropped
    their KV, at R. X = 0 in rewind execution, where admission keeps
    every live document resident; what a plan drops and re-reads is
    priced by PAPER.md section 6."""
    return reread_tokens / read_rate(model, device)


def t_dec(n_docs, workers, n_filters, selectivity, thinking_tokens,
          r_dec, answer_tokens=1):
    """Line 7, decode: G thinking tokens plus the answer tokens of
    every reached stage, at R_dec. answer_tokens is 1 under the
    one-token answer contract and up to 6 under the decisive-token
    window (Algorithm 2 line 14)."""
    A = stage_survivals(n_filters, selectivity)
    return (n_docs / workers) * sum(A) \
        * (thinking_tokens + answer_tokens) / r_dec


def t_mem(model, device, width_docs, doc_tokens, pool_bytes,
          store_bw=None):
    """Line 8: zero while the resident footprint w * f fits the pool
    K; otherwise the overflow bytes priced at the per-gap rate of
    PAPER.md Lemma 2, min(recompute, spill round trip), charged once.
    That single charge is a lower envelope of the section 6 schedule;
    it is kept simple because emitted plans never carry this term -
    the admission budget holds w * f <= K by construction."""
    footprint = width_docs * model.kappa * doc_tokens
    if footprint <= pool_bytes:
        return 0.0
    excess = footprint - pool_bytes
    recompute = (excess / model.kappa) / read_rate(model, device)
    if store_bw:
        return min(recompute, 2.0 * excess / store_bw)
    return recompute


def predict_makespan(model, device, *, n_docs, workers, shard_tokens,
                     n_filters, question_tokens=46, preamble_tokens=33,
                     selectivity=1.0, access="read", store_read_bw=None,
                     reread_tokens=0, thinking_tokens=None,
                     answer_tokens=1, decode_concurrency=None,
                     decode_context_tokens=None, width_docs=None,
                     doc_tokens=None, pool_bytes=None, store_bw=None,
                     c0=ENGINE_OVERHEAD_S):
    """Line 9: T_in + T_quest + T_reread + T_dec + T_mem + c0.

    Two terms are optional because the shipped planner's stated
    divergences (PAPER.md section 5.1, "what the shipped code
    computes") zero them: thinking_tokens=None prices T_dec at zero -
    at G = 0 with one-token answers the decode term sits inside the
    calibration residue that c0 absorbs - and width_docs=None prices
    T_mem at zero, because the admission budget keeps the footprint
    under the pool by construction. Pass thinking_tokens (with
    decode_concurrency and decode_context_tokens, which fix R_dec)
    and width_docs (with doc_tokens and pool_bytes) to price the full
    paper form.
    """
    total = t_in(model, device, shard_tokens, access, store_read_bw)
    total += t_quest(model, device, n_docs, workers, n_filters,
                     question_tokens, preamble_tokens, selectivity)
    total += t_reread(model, device, reread_tokens)
    if thinking_tokens is not None:
        if decode_concurrency is None or decode_context_tokens is None:
            raise ValueError("pricing T_dec needs decode_concurrency and "
                             "decode_context_tokens to fix R_dec")
        r_dec = decode_rate(model, device, decode_concurrency,
                            decode_context_tokens)
        total += t_dec(n_docs, workers, n_filters, selectivity,
                       thinking_tokens, r_dec, answer_tokens)
    if width_docs is not None:
        total += t_mem(model, device, width_docs, doc_tokens, pool_bytes,
                       store_bw)
    return total + c0
