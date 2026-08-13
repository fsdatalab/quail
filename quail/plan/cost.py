"""The calibrated cost estimator for filter queries.

COST(p) predicts the makespan of one candidate plan as

    T_in + T_quest + T_reread + T_mem + c0

where T_in is the input pass over the heaviest shard, T_quest the
per-stage question work thinned by survival, T_reread the document
tokens recomputed because retention dropped their KV (zero under
chain mode, where admission keeps every live document resident),
T_mem the overflow charge (zero by construction: the admission budget
holds the resident footprint under the pool), and c0 a fixed
per-query software residue.

Every calibrated constant of the estimator lives in one table here.
Update them here and nowhere else.
"""

# Anchors are from the prefill speed control on the CUDA 13 image
# (results/engine/speed_limit.json: best cell 97,889 tok/s), and the
# batch-size sweep reproduced 97,005 tok/s at B = 25,305 through the
# synchronous API. Constraint the code cannot show: the achieved rate
# is a property of the image, not the engine - the old slim image
# sustained 80,556 - and accuracy moves with the same substrate, so a
# stack change requires re-anchoring both together.
PHI = 97_000 / 275_000        # serving rate over spec ceiling, 4B anchor
ENGINE_OVERHEAD_S = 3.2       # c0: per-query software residue at 10k docs.
                              # NOT re-anchored: the flight measured up to
                              # 45 percent wall spread across containers
                              # (host variance), so c0 waits for a
                              # host-controlled protocol.
BOOT_POOL_FRACTION = 0.92     # gpu_memory_utilization we ship
POOL_HEADROOM = 0.80          # admission budget stays under the pool by this
ENGINE_SEQS_MAX = 4096        # hard bound on max_num_seqs at boot:
#                               per-sequence engine overheads (FlashInfer
#                               workspace, sampler buffers) live outside
#                               both the plan's and the engine's pool
#                               accounting, and the unbounded cap OOM'd
#                               a 25k-seq boot with ~6 GB unaccounted
ACT_BYTES_PER_HIDDEN = 32     # peak per-token activation bytes per hidden
#                               dim (the MLP gate and up intermediates
#                               dominate; ~82 KB/token at 4B). An
#                               architecture estimate, not yet measured.
STEP_TOKENS_MIN = 2048        # smallest step budget worth booting: below
#                               this the weight pass stops amortizing
STEP_TOKENS_MAX = 32768       # the sweep's largest tested budget; past
#                               it nothing is measured
STEP_POOL_FRACTION = 0.03     # the activation reservation a step budget
#                               may take from the KV pool: steps are
#                               compute-bound past ~400 tokens, so a
#                               bigger budget buys only amortization of
#                               the per-step host cost and must not
#                               charge a thin pool for it

# The measured end-to-end step model from the batch-size sweep
# (experiments/modal_profiling.py::batchsweep, 10k documents, one
# filter, synchronous API, B in 512..25,305):
#
#     T_step(B) = STEP_TOKEN_S * B + STEP_FIXED_S
#
# The per-token term is what a token costs once the dense projections
# are compute-bound; the fixed term is kernel launches, the sampler,
# and Python glue, paid once per step whatever the step holds. The
# knee (STEP_FIXED_S / STEP_TOKEN_S, about 290 tokens) is where the
# fixed cost stops dominating; the analytical dense-projection ridge
# sits near 400 tokens, which is the same story from the other side.
STEP_TOKEN_S = 10.7e-6        # a: seconds per token
STEP_FIXED_S = 3.1e-3         # b: seconds per step


# ------------------------------------------------------- rate primitives

def dense_seconds(model, tokens, compute_rate):
    """Seconds of dense forward-pass compute over `tokens`: 2P FLOPs
    per token at `compute_rate` FLOP/s (the spec ceiling R_D, or
    PHI * R_D when priced at the calibrated serving rate)."""
    return 2.0 * model.P * tokens / compute_rate


def step_seconds(tokens):
    """The measured step model: per-token cost plus the fixed per-step
    host cost. Predicts the sweep's walls; use it to choose a step
    budget, not to price a whole query (that is predict_makespan)."""
    return STEP_TOKEN_S * tokens + STEP_FIXED_S


def read_rate(model, device):
    """R, tokens per second: PHI applied to the device's spec ceiling
    at the dense cost of 2P FLOPs per token, so reading T tokens costs
    dense_seconds(model, T, PHI * device.R_D)."""
    return PHI * device.R_D / (2 * model.P)


# ------------------------------------------------------- the estimator

def stage_survivals(n_filters, selectivity):
    """A_j = surv(j-1) for j = 1..n, the fraction of documents that
    issue stage j when every stage is gated. A scalar selectivity is
    one pass rate for every stage; a sequence gives per-stage rates."""
    if hasattr(selectivity, "__len__"):
        out, s = [1.0], 1.0
        for x in selectivity[:n_filters - 1]:
            s *= x
            out.append(s)
        return tuple(out)
    return tuple(selectivity ** j for j in range(n_filters))


def t_in(model, device, shard_tokens, access="read", store_read_bw=None):
    """The input pass over the heaviest shard: read every token once
    at R, or load the persisted KV bytes from a warm store at the
    store's bandwidth - whichever access the plan selected (the
    restore side pays only the read, because the write was paid at
    ingest)."""
    if access == "restore":
        return shard_tokens * model.kappa / store_read_bw
    return shard_tokens / read_rate(model, device)


def t_quest(model, device, n_docs, workers, n_filters,
            question_tokens=46, preamble_tokens=33, selectivity=1.0):
    """Question work per worker: stage 1 pays its full question; each
    later stage pays only the tail past the shared preamble, thinned
    by survival. question_tokens may be one shared length or a
    per-stage sequence. The tail is clamped at one token: a reached
    stage always appends at least its answer cue."""
    A = stage_survivals(n_filters, selectivity)
    lens = question_tokens if hasattr(question_tokens, "__len__") \
        else (question_tokens,) * n_filters
    q = A[0] * lens[0]
    for j in range(1, n_filters):
        q += A[j] * max(1, lens[j] - preamble_tokens)
    return (n_docs / workers) * q / read_rate(model, device)


def t_reread(model, device, reread_tokens):
    """Document tokens recomputed because retention dropped their KV,
    at R. Zero in chain mode, where the document's KV belongs to a
    living request and admission keeps it resident."""
    return reread_tokens / read_rate(model, device)


def t_mem(model, device, width_docs, doc_tokens, pool_bytes,
          store_bw=None):
    """Zero while the resident footprint w * f fits the pool K;
    otherwise the overflow bytes at the cheaper of recompute and a
    spill round trip, charged once. Emitted plans never carry this
    term - the admission budget holds w * f <= K by construction."""
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
                     reread_tokens=0, width_docs=None, doc_tokens=None,
                     pool_bytes=None, store_bw=None,
                     c0=ENGINE_OVERHEAD_S):
    """T_in + T_quest + T_reread + T_mem + c0.

    T_mem is optional because the shipped planner zeroes it:
    width_docs=None prices it at zero, since the admission budget
    keeps the footprint under the pool by construction. Pass
    width_docs (with doc_tokens and pool_bytes) to price the full
    form."""
    total = t_in(model, device, shard_tokens, access, store_read_bw)
    total += t_quest(model, device, n_docs, workers, n_filters,
                     question_tokens, preamble_tokens, selectivity)
    total += t_reread(model, device, reread_tokens)
    if width_docs is not None:
        total += t_mem(model, device, width_docs, doc_tokens, pool_bytes,
                       store_bw)
    return total + c0
