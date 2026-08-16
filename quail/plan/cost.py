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

# The serving rate is derived from the measured step model below
# (STEP_TOKEN_S, the batch-size sweep on the production boot), not
# set by hand: 1/STEP_TOKEN_S = 96,180 tokens/s at 4B on the H100,
# 35.0 percent of the spec ceiling - matching the retired hand
# anchor (97,000/275,000) within one percent. Cross-checks that the
# rate is a property of the image, not the engine: the prefill speed
# control (results/engine/speed_limit.json, best cell 97,889 tok/s);
# the old slim image sustained 80,556, so a stack change requires
# re-anchoring rate and accuracy together.
#
# Constants measured on this configuration; other models and devices
# are priced by spec-ratio scaling from it (see _scale).
CAL_MODEL_P = 3.6e9           # Qwen3 4B dense params
CAL_DEVICE_RD = 1.979e15      # H100 SXM fp8 dense FLOP/s
CAL_SQ_PER_TOKEN = 472.44     # the calibration corpus's squared-length
#                               sum over its token sum
#                               (modal_filters.py::stats, 10k docs).
#                               STEP_TOKEN_S was measured on that
#                               corpus, so its rate already embeds
#                               a2 * this much attention per token;
#                               the quadratic surcharge must charge
#                               only the excess or it double-counts.
ENGINE_OVERHEAD_S = 0.026     # c0: per-query software residue at 10k docs,
                              # planned arm. From the c0 anchor protocol
                              # (results/engine/c0_anchor.json): wall minus
                              # reads x corpus at the rate the same
                              # container's probe served, so host speed
                              # cancels out of the subtraction. Rewind
                              # median 0.026 s (reps 0.013/0.026/0.244);
                              # stock median 0.659. The old 3.2 was
                              # fleet-anchored and booked host variance
                              # as overhead.
BOOT_POOL_FRACTION = 0.92     # gpu_memory_utilization we ship
SATURATION_SLACK = 2          # admission budget = this x n_filters x step
#                               budget. sigma*B tokens are live at once
#                               (sigma = filter count, the survival
#                               ceiling; the step trace measured 4.1-5.1
#                               live cohorts at five filters) and one
#                               more sigma*B sits queued so a client
#                               top-up stall of up to sigma steps never
#                               starves a step. Waiting documents hold
#                               no KV, so the queue costs no pool.
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
# knee (STEP_FIXED_S / STEP_TOKEN_S, about 283 tokens) is where the
# fixed cost stops dominating; the analytical dense-projection ridge
# sits near 400 tokens, which is the same story from the other side.
STEP_TOKEN_S = 10.39734565764918e-6  # a: seconds per token
STEP_FIXED_S = 2.939878716623444e-3  # b: seconds per step

# The calibration-sweep fits, one container
# (experiments/modal_calibrate.py --families all ->
# results/engine/calibrate_all.json, fitted by quail/plan/fit.py).
# Constraints the numbers cannot show: the sweep boots the engine
# synchronous and eager (async_scheduling off, enforce_eager on - the
# step trace and the single-graph padding required it), so STEP_B0_S
# is the eager engine's host launch floor, not the ~3 ms graph-mode
# step cost above. T_READ_S_PER_TOKEN is the f_P slope at the c=32
# reference; the read price grows with suffix length (raw two-cell
# slope 101 ns/token, fitted-over-raw 0.56), so it is a reference
# point, not a universal per-byte rate. HOST_HR_S is the block-table
# cost per resident KV block, paid every step a request stays
# scheduled; the host model fits with MAPE 0.15.
# --- calibration fits (results/engine/cost_model_fit.json) ---
ALPHA1_S_PER_TOKEN = 9.196521782489011e-06
ALPHA2_S_PER_TOKEN2 = 4.933635085554017e-10
T_READ_S_PER_TOKEN = 5.675842254567658e-08
EPSILON_READ_OVER_PRE = 0.0061717270820528716
STEP_B0_S = 0.016790614841073047
STEP_BETA_N_S = 5.027248077955851e-05
HOST_H0_S = 0.0
HOST_HN_S = 2.4330984797175877e-05
HOST_HA_S = 4.886350071536961e-07
HOST_HR_S = 1.1169313206227985e-06
TRANSPORT_BW_BPS = {'c6_d2h_unpinned': 11856448657.0, 'c6_h2d_unpinned': 10916219696.0, 'c6_d2h_pinned': 55339616543.0, 'c6_h2d_pinned': 55472110922.0, 'c6_disk_write': 2605516808.0, 'c6_disk_read': 3877713374.0, 'c6_volume_write': 856544232.0, 'c6_volume_read': 3242050855.0}
OFFLOAD_CROSSOVER_TOKENS = {'c6_d2h_unpinned': 0.0, 'c6_h2d_unpinned': 0.0, 'c6_d2h_pinned': 0.0, 'c6_h2d_pinned': 0.0, 'c6_disk_write': 38714.5770760633, 'c6_disk_read': 19897.590958957873, 'c6_volume_write': 155827.48237132592, 'c6_volume_read': 27453.67024434094}


# ------------------------------------------------------- rate primitives

def _scale(model, device):
    """Spec-ratio scaling from the calibrated configuration: a model
    with more params costs proportionally more per token, a device
    with a higher ceiling proportionally less. The measured efficiency
    is assumed to travel; the absolute rates do not."""
    return (model.P / CAL_MODEL_P) * (CAL_DEVICE_RD / device.R_D)


def dense_seconds(model, tokens, compute_rate):
    """Seconds of dense forward-pass compute over `tokens`: 2P FLOPs
    per token at `compute_rate` FLOP/s (the spec ceiling R_D)."""
    return 2.0 * model.P * tokens / compute_rate


def step_seconds(tokens):
    """The measured step model: per-token cost plus the fixed per-step
    host cost. Predicts the sweep's walls; use it to choose a step
    budget, not to price a whole query (that is predict_makespan)."""
    return STEP_TOKEN_S * tokens + STEP_FIXED_S


def token_seconds(model, device):
    """Seconds to compute one fresh token, sustained on the production
    boot: the batch sweep's measured per-token cost, spec-scaled."""
    return STEP_TOKEN_S * _scale(model, device)


def read_rate(model, device):
    """R, tokens per second: the sustained fresh-token rate, the
    reciprocal of token_seconds."""
    return 1.0 / token_seconds(model, device)


def read_seconds_per_token(model, device):
    """Seconds to re-read one resident token under a filter-width
    suffix (the calibration's c=32 reference), spec-scaled. The
    measured price is pair work, not bytes, so it scales with the
    compute ceiling like the fresh rate."""
    return T_READ_S_PER_TOKEN * _scale(model, device)


def attn_seconds_per_token2(model, device):
    """The quadratic prefill surcharge, spec-scaled: a document of h
    tokens costs this times h^2 on top of its linear token work."""
    return ALPHA2_S_PER_TOKEN2 * _scale(model, device)


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


def t_in(model, device, shard_tokens, access="read", store_read_bw=None,
         doc_sq_tokens=0.0):
    """The input pass over the heaviest shard: compute every token
    once at the sustained rate plus the quadratic attention
    surcharge, or load the persisted KV bytes from a warm store at
    the store's bandwidth - whichever access the plan selected (the
    restore side pays only the read, because the write was paid at
    ingest).

    The surcharge is centered: the sustained rate was measured on the
    calibration corpus and already embeds a2 * CAL_SQ_PER_TOKEN of
    attention per token, so only the shard's excess squared length
    over that profile is charged (a shorter-profile corpus gets the
    matching discount). doc_sq_tokens = sum of squared document
    lengths; 0 drops the surcharge entirely for callers that cannot
    supply corpus shape."""
    if access == "restore":
        return shard_tokens * model.kappa / store_read_bw
    linear = shard_tokens * token_seconds(model, device)
    if not doc_sq_tokens:
        return linear
    excess = doc_sq_tokens - shard_tokens * CAL_SQ_PER_TOKEN
    return linear + excess * attn_seconds_per_token2(model, device)


def quest_token_count(n_docs, workers, n_filters, question_tokens=46,
                      preamble_tokens=33, selectivity=1.0):
    """Question tokens computed per worker: stage 1 pays its full
    question; each later stage pays only the tail past the shared
    preamble, thinned by survival. question_tokens may be one shared
    length or a per-stage sequence. The tail is clamped at one token:
    a reached stage always appends at least its answer cue. Pure
    token arithmetic, so callers can count as well as price."""
    A = stage_survivals(n_filters, selectivity)
    lens = question_tokens if hasattr(question_tokens, "__len__") \
        else (question_tokens,) * n_filters
    q = A[0] * lens[0]
    for j in range(1, n_filters):
        q += A[j] * max(1, lens[j] - preamble_tokens)
    return (n_docs / workers) * q


def quest_read_tokens(n_docs, workers, n_filters, mean_doc_tokens,
                      preamble_tokens=33, selectivity=1.0,
                      kept_preamble=True):
    """Resident tokens the later stages re-read at the cached rate:
    each stage past the first evaluates its suffix against the
    document (plus the kept preamble in chain mode), thinned by
    survival. Stage 1 reads nothing extra - its document is fresh in
    the same pass and t_in already priced it."""
    A = stage_survivals(n_filters, selectivity)
    ctx = mean_doc_tokens + (preamble_tokens if kept_preamble else 0)
    return (n_docs / workers) * sum(A[1:]) * ctx


def t_quest(model, device, n_docs, workers, n_filters,
            question_tokens=46, preamble_tokens=33, selectivity=1.0,
            mean_doc_tokens=0.0):
    """Question work per worker: computed question tokens at the
    sustained fresh rate, plus the later stages' re-read of resident
    context at the cached-read rate. mean_doc_tokens=0 drops the read
    term for callers that cannot supply it."""
    compute = quest_token_count(
        n_docs, workers, n_filters, question_tokens, preamble_tokens,
        selectivity) * token_seconds(model, device)
    reads = quest_read_tokens(
        n_docs, workers, n_filters, mean_doc_tokens, preamble_tokens,
        selectivity) * read_seconds_per_token(model, device)
    return compute + reads


def t_reread(model, device, reread_tokens):
    """Document tokens recomputed because retention dropped their KV,
    at the sustained rate. Zero in chain mode, where the document's
    KV belongs to a living request and admission keeps it resident."""
    return reread_tokens * token_seconds(model, device)


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
                     mean_doc_tokens=0.0, doc_sq_tokens=0.0,
                     step_tokens=STEP_TOKENS_MAX,
                     c0=ENGINE_OVERHEAD_S):
    """T_in + T_quest + T_reread + T_step_fixed + T_mem + c0.

    mean_doc_tokens and doc_sq_tokens carry the composition terms
    (later stages' cached re-reads; the quadratic prefill surcharge);
    zero drops each for callers that cannot supply corpus shape.
    T_step_fixed amortizes the measured per-step fixed cost over the
    fresh tokens the query computes, at the boot's step budget.
    T_mem is optional because the shipped planner zeroes it:
    width_docs=None prices it at zero, since the admission budget
    keeps the footprint under the pool by construction. Pass
    width_docs (with doc_tokens and pool_bytes) to price the full
    form."""
    total = t_in(model, device, shard_tokens, access, store_read_bw,
                 doc_sq_tokens=doc_sq_tokens)
    total += t_quest(model, device, n_docs, workers, n_filters,
                     question_tokens, preamble_tokens, selectivity,
                     mean_doc_tokens=mean_doc_tokens)
    total += t_reread(model, device, reread_tokens)
    fresh = (shard_tokens if access == "read" else 0) + reread_tokens \
        + quest_token_count(n_docs, workers, n_filters, question_tokens,
                            preamble_tokens, selectivity)
    total += STEP_FIXED_S * fresh / max(1, step_tokens)
    if width_docs is not None:
        total += t_mem(model, device, width_docs, doc_tokens, pool_bytes,
                       store_bw)
    return total + c0
