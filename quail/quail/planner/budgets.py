"""Every derived quantity in the design's spec table (engine_design.md
section 8). Pure arithmetic over the two spec structs; the only
measured inputs are the calibration constants, and only the
restore break-even row consumes them.

The two token budgets, and why neither makes the GPU faster:

- The chunk budget is the batch size: tokens per forward pass. It
  keeps the per-chunk fixed cost amortized; the rate is flat past the
  compute knee, so we run at the kernel index cap because bigger is
  free, not because it is needed.
- The admission budget (arena tokens) is KV residency. It guarantees
  no document token is ever computed twice: a preallocated,
  never-oversubscribed arena turns "computed once" from a cache
  policy into a certainty.
"""

from quail.executor.kvstore import PinnedStore
from quail.specs import DeviceSpec, ModelSpec

POOL_FRACTION = 0.92    # the fraction of device memory the executor
#                         may claim (gpu_memory_utilization we ship)
CHUNK_SLACK = 2         # declared slack on the activation bound: it
#                         carries the unmeasured activation estimate
PAGE_TOKENS = 16        # KV arena page size, tokens
INT32_MAX = 2**31 - 1
ACT_BYTES = 2           # bf16 activations, bytes per element
ACT_RESERVE_CHUNKS = 2  # chunks of activation memory reserved outside
#                         the arena: chunk construction overlaps the
#                         previous chunk's forward pass, and the
#                         caching allocator fragments across variable
#                         chunk shapes. Measured, not guessed: the
#                         single-chunk reservation OOMed the milestone
#                         filter run with 4.7 GiB reserved-but-
#                         unallocated on top of the live set.
STORE_KV_BYTES = 2.0    # the pinned store's staging tensors are always
#                         bf16, independent of a model's own kv_bytes
#                         (kept only for the arena's synthetic-fp8
#                         test): PinnedStore._alloc_staging never asks
#                         the model, it hardcodes torch.bfloat16.


def tensor_parallel(model: ModelSpec, device: DeviceSpec) -> int:
    """Smallest power-of-two card count whose pooled memory holds the
    weights. The caller (the planner) refuses when this exceeds the
    configured GPU count."""
    tp = 1
    while model.W_mem > device.mem_bytes * POOL_FRACTION * tp:
        tp *= 2
    return tp


def kernel_index_cap(model: ModelSpec) -> int:
    """The fused kernels compute element offsets in 32-bit ints, so a
    chunk needs rows x widest_row < 2^31. Model-dependent through
    ffn_width; never hardcoded (a 421,750-token chunk died with an
    illegal address in the MLP before this cap existed)."""
    return INT32_MAX // model.ffn_width


def chunk_memory_bound(model: ModelSpec, device: DeviceSpec) -> int:
    """Tokens per chunk the activation memory allows, over the
    declared slack: (M x fraction - weights) / act bytes per token."""
    free = device.mem_bytes * POOL_FRACTION - model.W_mem
    return int(free // model.act_per_token) // CHUNK_SLACK


def chunk_budget(model: ModelSpec, device: DeviceSpec) -> int:
    """min(memory bound, kernel index cap), floored at the compute
    knee. At 4B/H100 the index cap binds: 110,376."""
    b = min(chunk_memory_bound(model, device), kernel_index_cap(model))
    return max(b, int(compute_knee(model, device)))


def store_staging_bytes(model: ModelSpec) -> float:
    """Device memory the pinned KV store's staging ring can hold at
    once: STAGING_BUDGET_TOKENS total token-rows, in bf16. Reserved
    unconditionally (not just when a query's payload asks for a
    store): the arena is built once per warm container and outlives
    any single query, so a later query turning the store on must not
    be able to blow past what the first query's boot already
    committed."""
    return (PinnedStore.STAGING_BUDGET_TOKENS
            * model.kv_elements_per_token * STORE_KV_BYTES)


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None) -> int:
    """The admission budget: tokens of document KV resident at once.
    What is left after weights, the chunk's activation reservation,
    and the store's staging reservation, in KV bytes. Token-based,
    never a document count (a count cannot see length; the 4,096-seq
    default thrashed at 2.40x reads)."""
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    free = (device.mem_bytes * POOL_FRACTION - model.W_mem
            - ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token
            - store_staging_bytes(model))
    return int(free // model.kappa)


# ---- roofline arithmetic (ported from the exploration's roofline.py)

def _projection_shapes(model: ModelSpec):
    """(in_dim, out_dim) of every dense projection in a layer. Gate
    and up share an input, so they fuse into one GEMM of ffn_width."""
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_head
    inter = model.intermediate
    return ((model.hidden, qkv_out),
            (model.n_q * model.d_head, model.hidden),
            (model.hidden, model.ffn_width),
            (inter, model.hidden))


def compute_knee(model: ModelSpec, device: DeviceSpec) -> float:
    """The chunk size where the dense projections cross the roofline
    ridge and go compute-bound. Combined over all projections:
    intensity = ridge solved for B. ~416 tokens at 4B/H100; the
    measured knee (fixed cost over per-token cost) was 283 - same
    story from the other side."""
    ridge = device.peak_flops / device.hbm_bw
    tot_p = tot_io = 0.0
    for din, dout in _projection_shapes(model):
        tot_p += din * dout
        tot_io += din + dout
    denom = 2.0 * tot_p - ridge * tot_io * ACT_BYTES
    if denom <= 0:
        raise ValueError("projections never cross the ridge")
    return ridge * tot_p * model.w_bytes / denom


def _projection_time(model: ModelSpec, device: DeviceSpec,
                     chunk: int) -> float:
    """Ideal seconds for all dense projections in one chunk, all
    layers: each GEMM takes max(math, memory).

    chunk=0 short-circuits to 0.0 rather than falling through to the
    loop below: `moved`'s weight-read term (params * w_bytes) is
    chunk-independent, so without this guard a zero-token chunk would
    still price reading every weight matrix once - a kernel that
    never launches shouldn't cost anything. Caught by a property test
    (test_budgets_sol_properties.py) checking sol_seconds() of a
    genuinely empty workload is exactly 0.0, not the ~1ms this bug
    produced."""
    if chunk == 0:
        return 0.0
    t = 0.0
    for din, dout in _projection_shapes(model):
        params = din * dout
        flops = 2.0 * params * chunk
        moved = (params * model.w_bytes
                 + chunk * (din + dout) * ACT_BYTES)
        t += max(flops / device.peak_flops, moved / device.hbm_bw)
    return t * model.layers


def _attention_time(model: ModelSpec, device: DeviceSpec,
                    chunk: int, context: int) -> float:
    """Ideal seconds for the attention kernels in one chunk, all
    layers: 4 * B * S * n_q * d_head FLOPs against the KV read plus
    Q/O traffic."""
    flops = 4.0 * chunk * context * model.n_q * model.d_head
    moved = (context * model.kappa / model.layers
             + 2.0 * chunk * model.n_q * model.d_head * ACT_BYTES)
    return max(flops / device.peak_flops,
               moved / device.hbm_bw) * model.layers


def attention_crossover(model: ModelSpec, device: DeviceSpec,
                        chunk_tokens: int | None = None) -> float:
    """The document length where attention pair work overtakes the
    dense projections at the same chunk size. Below it the chunk is a
    GEMM problem ("prefill dominated" holds); above it the quadratic
    term owns the wall. ~12,200 tokens at 4B/H100 at the kernel-cap
    chunk."""
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    t_dense = _projection_time(model, device, chunk_tokens)
    lo, hi = 1.0, 1e7
    for _ in range(200):
        mid = (lo + hi) / 2
        if _attention_time(model, device, chunk_tokens, int(mid)) < t_dense:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---- speed-of-light (issue #26): the floor a query's measured wall
# time can never beat. See reports/2026-08-23-sol-throughput-cost.md
# for the derivation this section implements.
#
# _attention_time (above) models `chunk` query tokens all reading ONE
# shared `context` - exactly what a join stage does (KV rewind: every
# partner reads the anchor's one cached prefix) and what a filter
# stage after the first does (a new predicate's few tokens reading
# the document's already-resident KV). It does NOT model a filter's
# FIRST stage: that's causal self-attention, where each document
# attends only to itself, and different documents in the same batched
# chunk have different lengths - there is no single shared `context`
# to pass in. That needs its own FLOPs count (sum of each document's
# own triangular pair count), so it gets its own function below.
#
# Both functions below take a whole batch (every document, or every
# streaming item, in one query) and aggregate FLOPs and bytes-moved
# BEFORE taking max(), not after: summing many small per-item
# max(compute, memory) calls overstates the floor versus how one
# fused batched kernel actually behaves (sum of maxes >= max of
# sums). Getting this backwards is exactly the kind of SOL-formula
# bug that manufactures a false "efficiency > 100%" alarm.
#
# Neither function prices softmax (the normalize step between the
# two attention matmuls) separately, and that's deliberate, not an
# oversight - checked, not assumed:
#   - Compute: softmax is ~4 elementwise ops per (query, key) score
#     pair, with no d_head factor. The matmul FLOPs above DO have a
#     d_head factor, so softmax adds about 1/d_head of the matmul
#     term (~0.8% at this model's d_head=128) - negligible on its own.
#   - Memory: this is the term that actually matters, and it hinges
#     on kernel fusion. A naive attention pass would write the full
#     (chunk x context) score matrix to HBM for softmax to read back
#     - for a typical streaming item that's MORE bytes than the KV
#     read this module already counts, not negligible at all. The
#     real kernel (executor/attention.py) is FlashAttention-3 with
#     online softmax (`return_softmax_lse=True`) - the score matrix
#     never leaves on-chip memory. The only thing that does is the
#     LSE (log-sum-exp) state, one scalar per (query token, head),
#     which comes out to about 0.1% of the KV traffic already
#     counted. If the engine ever moved to an unfused attention
#     kernel, this omission would need revisiting - it is correct
#     for the kernel this project actually runs, not attention in
#     general.

def spec_ceiling_tokens_per_s(model: ModelSpec, device: DeviceSpec
                              ) -> float:
    """The headline number: tokens/second if every FLOP the card can
    do went into the forward pass and nothing else existed. 2P FLOPs
    per token against the peak rate. ~275k tok/s for Qwen3-4B/H100."""
    return device.peak_flops / (2.0 * model.params)


def elementwise_time(model: ModelSpec, device: DeviceSpec,
                     chunk: int) -> float:
    """Ideal seconds for the per-token tax: two RMSNorms, the fp8
    quantization before each GEMM, the SwiGLU activation, and two
    residual adds. Memory-bound at every batch size - a fixed
    per-token cost with no compute knee, so no max() here."""
    inter = model.intermediate
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_head
    norm = 2 * (2 * model.hidden * ACT_BYTES)
    quant = ((model.hidden + qkv_out + model.hidden + inter)
             * (ACT_BYTES + model.w_bytes))
    swiglu = 3 * inter * ACT_BYTES
    residual = 2 * (3 * model.hidden * ACT_BYTES)
    per_token = norm + quant + swiglu + residual
    return per_token * chunk * model.layers / device.hbm_bw


def _causal_prefill_attention_time(model: ModelSpec, device: DeviceSpec,
                                   doc_lengths) -> float:
    """Ideal seconds for a batch of documents each doing causal
    self-attention over only their own tokens (a filter chain's
    first stage, or a join anchor's first-ever prefix build) - never
    against each other, so this is NOT chunk-tokens-times-one-context.

    Token at position p in a document of length L attends to p prior
    tokens (0-indexed), so one document's pair count is
    L*(L+1)/2 and its FLOPs are 4x that (2 matmuls, 2 FLOPs/pair):
    2*n_q*d_head*L*(L+1). Summed over the batch, then compared
    against the memory side (each document writes its own KV once,
    plus its own Q/O traffic) - same shape as _attention_time's
    `moved` term, with each document supplying its own length as
    both chunk and context since it reads nothing external."""
    doc_lengths = list(doc_lengths)
    chunk = sum(doc_lengths)
    if chunk == 0:
        return 0.0
    flops = 2.0 * model.n_q * model.d_head * sum(
        L * (L + 1) for L in doc_lengths)
    moved = (chunk * model.kappa / model.layers
             + 2.0 * chunk * model.n_q * model.d_head * ACT_BYTES)
    return max(flops / device.peak_flops,
              moved / device.hbm_bw) * model.layers


def _shared_context_attention_time(model: ModelSpec, device: DeviceSpec,
                                   chunks_contexts) -> float:
    """Ideal seconds for a batch of streaming items - a filter's
    later stages, or a join's partners - each a few new tokens
    (`chunk_i`) reading one already-resident, per-item cached prefix
    (`context_i`). This is `_attention_time`'s shape, exactly, just
    aggregated across many small items instead of called once per
    item (see the module note on why: summing per-item max() calls
    would overstate the floor)."""
    chunks_contexts = list(chunks_contexts)
    if not chunks_contexts:
        return 0.0
    flops = sum(4.0 * c * s * model.n_q * model.d_head
               for c, s in chunks_contexts)
    moved = sum(s * model.kappa / model.layers
               + 2.0 * c * model.n_q * model.d_head * ACT_BYTES
               for c, s in chunks_contexts)
    return max(flops / device.peak_flops,
              moved / device.hbm_bw) * model.layers


def sol_seconds(model: ModelSpec, device: DeviceSpec, *,
                causal_doc_lengths, streaming_chunks_contexts) -> float:
    """The floor: minimum seconds to process a query's real workload
    if every kernel ran at peak. `causal_doc_lengths`: one entry per
    document whose KV gets built fresh this query (filter stage 0,
    or a join anchor not already resident from an earlier operator
    and not restored from the store - see the caller in
    runtime/session.py for how "already resident" is tracked so a
    shared document isn't charged for its prefix build twice).
    `streaming_chunks_contexts`: one (chunk_tokens, context_tokens)
    pair per streaming item - a later filter stage's predicate
    against its document, or one join tuple's suffix against its
    anchor's cached prefix.

    Projection and elementwise cost don't care how the total chunk
    is split across documents (dense GEMMs and per-token bookkeeping
    are batch-composition-agnostic), so those run once on the grand
    total. Only attention needs the causal/streaming split, per the
    module note above."""
    return sum(sol_seconds_breakdown(
        model, device, causal_doc_lengths=causal_doc_lengths,
        streaming_chunks_contexts=streaming_chunks_contexts).values())


def sol_seconds_breakdown(model: ModelSpec, device: DeviceSpec, *,
                          causal_doc_lengths, streaming_chunks_contexts
                          ) -> dict:
    """The same four terms sol_seconds() sums, kept separate. Exists
    for two things sol_seconds()'s single number can't do: property
    tests that check one term's behavior in isolation (e.g. that
    causal attention alone scales up with document length, without
    the other three terms diluting the signal), and comparing against
    a real per-kernel-class profiler trace (torch.profiler groups
    kernels into roughly these same four buckets - matmuls, norm/
    quant/activation, and the two attention shapes - so this is the
    number to hold the measured split against, not the summed total).

    Keys: "projection", "elementwise", "causal_attention",
    "streaming_attention" - sol_seconds(...) is exactly sum(this
    dict's values()), kept in sync by construction (sol_seconds calls
    this function rather than duplicating the arithmetic)."""
    causal_doc_lengths = list(causal_doc_lengths)
    streaming_chunks_contexts = list(streaming_chunks_contexts)
    total_chunk = (sum(causal_doc_lengths)
                  + sum(c for c, _ in streaming_chunks_contexts))
    return dict(
        projection=_projection_time(model, device, total_chunk),
        elementwise=elementwise_time(model, device, total_chunk),
        causal_attention=_causal_prefill_attention_time(
            model, device, causal_doc_lengths),
        streaming_attention=_shared_context_attention_time(
            model, device, streaming_chunks_contexts))


# ---- rows that consume a calibration constant

def store_break_even_bytes_per_s(model: ModelSpec,
                                 a_s_per_token: float) -> float:
    """The bandwidth a KV store must beat for restore to win over
    recompute: kappa x the serving rate. About 18 GB/s at 4B/H100
    bf16 KV and the packed rate; pinned host memory's 55 GB/s
    clears it, disk and volumes do not."""
    return model.kappa / a_s_per_token


def derived_table(model: ModelSpec, device: DeviceSpec,
                  a_s_per_token: float) -> dict:
    """The full section-8 table, for explain() and the tests."""
    chunk = chunk_budget(model, device)
    return {
        "tensor_parallel": tensor_parallel(model, device),
        "arena_tokens": arena_tokens(model, device, chunk),
        "chunk_memory_bound": chunk_memory_bound(model, device),
        "kernel_index_cap": kernel_index_cap(model),
        "chunk_budget": chunk,
        "compute_knee": compute_knee(model, device),
        "attention_crossover": attention_crossover(model, device, chunk),
        "store_break_even_bytes_per_s":
            store_break_even_bytes_per_s(model, a_s_per_token),
        "serving_rate_tokens_per_s": 1.0 / a_s_per_token,
    }
