"""Every derived quantity in the design's spec table (engine_design.md
section 8). Pure arithmetic over the two spec structs; the only
measured inputs are the calibration constants.

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


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None) -> int:
    """The admission budget: tokens of document KV resident at once.
    What is left after weights and the chunk's activation reservation,
    in KV bytes. Token-based, never a document count (a count cannot
    see length; the 4,096-seq default thrashed at 2.40x reads)."""
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    free = (device.mem_bytes * POOL_FRACTION - model.W_mem
            - ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token)
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
    layers: each GEMM takes max(math, memory)."""
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
        "serving_rate_tokens_per_s": 1.0 / a_s_per_token,
    }
