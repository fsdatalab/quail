"""Budget quantities derived from model and device specs.

Chunk budget, admission budget, and roofline arithmetic.
"""

from quail.specs import DeviceSpec, ModelSpec

POOL_FRACTION = 0.95    # fraction of device memory the executor claims
CHUNK_SLACK = 2         # slack factor on the activation bound
PAGE_TOKENS = 16        # KV arena page size, tokens
INT32_MAX = 2**31 - 1
ACT_BYTES = 2           # bf16 activations, bytes per element
ACT_RESERVE_CHUNKS = 2  # chunks of activation memory reserved outside
#                         the arena for overlapped chunk construction


def minimum_weight_gpus(model: ModelSpec, device: DeviceSpec) -> int:
    """Return how many pooled GPU memories would hold the loaded weights.

    Quail does not split weights across GPUs. A result above one causes a
    planning refusal.
    """
    gpus = 1
    while model.W_mem > device.mem_bytes * POOL_FRACTION * gpus:
        gpus *= 2
    return gpus


def kernel_index_cap(model: ModelSpec) -> int:
    """Max tokens per chunk from the int32 element-offset limit.

    Constraint: rows x ffn_width < 2^31.
    """
    return INT32_MAX // model.ffn_width


def chunk_memory_bound(model: ModelSpec, device: DeviceSpec) -> int:
    """Tokens per chunk the activation memory allows, with slack.

    Uses resident weights after the full untied output head is discarded.
    """
    free = device.mem_bytes * POOL_FRACTION - model.W_resident
    return int(free // model.act_per_token) // CHUNK_SLACK


def chunk_budget(model: ModelSpec, device: DeviceSpec) -> int:
    """Effective chunk budget, floored at the compute knee.

    The budget is min(memory bound, kernel index cap) before the floor.
    """
    b = min(chunk_memory_bound(model, device), kernel_index_cap(model))
    return max(b, int(compute_knee(model, device)))


def arena_tokens(model: ModelSpec, device: DeviceSpec,
                 chunk_tokens: int | None = None) -> int:
    """Admission budget: tokens of document KV that can be resident at once.

    Computed from the memory left after resident weights and the
    activation reservation.
    """
    if chunk_tokens is None:
        chunk_tokens = chunk_budget(model, device)
    free = (device.mem_bytes * POOL_FRACTION - model.W_resident
            - ACT_RESERVE_CHUNKS * chunk_tokens * model.act_per_token)
    return int(free // model.kappa)


# ---- roofline arithmetic

def _projection_shapes(model: ModelSpec):
    """Return (in_dim, out_dim) of every dense projection in a layer."""
    qkv_out = (model.n_q + 2 * model.n_kv) * model.d_head
    inter = model.intermediate
    return ((model.hidden, qkv_out),
            (model.n_q * model.d_head, model.hidden),
            (model.hidden, model.ffn_width),
            (inter, model.hidden))


def compute_knee(model: ModelSpec, device: DeviceSpec) -> float:
    """Chunk size (tokens) where the dense projections become compute-bound.

    That is where they cross the roofline ridge.
    """
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
    """Ideal seconds for all dense projections in one chunk, all layers."""
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
    """Ideal seconds for the attention kernels in one chunk, all layers."""
    flops = 4.0 * chunk * context * model.n_q * model.d_head
    moved = (context * model.kappa / model.layers
             + 2.0 * chunk * model.n_q * model.d_head * ACT_BYTES)
    return max(flops / device.peak_flops,
               moved / device.hbm_bw) * model.layers


def attention_crossover(model: ModelSpec, device: DeviceSpec,
                        chunk_tokens: int | None = None) -> float:
    """Document length (tokens) where attention overtakes dense projections.

    At the given chunk size.
    """
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


def derived_table(model: ModelSpec, device: DeviceSpec) -> dict:
    """Return all derived budget quantities as a dict."""
    chunk = chunk_budget(model, device)
    return {
        "minimum_weight_gpus": minimum_weight_gpus(model, device),
        "arena_tokens": arena_tokens(model, device, chunk),
        "chunk_memory_bound": chunk_memory_bound(model, device),
        "kernel_index_cap": kernel_index_cap(model),
        "chunk_budget": chunk,
        "compute_knee": compute_knee(model, device),
        "attention_crossover": attention_crossover(model, device, chunk),
    }
