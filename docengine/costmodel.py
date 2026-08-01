"""Batch statistics and the analytical latency tau_0 (paper sec. 3-4).

A batch is a list of Ops. Three op kinds cover every policy family:

  prompt_prefill  -- prefill of the shared task-prompt block for filter j
                     (task-first template [F_j][D_i]; once per query, pinnable)
  doc_chunk       -- Delta_i new document tokens for doc i under a context:
                     ctx='task' (cached = p_j + r_i) or ctx='doc' (cached = r_i)
  branch          -- p_j prompt tokens evaluated after a complete document
                     prefix (document-first / speculative template); cached = d_i

Accounting conventions (C3):
  K_W = new doc_chunk tokens + new prompt_prefill tokens (their KV has future
        consumers: later chunks, branches, or batches). branch tokens end at
        logits and are never written.
  K_tmp = all new tokens of the batch (doc chunks + prompt blocks + branches)
        live simultaneously at the batch peak; branch KV is discarded at the
        boundary, doc/prompt KV persists into K_{t+1}.
  K_R = resident tokens read, deduplicated per physical block: a shared task
        prompt block is counted once per batch regardless of how many docs use
        it; a document prefix block is counted once even if several branches
        of that doc read it (tree-aware kernel assumption, paper sec. 4.5).
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .configs import DeviceConfig, ModelConfig


def a_pairs(c: int, q: int) -> int:
    """Allowed query-key pairs of a linear segment: eq. (14)."""
    return c * q + q * (q + 1) // 2


@dataclass(frozen=True)
class Op:
    kind: str                 # 'prompt_prefill' | 'doc_chunk' | 'branch'
    doc: int                  # document index; -1 for prompt_prefill
    stage: int                # filter j (1-based); for doc_chunk under ctx='doc': 0
    new: int                  # new tokens evaluated by this op
    cached_resident: int      # prefix tokens resident at batch start visible to op
    cached_inbatch: int = 0   # prefix tokens produced earlier in this same batch
    read_blocks: tuple = ()   # physical block ids read (resident at batch start)
    # block id conventions: ('prompt', j) shared task-prompt block;
    # ('doc', i) document prefix under doc-first; ('taskdoc', i, j) document
    # prefix under the F_j task template (specific to (i,j), paper Table 1).

    @property
    def cached(self) -> int:
        return self.cached_resident + self.cached_inbatch


@dataclass
class BatchStats:
    U: int = 0
    A: int = 0
    K_R: int = 0          # deduplicated resident tokens read
    K_W: int = 0          # new token positions written for a future consumer
    K_tmp: int = 0        # peak additional token-equivalent KV during the batch
    n_segments: int = 0   # distinct causal segments (for tau_theta's beta_Q term)


def batch_stats(ops: Iterable[Op], resident_block_tokens: dict) -> BatchStats:
    """Compute U, A, K_R, K_W, K_tmp for a batch.

    resident_block_tokens maps physical block id -> resident token count at
    batch start, used to deduplicate K_R across ops sharing a block. An op's
    read_blocks must reference only blocks present in this map; the tokens it
    actually reads from a block are min(op.cached_resident allocation) --
    here we charge the full referenced extent per block once (union), which
    matches the paper's "union of physical resident KV token blocks".
    """
    st = BatchStats()
    read: dict = {}
    for op in ops:
        st.U += op.new
        st.A += a_pairs(op.cached, op.new)
        st.n_segments += 1
        st.K_tmp += op.new
        if op.kind in ("doc_chunk", "prompt_prefill"):
            st.K_W += op.new
        for b in op.read_blocks:
            if b not in resident_block_tokens:
                raise ValueError(f"op reads non-resident block {b}")
            # union semantics: full resident extent of the block, once
            read[b] = resident_block_tokens[b]
    st.K_R = sum(read.values())
    return st


@dataclass(frozen=True)
class CostParams:
    """tau_theta coefficients (eq. 27); the analytical tau_0 uses defaults."""
    beta_0: float = 0.0
    beta_D: float = 1.0
    beta_H: float = 1.0
    beta_U: float = 0.0
    beta_Q: float = 0.0


def dense_time(model: ModelConfig, device: DeviceConfig, U: int) -> float:
    """D(B), eq. (24)."""
    if U <= 0:
        return 0.0
    return max(2.0 * model.P * U / device.R_D, model.W_run / device.BW)


def attn_time(model: ModelConfig, device: DeviceConfig, A: int,
              K_R: int, K_W: int) -> float:
    """H(B), eq. (25), with the C1 attention width n_q*d_h."""
    f_a = 4.0 * model.L * model.attn_width * A
    b_kv = model.kappa * (K_R + K_W)
    return max(f_a / device.R_A, b_kv / device.BW)


def tau(model: ModelConfig, device: DeviceConfig, st: BatchStats,
        params: Optional[CostParams] = None) -> float:
    """tau_theta(B), eq. (26)/(27). Default params give tau_0."""
    p = params or CostParams()
    D = dense_time(model, device, st.U)
    H = attn_time(model, device, st.A, st.K_R, st.K_W)
    return p.beta_0 + p.beta_D * D + p.beta_H * H + p.beta_U * st.U \
        + p.beta_Q * st.n_segments


def peak_memory(model: ModelConfig, resident_tokens: int, k_tmp: int) -> float:
    """LHS of eq. (22)."""
    return model.W_mem + model.kappa * (resident_tokens + k_tmp)


def memory_ok(model: ModelConfig, device: DeviceConfig,
              resident_tokens: int, k_tmp: int) -> bool:
    return peak_memory(model, resident_tokens, k_tmp) + device.S <= device.M


def kv_capacity_tokens(model: ModelConfig, device: DeviceConfig) -> int:
    """Free-for-KV capacity in tokens: (M - W_mem - S) / kappa."""
    return int((device.M - model.W_mem - device.S) // model.kappa)
