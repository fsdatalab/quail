"""Batch statistics and the analytical latency tau_0 (paper secs. 3-4).

A batch is a list of Ops. Three op kinds cover every policy family:

  prompt_prefill  -- prefill of the shared task-prompt block for filter j
                     (task-first template [F_j][D_i]; once per query, pinnable)
  doc_chunk       -- Delta_i new document tokens for doc i under a context:
                     ctx='task' (cached = p_j + r_i) or ctx='doc' (cached = r_i)
  branch          -- p_j prompt tokens evaluated after a complete document
                     prefix (document-first / speculative template); cached = d_i

KV accounting follows the paper's primary write-through fused ledger
(sec. 4.5 of the revised paper): every new KV position with a later causal
consumer has one store event; only a terminal leaf position with no
descendant is ephemeral. Fusion removes later loads, never the store.
Concretely, per op, K_W counts op.new - op.ephemeral_tail where
ephemeral_tail is 1 for a branch (its final decision position has no
descendant) and 1 for a task-context doc chunk that completes the document
(the decision is read at the final document token), else 0. Interior prompt
and document tokens are always stored because later tokens of the same
sequence, or later branches, attend to them.

K_L counts HBM load events in token spans. Each batch is modeled as one
supported fused producer-consumer group, so blocks produced earlier in the
batch are streamed (no load event); only launch-resident blocks create
loads, deduplicated per physical block (tree-aware kernel assumption).

Peak memory uses the conservative lifetime convention that all of a batch's
new tokens are live simultaneously with the launch-resident set. Blocks
retained under any outcome must stay live through the outcome boundary, so
the convention satisfies the paper's exact ordinal rule; it can only
overestimate the true peak.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from .configs import DeviceConfig, ModelConfig


def a_pairs(c: int, q: int) -> int:
    """Allowed query-key pairs of a linear segment: eq. (14)."""
    return c * q + q * (q + 1) // 2


@dataclass(frozen=True)
class Op:
    kind: str                 # 'prompt_prefill' | 'doc_chunk' | 'branch'
    doc: int                  # document index; -1 for prompt_prefill
    stage: int                # filter j (1-based); doc_chunk under ctx='doc': 0
    new: int                  # new tokens evaluated by this op
    cached_resident: int      # prefix tokens resident at batch start visible to op
    cached_inbatch: int = 0   # prefix tokens produced earlier in this same batch
    read_blocks: tuple = ()   # physical block ids read (resident at batch start)
    ephemeral_tail: int = 0   # trailing positions with no causal descendant
    # block id conventions: ('prompt', j) shared task-prompt block;
    # ('doc', i) document prefix under doc-first; ('taskdoc', i) document
    # prefix under the current task template (specific to (i, z_i)).

    @property
    def cached(self) -> int:
        return self.cached_resident + self.cached_inbatch


@dataclass
class BatchStats:
    U: int = 0
    A: int = 0
    K_L: int = 0          # token spans over HBM load events (launch-resident reads)
    K_W: int = 0          # new positions stored under the write-through ledger
    K_tmp: int = 0        # peak additional token-equivalent KV during the batch
    n_segments: int = 0   # distinct causal segments (for tau_theta's beta_Q term)


def batch_stats(ops: Iterable[Op], resident_block_tokens: dict) -> BatchStats:
    """Compute U, A, K_L, K_W, K_tmp for a batch.

    resident_block_tokens maps physical block id -> resident token count at
    batch start, used to deduplicate K_L across ops sharing a block (the
    paper's one-load credit for a shared block in one load group)."""
    st = BatchStats()
    read: dict = {}
    for op in ops:
        st.U += op.new
        st.A += a_pairs(op.cached, op.new)
        st.n_segments += 1
        st.K_tmp += op.new
        st.K_W += op.new - op.ephemeral_tail
        for b in op.read_blocks:
            if b not in resident_block_tokens:
                raise ValueError(f"op reads non-resident block {b}")
            read[b] = resident_block_tokens[b]
    st.K_L = sum(read.values())
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
              K_L: int, K_W: int) -> float:
    """H(B), eq. (25), with the attention width w_Q = n_Q * d_h."""
    f_a = 4.0 * model.L * model.attn_width * A
    b_kv = model.kappa * (K_L + K_W)
    return max(f_a / device.R_A, b_kv / device.BW)


def tau(model: ModelConfig, device: DeviceConfig, st: BatchStats,
        params: Optional[CostParams] = None) -> float:
    """tau_theta(B), eq. (26)/(27). Default params give tau_0."""
    p = params or CostParams()
    D = dense_time(model, device, st.U)
    H = attn_time(model, device, st.A, st.K_L, st.K_W)
    return p.beta_0 + p.beta_D * D + p.beta_H * H + p.beta_U * st.U \
        + p.beta_Q * st.n_segments


def peak_memory(model: ModelConfig, resident_tokens: int, k_tmp: int) -> float:
    """LHS of the peak-memory condition under the conservative lifetime rule."""
    return model.W_mem + model.kappa * (resident_tokens + k_tmp)


def memory_ok(model: ModelConfig, device: DeviceConfig,
              resident_tokens: int, k_tmp: int) -> bool:
    return peak_memory(model, resident_tokens, k_tmp) + device.S <= device.M


def kv_capacity_tokens(model: ModelConfig, device: DeviceConfig) -> int:
    """Free-for-KV capacity in tokens: (M - W_mem - S) / kappa."""
    return int((device.M - model.W_mem - device.S) // model.kappa)
