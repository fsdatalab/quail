"""Model and device spec structs for the planner."""

from dataclasses import dataclass, replace
from typing import Literal


Precision = Literal["fp8", "bf16"]

# Peak per-token activation bytes per hidden dim.
ACT_BYTES_PER_HIDDEN = 32


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float        # P: dense parameter count; 2P FLOPs per token
    layers: int          # L
    hidden: int          # h
    n_q: int             # query heads
    n_kv: int            # KV heads
    d_head: int          # head dim
    ffn_width: int       # widest projection's output columns
    #                      (gate_up: 2 x intermediate at Qwen3)
    w_bytes: float       # weight bytes per param (fp8 = 1)
    hf_name: str = ""    # the checkpoint (weights + tokenizer)
    kv_bytes: float = 2.0    # KV bytes per element (bf16 default)
    w_mem_bytes: float = 0.0    # measured resident weight footprint;
    #                             0 falls back to params * w_bytes.
    #                             Embeddings and quant scales sit
    #                             outside the dense param count, so the
    #                             measured number is larger.
    weight_precision: Precision = "fp8"
    attention_precision: Precision = "bf16"

    @property
    def kappa(self) -> float:
        """KV bytes per cached token: 2 * L * n_kv * d_head * kv_bytes."""
        return 2 * self.layers * self.n_kv * self.d_head * self.kv_bytes

    @property
    def kv_elements_per_token(self) -> int:
        """KV elements per token, dtype-free: 2 * L * n_kv * d_head."""
        return 2 * self.layers * self.n_kv * self.d_head

    @property
    def W_mem(self) -> float:
        """Resident weight bytes."""
        return self.w_mem_bytes or self.params * self.w_bytes

    @property
    def act_per_token(self) -> float:
        """Peak activation bytes per batched token."""
        return ACT_BYTES_PER_HIDDEN * self.hidden

    @property
    def intermediate(self) -> int:
        """MLP intermediate width; gate_up packs two of them."""
        return self.ffn_width // 2

    def with_kv_bytes(self, kv_bytes: float) -> "ModelSpec":
        return replace(self, kv_bytes=kv_bytes)


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    mem_bytes: float     # M: physical memory
    hbm_bw: float        # BW: memory bandwidth, bytes/s
    peak_flops: float    # R_D at the executor's compute dtype
    bf16_flops: float = 0.0    # dense bf16 peak. Attention runs in
    #                            bf16 (FlashAttention-3 over bf16 KV),
    #                            so the pair FLOPs price against this,
    #                            not against the fp8 ceiling. 0 falls
    #                            back to half of peak_flops, the fp8-
    #                            to-bf16 ratio on every tensor core
    #                            generation we run on.

    @property
    def attn_flops(self) -> float:
        """Dense peak for the attention kernels, FLOP/s."""
        return self.bf16_flops or self.peak_flops / 2

    def arithmetic_bandwidth(self, precision: Precision) -> float:
        """Return arithmetic throughput for one component precision."""

        if precision == "fp8":
            return self.peak_flops
        if precision == "bf16":
            return self.attn_flops
        raise ValueError(f"unsupported precision: {precision}")
