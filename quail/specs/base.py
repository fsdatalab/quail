"""Model and device spec structs for the planner."""

from dataclasses import dataclass, replace
from typing import Literal

Precision = Literal["fp8", "bf16"]
# What a model answers with: a generative model scores TRUE against
# FALSE for AI_FILTER and AI_JOIN; a reranker scores yes against no
# for AI.SCORE.
Role = Literal["generative", "reranker"]

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
    revision: str = ""   # pinned hub commit hash; resolves from the
    #                      HF cache without the API round trips a
    #                      branch name pays. "" tracks the default.
    kv_bytes: float = 2.0    # KV bytes per element (bf16 default)
    w_mem_bytes: float = 0.0    # measured weight footprint as loaded;
    #                             0 falls back to params * w_bytes.
    #                             Embeddings and quant scales sit
    #                             outside the dense param count, so the
    #                             measured number is larger.
    vocab: int = 0       # vocabulary rows in the embedding and lm_head
    tied_head: bool = False    # lm_head shares the embedding tensor
    weight_precision: Precision = "fp8"
    attention_precision: Precision = "bf16"
    arch: str = "qwen3"    # forward pass in executor/models/<arch>.py
    role: Role = "generative"

    @property
    def kappa(self) -> float:
        """KV bytes per cached token: 2 * L * n_kv * d_head * kv_bytes."""
        return 2 * self.layers * self.n_kv * self.d_head * self.kv_bytes

    @property
    def kv_elements_per_token(self) -> int:
        """KV elements per token, dtype-free: 2 * L * n_kv * d_head."""
        return 2 * self.layers * self.n_kv * self.d_head

    # W_mem and W_resident are the weight-memory names used throughout
    # the code and the engine wiki.
    @property
    def W_mem(self) -> float:  # noqa: N802
        """Weight bytes as loaded, before the full untied head is discarded."""
        return self.w_mem_bytes or self.params * self.w_bytes

    @property
    def head_mem_bytes(self) -> float:
        """Bytes of an untied bf16 lm_head weight; 0 when tied.

        The executor discards this matrix after retaining its TRUE/FALSE
        rows. Both Qwen3 checkpoints store the
        head in bf16, hence the 2 bytes per element.
        """
        if self.tied_head:
            return 0.0
        return self.vocab * self.hidden * 2.0

    @property
    def W_resident(self) -> float:  # noqa: N802
        """Weight bytes resident on the GPU after boot."""
        return self.W_mem - self.head_mem_bytes

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
    usd_per_hour: float = 0.0    # rental price of one device; 0 means
    #                              no price is known
    price_source: str = ""       # where usd_per_hour was read from

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
