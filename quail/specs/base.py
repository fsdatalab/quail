"""Model and device spec structs for the planner."""

from dataclasses import dataclass, replace
from typing import Literal

Precision = Literal["fp8", "bf16"]
# What a model answers with: a generative model scores TRUE against
# FALSE for AI_FILTER and AI_JOIN; a reranker scores yes against no
# for AI.SCORE.
Role = Literal["generative", "reranker"]
InputModality = Literal["text", "image"]

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
    # Layers that keep every token of KV, when the model mixes them
    # with sliding layers: every full_attention_period-th layer, with
    # its own KV geometry. 0 means every layer looks like n_kv x d_head.
    full_attention_period: int = 0
    full_n_kv: int = 0         # KV heads of a full-attention layer
    full_d_head: int = 0       # head dim of a full-attention layer
    sliding_window: int = 0    # tokens the other layers see behind each
    #                            row; 0 means no layer slides
    # Experts beside the dense MLP, for a mixture-of-experts model;
    # 0 means the dense MLP alone.
    experts: int = 0
    experts_active: int = 0    # experts one token multiplies
    expert_intermediate: int = 0    # one expert's MLP width
    canvas_tokens: int = 0    # rows a diffusion model appends after
    #                           the answer cue. 0 for an autoregressive
    #                           model, which answers at the last
    #                           prompt row.
    canvas_answer_row: int = 0    # canvas row the answer is read at;
    #                               later than 0 when the model opens
    #                               its turn with fixed tokens, such as
    #                               an empty thinking channel
    turn_prefix: str = ""     # chat-turn text before every prompt
    turn_suffix: str = ""     # chat-turn text after the answer cue
    prompt_format: str = "raw-v1"    # names the turn layout in run records
    chunk_cap_tokens: int = 0    # upper bound on tokens per chunk; 0
    #                              leaves the memory and kernel bounds
    moe_backend: str | None = None    # vLLM's moe_backend setting, one
    #                                   of its MoEBackend names ("triton",
    #                                   "cutlass", "deep_gemm", ...);
    #                                   None lets vLLM pick
    # Image input. A text-only model leaves these at their defaults.
    input_modalities: frozenset[InputModality] = frozenset({"text"})
    image_token_budgets: tuple[int, ...] = ()    # soft tokens one image
    #                                              may use, as the image
    #                                              processor accepts
    default_image_tokens: int = 0    # budget when EngineConfig names none
    max_images_per_request: int | None = None    # largest image count
    #                                              one prompt was tested
    #                                              with; None: untested
    image_patch_pixels: int = 0    # side of one vision patch, in pixels
    image_pool_kernel: int = 0     # patches pooled into one soft token,
    #                                per side
    image_frame_tokens: int = 0    # tokens around one image's soft
    #                                tokens (begin and end markers)

    def is_full_layer(self, layer: int) -> bool:
        """Whether the layer keeps every token with the full KV geometry."""
        period = self.full_attention_period
        return bool(period) and (layer + 1) % period == 0

    @property
    def kv_shapes(self) -> tuple:
        """Per layer, the (KV heads, head dim) its KV stores."""
        full = (self.full_n_kv or self.n_kv, self.full_d_head or self.d_head)
        return tuple(full if self.is_full_layer(i) else (self.n_kv, self.d_head)
                     for i in range(self.layers))

    @property
    def widest_projection(self) -> int:
        """Output columns of the widest dense projection in any layer."""
        return max(self.ffn_width, max(
            (self.n_q + 2 * n_kv) * d_head for n_kv, d_head in self.kv_shapes))

    @property
    def kappa(self) -> float:
        """KV bytes per cached token, summed over the layers."""
        return self.kv_elements_per_token * self.kv_bytes

    @property
    def kv_elements_per_token(self) -> int:
        """KV elements per token, dtype-free: 2 * sum of n_kv * d_head."""
        return 2 * sum(n_kv * d_head for n_kv, d_head in self.kv_shapes)

    @property
    def sliding_layer_set(self) -> frozenset:
        """The layers that keep only the last sliding_window tokens of KV."""
        if not self.sliding_window:
            return frozenset()
        return frozenset(i for i in range(self.layers)
                         if not self.is_full_layer(i))

    @property
    def kappa_sliding(self) -> float:
        """KV bytes per token on the sliding layers alone."""
        sliding = self.sliding_layer_set
        return 2 * self.kv_bytes * sum(
            n_kv * d_head for i, (n_kv, d_head) in enumerate(self.kv_shapes)
            if i in sliding)

    @property
    def kappa_full(self) -> float:
        """KV bytes per token on the layers that keep every token."""
        return self.kappa - self.kappa_sliding

    @property
    def turn(self) -> tuple[str, str]:
        """The chat-turn text wrapped around every prompt."""
        return self.turn_prefix, self.turn_suffix

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
