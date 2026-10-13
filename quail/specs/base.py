"""Model and device spec structs for the planner."""

import bisect
import math
from dataclasses import dataclass
from typing import Literal

Precision = Literal["fp8", "bf16", "fp32"]
# What a model answers with: a generative model scores TRUE against
# FALSE for AI_FILTER and AI_JOIN; a reranker scores yes against no
# for AI.SCORE; a decision model scores its yes and no options with
# its own head for AI_FILTER and AI_JOIN.
Role = Literal["generative", "reranker", "decision"]

# Peak per-token activation bytes per hidden dim.
ACT_BYTES_PER_HIDDEN = 32


# Every random canvas token comes from this seed: a filter's fixed
# canvas, and a classified document's canvas with the document's row.
CANVAS_SEED = 0
# A canvas answer whose first draw is less certain than this, in nats,
# takes the rest of its noise draws and averages them.
CANVAS_ENTROPY_NATS = 0.1


@dataclass(frozen=True)
class AnswerCanvas:
    """Token layout for a diffusion model's classification answer.

    Attributes:
        rows: Rows of the canvas packed after the classification cue.
        turn_close_id: The token that closes the model's turn, seeded
            after the answer's row.
        pad_id: The padding token, which fills the rows after it.
    """

    rows: int
    turn_close_id: int
    pad_id: int


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
    decision_head_dim: int = 0    # width of a decision model's option
    #                               head (its key, query, and MLP rows);
    #                               the head's FLOPs and bytes follow
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
    answer_canvas: "AnswerCanvas | None" = None    # a diffusion model's
    #                                        answer canvas, which the
    #                                        letters rule reads
    turn_prefix: str = ""     # chat-turn text before every prompt
    turn_suffix: str = ""     # chat-turn text after the answer cue
    extract_turn_prefix: str = ""    # chat-turn text before an AI.EXTRACT
    #                                  prompt, when it differs from
    #                                  turn_prefix; a prefix unlike the
    #                                  AI.IF turn's keeps an extraction
    #                                  from reading a filter's document KV
    extract_turn_suffix: str = ""    # chat-turn text between an AI.EXTRACT
    #                                  prompt's answer template and its cue
    prompt_format: str = "raw-v1"    # names the turn layout in run records
    prompt_layout: str = "ai-if"     # the PROMPT_LAYOUTS entry AI.IF
    #                                  prompts are written in
    chunk_cap_tokens: int = 0    # upper bound on tokens per chunk; 0
    #                              leaves the memory and kernel bounds
    moe_backend: str | None = None    # vLLM's moe_backend setting, one
    #                                   of its MoEBackend names ("triton",
    #                                   "cutlass", "deep_gemm", ...);
    #                                   None lets vLLM pick

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

    @property
    def extract_turn(self) -> tuple[str, str]:
        """The chat-turn text wrapped around an AI.EXTRACT prompt."""
        if self.extract_turn_prefix or self.extract_turn_suffix:
            return self.extract_turn_prefix, self.extract_turn_suffix
        return self.turn

    # W_mem and W_resident are the weight-memory names used throughout
    # the code and the engine wiki.
    @property
    def W_mem(self) -> float:  # noqa: N802
        """Weight bytes as loaded, before the full untied head is discarded."""
        return self.w_mem_bytes or self.params * self.w_bytes

    @property
    def head_mem_bytes(self) -> float:
        """Return untied output-head memory in bytes, or zero for tied weights.

        The executor keeps this matrix for AI.CLASSIFY, so it stays
        resident beside the other weights. Both Qwen3 checkpoints store
        the head in bf16, hence the 2 bytes per element.
        """
        if self.tied_head:
            return 0.0
        return self.vocab * self.hidden * 2.0

    @property
    def W_resident(self) -> float:  # noqa: N802
        """Return model weight bytes resident on the GPU, including the output head."""
        return self.W_mem

    @property
    def act_per_token(self) -> float:
        """Peak activation bytes per batched token."""
        return ACT_BYTES_PER_HIDDEN * self.hidden

    @property
    def intermediate(self) -> int:
        """MLP intermediate width; gate_up packs two of them."""
        return self.ffn_width // 2



def _log_interp(xs, ys, x: float) -> float:
    """Interpolate ys over increasing xs linearly in log x, clamped at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x)
    t = (math.log(x) - math.log(xs[i - 1])) / (math.log(xs[i]) - math.log(xs[i - 1]))
    return ys[i - 1] + t * (ys[i] - ys[i - 1])


def causal_rate(tokens_axis, rates, tokens: float) -> float:
    """Return FLOP/s of causal attention over documents of this many tokens.

    Below the shortest measured length the rate falls in proportion to
    the length; past the longest it is held.
    """
    low = tokens_axis[0]
    if tokens < low:
        return rates[0] * max(tokens, 1.0) / low
    return _log_interp(tokens_axis, rates, tokens)


@dataclass(frozen=True)
class AttentionRates:
    """Measured attention FLOP rates for one head layout on one device.

    Below the shortest document or the fewest query rows measured, the
    time of a call no longer falls with its size, so the rate falls in
    proportion to it. Past the other ends the rate is held.

    Attributes:
        causal_tokens: Document lengths of the causal measurements.
        causal: FLOP/s of causal attention over packed documents of each
            length.
        read_rows: Query rows per sequence of the cached-read
            measurements.
        read_contexts: Cached context tokens of the cached-read
            measurements.
        read: FLOP/s of query rows reading cached KV, one row of the
            table per read_rows value, one column per read_contexts value.
    """

    causal_tokens: tuple[int, ...]
    causal: tuple[float, ...]
    read_rows: tuple[int, ...]
    read_contexts: tuple[int, ...]
    read: tuple[tuple[float, ...], ...]

    def causal_rate(self, tokens: float) -> float:
        """Return FLOP/s of causal attention over documents of this length."""
        return causal_rate(self.causal_tokens, self.causal, tokens)

    def read_rate(self, rows: float, context: float) -> float:
        """Return FLOP/s of this many query rows per sequence reading cached KV."""
        low = self.read_rows[0]
        scale = max(rows, 1.0) / low if rows < low else 1.0
        by_rows = [_log_interp(self.read_contexts, line, context)
                   for line in self.read]
        return _log_interp(self.read_rows, by_rows, max(rows, low)) * scale


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
    fp32_flops: float = 0.0    # fp32 peak on the CUDA cores, not the
    #                            tensor cores: the decision head runs
    #                            there. 0 means no rate is known.
    usd_per_hour: float = 0.0    # rental price of one device; 0 means
    #                              no price is known
    price_source: str = ""       # where usd_per_hour was read from
    sustained_fp8_flops: float = 0.0    # measured fp8 GEMM rate held
    #                                     at the power limit; 0 means
    #                                     none was measured
    sustained_hbm_bw: float = 0.0       # measured device-to-device copy
    #                                     bandwidth, bytes/s; 0 means
    #                                     none was measured
    attention_rates: tuple[tuple[tuple[int, int, int], AttentionRates], ...] = ()
    #   measured attention rates keyed by (query heads, KV heads, head
    #   dimension); a layout without an entry prices at attn_flops

    @property
    def attn_flops(self) -> float:
        """Dense peak for the attention kernels, FLOP/s."""
        return self.bf16_flops or self.peak_flops / 2

    def attention_rates_for(self, n_q: int, n_kv: int,
                            d_head: int) -> AttentionRates | None:
        """Return the measured attention rates of a head layout, or None."""
        for layout, rates in self.attention_rates:
            if layout == (n_q, n_kv, d_head):
                return rates
        return None

    @property
    def gemm_flops(self) -> float:
        """Return the fp8 GEMM rate: sustained when measured, else the peak."""
        return self.sustained_fp8_flops or self.peak_flops

    @property
    def memory_bw(self) -> float:
        """Return memory bandwidth in bytes/s: sustained when measured, else spec."""
        return self.sustained_hbm_bw or self.hbm_bw

    def arithmetic_bandwidth(self, precision: Precision) -> float:
        """Return arithmetic throughput for one component precision."""
        if precision == "fp8":
            return self.gemm_flops
        if precision == "bf16":
            return self.attn_flops
        if precision == "fp32" and self.fp32_flops:
            return self.fp32_flops
        raise ValueError(f"unsupported precision: {precision}")
