"""Model and device spec structs for the planner."""

import json
from dataclasses import asdict, dataclass
from typing import Literal, get_args

import _quail_native

Precision = Literal["fp4", "fp8", "bf16", "fp32"]
Role = Literal["generative", "reranker", "decision"]
LayerKind = Literal[
    "full_attention", "sliding_attention", "gated_delta_net", "short_conv"]

GDN_STATE_ELEMENT_BYTES = _quail_native.GDN_STATE_ELEMENT_BYTES
CONV_STATE_ELEMENT_BYTES = _quail_native.CONV_STATE_ELEMENT_BYTES

ACT_BYTES_PER_HIDDEN = _quail_native.ACT_BYTES_PER_HIDDEN


CANVAS_SEED = 0
CANVAS_ENTROPY_NATS = 0.1


@dataclass(frozen=True)
class SupportResult:
    """Result of checking one backend configuration."""

    supported: bool
    reason: str | None = None

    @classmethod
    def accept(cls) -> "SupportResult":
        return cls(True)

    @classmethod
    def reject(cls, reason: str) -> "SupportResult":
        return cls(False, reason)


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
class SaeSpec:
    """A sparse autoencoder (SAE) read beside a model, for AI.SAE_FEATURES.

    A model with an SAE returns, from its forward pass, the residual
    stream after its last loaded layer instead of the final-normed
    rows, and the SAE encodes that stream. Such a model serves feature
    calls only.

    Attributes:
        hf_name: Hugging Face repo of the SAE checkpoints.
        revision: Pinned hub commit hash.
        file: The read layer's tensors, W_enc and b_enc.
        num_features: Encoder rows, one per feature.
        keep: Features kept per token, the SAE's top k.
    """

    hf_name: str
    revision: str
    file: str
    num_features: int
    keep: int


def _tuples(value):
    """A JSON value with its lists as tuples, as the spec's fields hold them."""
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float
    layers: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    d_head: int
    ffn_width: int
    w_bytes: float
    hf_name: str = ""
    revision: str = ""
    kv_bytes: float = 2.0
    w_mem_bytes: float = 0.0
    vocab: int = 0
    tied_head: bool = False
    weight_precision: Precision = "fp8"
    attention_precision: Precision = "bf16"
    arch: str = "qwen3"
    role: Role = "generative"
    readout: str = ""
    decision_head_dim: int = 0
    full_num_kv_heads: int = 0
    full_d_head: int = 0
    sliding_window: int = 0
    experts: int = 0
    experts_active: int = 0
    expert_intermediate: int = 0
    canvas_tokens: int = 0
    canvas_answer_row: int = 0
    answer_canvas: "AnswerCanvas | None" = None
    turn_prefix: str = ""
    turn_suffix: str = ""
    bos_text: str = ""
    answer_temperature: float = 1.0
    read_found: tuple[tuple[float, float], ...] = ()
    read_span_tokens: int = 0
    set_size: int = 0
    attention_mask: Literal["causal", "set", "document_isolated"] = "causal"
    answer_spellings: tuple[tuple[str, ...], tuple[str, ...]] = ()
    prompt_format: str = "raw-v1"
    prompt_layout: str = "ai-if"
    chunk_cap_tokens: int = 0
    moe_backend: str | None = None
    vllm_reserve_bytes: float = 0.0
    layer_kinds: tuple[LayerKind, ...] = ()
    vision_tower: bool = False
    vision_mem_bytes: float = 0.0
    image_patch_pixels: int = 0
    image_merge: int = 0
    image_min_pixels: int = 0
    image_max_pixels: int = 0
    max_num_seqs: int = 0
    attn_output_gate: bool = False
    gdn_key_heads: int = 0
    gdn_value_heads: int = 0
    gdn_key_dim: int = 0
    gdn_value_dim: int = 0
    gdn_weight_precision: Precision | None = None
    mlp_weight_precision: Precision | None = None
    conv_width: int = 0
    attention_conv_channels: int = 0
    weight_layers: int = 0
    mlp_fp8_layers: int = 0
    extra_projection_params: int = 0
    checkpoint_layers: int = 0
    sae: SaeSpec | None = None

    def __post_init__(self):
        if self.checkpoint_layers and self.checkpoint_layers < self.layers:
            raise ValueError(
                f"{self.name}: checkpoint_layers ({self.checkpoint_layers}) "
                f"is below layers ({self.layers})")
        if (self.vision_tower and self.role != "decision"
                and not self.checkpoint_layers and not (
                    self.image_patch_pixels and self.image_merge
                    and self.image_max_pixels)):
            raise ValueError(
                f"{self.name}: a vision tower needs image_patch_pixels, "
                "image_merge, and image_max_pixels")
        if self.layer_kinds and len(self.layer_kinds) != self.layers:
            raise ValueError(
                f"{self.name}: layer_kinds has {len(self.layer_kinds)} "
                f"entries for {self.layers} layers")
        if self.sliding_window and not self.layer_kinds:
            raise ValueError(
                f"{self.name}: sliding_window needs layer_kinds to say which "
                f"layers slide")

    def _properties(self) -> dict:
        """The derived sizes, computed once by Rust's spec."""
        found = self.__dict__.get("_derived")
        if found is None:
            found = {key: _tuples(value) for key, value in json.loads(
                _quail_native.model_properties(json.dumps(asdict(self)))).items()}
            object.__setattr__(self, "_derived", found)
        return found

    def layer_kind(self, layer: int) -> LayerKind:
        """Return the kind of one layer."""
        return self.layer_kinds[layer] if self.layer_kinds else "full_attention"

    @property
    def bidirectional(self) -> bool:
        """Whether the full-attention layers read whole blocks, both ways.

        A "document_isolated" model's do; its Gated DeltaNet layers stay
        causal.
        """
        return self.attention_mask == "document_isolated"

    @property
    def reuses_kv(self) -> bool:
        """Whether a document's KV can be kept and read again by later questions."""
        return self._properties()["reuses_kv"]

    @property
    def attention_layers(self) -> tuple[int, ...]:
        """The layers that store KV."""
        return self._properties()["attention_layers"]

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        """The layers that keep a Gated DeltaNet state."""
        return self._properties()["gdn_layers"]

    @property
    def short_conv_layers(self) -> tuple[int, ...]:
        """The layers that keep a short convolution's input columns."""
        return self._properties()["short_conv_layers"]

    @property
    def state_layers(self) -> tuple[int, ...]:
        """The layers that keep state between a document's tokens."""
        return self._properties()["state_layers"]

    @property
    def conv_state_channels(self) -> int:
        """Channels of the convolution input a state layer keeps."""
        return self._properties()["conv_state_channels"]

    @property
    def conv_channels(self) -> int:
        """Channels of a Gated DeltaNet layer's convolution: q, k, and v joined."""
        return self._properties()["conv_channels"]

    @property
    def gdn_state_bytes(self) -> int:
        """Bytes of one layer's recurrent state, a matrix per value head."""
        return self._properties()["gdn_state_bytes"]

    @property
    def conv_state_bytes(self) -> int:
        """Bytes of one layer's convolution input columns kept between tokens."""
        return self._properties()["conv_state_bytes"]

    @property
    def checkpoint_bytes(self) -> int:
        """Bytes of the state a sequence leaves in every state layer."""
        return self._properties()["checkpoint_bytes"]

    @property
    def kv_shapes(self) -> tuple:
        """Per layer, the (KV heads, head dim) its KV stores."""
        return self._properties()["kv_shapes"]

    @property
    def widest_projection(self) -> int:
        """Output columns of the widest dense projection in any layer."""
        return self._properties()["widest_projection"]

    @property
    def kappa(self) -> float:
        """KV bytes per cached token, summed over the layers."""
        return self._properties()["kappa"]

    @property
    def kv_elements_per_token(self) -> int:
        """KV elements per token, dtype-free: 2 * sum of num_kv_heads * d_head."""
        return self._properties()["kv_elements_per_token"]

    @property
    def sliding_layer_set(self) -> frozenset:
        """The layers that keep only the last sliding_window tokens of KV."""
        if not self.sliding_window:
            return frozenset()
        return frozenset(i for i in range(self.layers)
                         if self.layer_kind(i) == "sliding_attention")

    @property
    def kappa_sliding(self) -> float:
        """KV bytes per token on the sliding layers alone."""
        return self._properties()["kappa_sliding"]

    @property
    def kappa_full(self) -> float:
        """KV bytes per token on the layers that keep every token."""
        return self._properties()["kappa_full"]

    @property
    def turn(self) -> tuple[str, str]:
        """The chat-turn text wrapped around every prompt."""
        return self.turn_prefix, self.turn_suffix

    @property
    def W_mem(self) -> float:  # noqa: N802
        """Weight bytes as loaded, before the full untied head is discarded."""
        return self._properties()["W_mem"]

    @property
    def head_mem_bytes(self) -> float:
        """Return untied output-head memory in bytes, or zero for tied weights."""
        return self._properties()["head_mem_bytes"]

    @property
    def W_resident(self) -> float:  # noqa: N802
        """Return model weight bytes resident on the GPU, including the output head."""
        return self._properties()["W_resident"]

    @property
    def act_per_token(self) -> float:
        """Reserved activation bytes per batched token."""
        return self._properties()["act_per_token"]

    @property
    def intermediate(self) -> int:
        """MLP intermediate width; gate_up packs two of them."""
        return self._properties()["intermediate"]

@dataclass(frozen=True)
class DeviceSpec:
    name: str
    mem_bytes: float
    hbm_bw: float
    fp8_flops: float
    bf16_flops: float
    fp32_flops: float
    usd_per_hour: float = 0.0
    price_source: str = ""
    cuda_name: str = ""
    modal_gpu: str = ""
    page_tokens: int = 16
    mamf_fp8_flops: float = 0.0
    mamf_bf16_flops: float = 0.0
    fp4_flops: float = 0.0

    def __post_init__(self):
        for precision in get_args(Precision):
            if precision != "fp4" and not self.arithmetic_bandwidth(precision) > 0:
                raise ValueError(
                    f"{self.name}: {precision}_flops must be positive")

    def arithmetic_bandwidth(self, precision: Precision) -> float:
        """Return arithmetic throughput for one component precision."""
        if precision == "fp4":
            return self.fp4_flops
        if precision == "fp8":
            return self.fp8_flops
        if precision == "bf16":
            return self.bf16_flops
        if precision == "fp32":
            return self.fp32_flops
        raise ValueError(f"unsupported precision: {precision}")
