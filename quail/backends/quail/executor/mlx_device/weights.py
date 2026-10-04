"""Qwen3 weights as MLX arrays."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3Config:
    """The shape of a Qwen3 checkpoint, read from its config.json.

    Attributes:
        hidden: Hidden size.
        layers: Number of layers.
        n_q: Query heads.
        n_kv: KV heads.
        head_dim: Head dimension.
        intermediate: MLP intermediate width.
        rms_eps: RMSNorm epsilon.
        rope_theta: Rotary base.
    """

    hidden: int
    layers: int
    n_q: int
    n_kv: int
    head_dim: int
    intermediate: int
    rms_eps: float
    rope_theta: float

    @classmethod
    def from_dict(cls, config: dict) -> "Qwen3Config":
        """Build the config from a parsed Hugging Face config.json."""
        rope = config.get("rope_parameters") or {}
        heads = config["num_attention_heads"]
        return cls(
            hidden=config["hidden_size"],
            layers=config["num_hidden_layers"],
            n_q=heads,
            n_kv=config["num_key_value_heads"],
            head_dim=config.get("head_dim") or config["hidden_size"] // heads,
            intermediate=config["intermediate_size"],
            rms_eps=config["rms_norm_eps"],
            rope_theta=float(rope.get("rope_theta", config.get("rope_theta"))),
        )


@dataclass
class Qwen3Layer:
    """One layer's weights, with Q, K, V and gate, up each merged.

    Attributes:
        input_norm: Input RMSNorm weight, (hidden,).
        qkv: Q, K, and V projections stacked in that order,
            ((n_q + 2 n_kv) * head_dim, hidden).
        q_norm: Per-head Q RMSNorm weight, (head_dim,).
        k_norm: Per-head K RMSNorm weight, (head_dim,).
        o_proj: Output projection, (hidden, n_q * head_dim).
        post_norm: Post-attention RMSNorm weight, (hidden,).
        gate_up: Gate and up projections stacked in that order,
            (2 * intermediate, hidden).
        down: Down projection, (hidden, intermediate).
    """

    input_norm: object
    qkv: object
    q_norm: object
    k_norm: object
    o_proj: object
    post_norm: object
    gate_up: object
    down: object


@dataclass
class Qwen3Weights:
    """A Qwen3 model's weights as MLX arrays.

    Attributes:
        config: The checkpoint's shape.
        embed: Token embedding, (vocab, hidden).
        layers: One Qwen3Layer per layer.
        norm: Final RMSNorm weight, (hidden,).
    """

    config: Qwen3Config
    embed: object
    layers: list
    norm: object

    @classmethod
    def from_tensors(cls, tensors, config: Qwen3Config, dtype,
                     prefix: str = "") -> "Qwen3Weights":
        """Build the weights from a checkpoint's tensors by name.

        Args:
            tensors: Mapping from Hugging Face tensor name to MLX array.
            config: The checkpoint's shape.
            dtype: MLX dtype every weight is cast to.
            prefix: Text before every name, such as "model.".
        """
        import mlx.core as mx

        def get(name):
            return tensors[prefix + name].astype(dtype)

        layers = []
        for index in range(config.layers):
            at = f"layers.{index}."
            attn = at + "self_attn."
            layers.append(Qwen3Layer(
                input_norm=get(at + "input_layernorm.weight"),
                qkv=mx.concatenate([get(attn + "q_proj.weight"),
                                    get(attn + "k_proj.weight"),
                                    get(attn + "v_proj.weight")]),
                q_norm=get(attn + "q_norm.weight"),
                k_norm=get(attn + "k_norm.weight"),
                o_proj=get(attn + "o_proj.weight"),
                post_norm=get(at + "post_attention_layernorm.weight"),
                gate_up=mx.concatenate([get(at + "mlp.gate_proj.weight"),
                                        get(at + "mlp.up_proj.weight")]),
                down=get(at + "mlp.down_proj.weight"),
            ))
        weights = cls(config=config, embed=get("embed_tokens.weight"),
                      layers=layers, norm=get("norm.weight"))
        mx.eval(weights.arrays())
        return weights

    def arrays(self) -> list:
        """Return every weight array."""
        out = [self.embed, self.norm]
        for layer in self.layers:
            out.extend(vars(layer).values())
        return out

    @property
    def dtype(self):
        return self.embed.dtype

    @property
    def nbytes(self) -> int:
        """Bytes the weights hold."""
        return sum(array.nbytes for array in self.arrays())
