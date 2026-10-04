"""Qwen3 forward pass over packed rows with MLX.

Layer order: input RMSNorm, fused QKV projection, per-head Q and K
RMSNorm with rotary, paged attention, output projection, post-attention
RMSNorm, gate_up projection, SiLU-gated product, down projection. The
Decision 2.0 backbone is this architecture.

A chunk's rows are packed back to back, each with its own rotary
position. Every chunk carries arena pages: the fresh rows' K and V are
written into the pools, then one causal paged attention call reads the
retained and the fresh KV together.
"""

from dataclasses import dataclass

import numpy as np

from quail.backends.quail.executor.mlx_device.kernels import (
    check_geometry,
    paged_attention,
)
from quail.backends.quail.executor.models.base import ModelPipeline


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


class MlxQwen3Pipeline(ModelPipeline):
    """Forward passes for Qwen3 weights held as MLX arrays.

    Args:
        weights: The model's Qwen3Weights.
        pools: The MlxKVPools the chunks' pages index into.
    """

    tree_attention = False
    needs_pages = True
    gemm_warmup = False

    def __init__(self, weights: Qwen3Weights, pools):
        import mlx.core as mx

        config = weights.config
        if (pools.n_layers, pools.n_kv, pools.d_head) != (
                config.layers, config.n_kv, config.head_dim):
            raise ValueError("the KV pools do not match the model's KV shape")
        if pools.dtype != weights.dtype:
            raise ValueError("the KV pools and the weights differ in dtype")
        check_geometry(config.n_q, config.n_kv, config.head_dim,
                       pools.page_tokens, weights.dtype)
        self.mx = mx
        self.weights = weights
        self.pools = pools
        self.scale = config.head_dim ** -0.5
        # MLX shapes are 32-bit, so a chunk needs rows x widest_row < 2^31.
        widest = max((config.n_q + 2 * config.n_kv) * config.head_dim,
                     2 * config.intermediate)
        self.max_chunk_tokens = (2**31 - 1) // widest

    def _array(self, values, dtype):
        return self.mx.array(np.asarray(values), dtype=dtype)

    def _reads(self, chunk) -> dict:
        """Return the chunk's KV writes and paged read as MLX arrays.

        Raises:
            ValueError: The chunk was packed for a path this forward
                pass does not run.
        """
        mx = self.mx
        meta = chunk.meta
        unified = meta["unified"]
        if chunk.attention_mode != "unified":
            raise ValueError("the MLX forward pass runs unified attention only")
        if unified is None:
            raise ValueError("the MLX forward pass reads paged KV only; "
                             "pack the chunk with arena pages")
        if meta.get("canvas") is not None or "sliding" in unified:
            raise ValueError("the MLX forward pass has no canvas rows and "
                             "no sliding layers")
        src = np.asarray(unified["src"])
        tail = unified["tail_src"] is not None
        return dict(
            src=mx.array(src, dtype=mx.int32) if src.size else None,
            dst=self._array(unified["dst"], mx.int64),
            tail_src=self._array(unified["tail_src"], mx.int32) if tail else None,
            tail_dst=self._array(unified["tail_dst"], mx.int64) if tail else None,
            table=self._array(unified["table"], mx.int32),
            used=self._array(unified["used"], mx.int32),
            cu_q=self._array(unified["cu_q"], mx.int32),
            max_used=int(unified["max_used"]),
        )

    def _rope(self, x, positions):
        """Rotate (rows, heads, head dim) by each row's own position."""
        n, heads, dim = x.shape
        # mx.fast.rope takes one offset per batch element, so every row
        # is a batch element of one token
        rotated = self.mx.fast.rope(
            x.reshape(n, heads, 1, dim), dim, traditional=False,
            base=self.weights.config.rope_theta, scale=1.0, offset=positions)
        return rotated.reshape(n, heads, dim)

    def _attention(self, layer: int, q, k, v, reads):
        """Write the fresh K and V to their slots, then read paged KV."""
        mx = self.mx
        pools = self.pools
        if reads["src"] is not None:
            pools.write(layer, k[reads["src"]], v[reads["src"]], reads["dst"])
        if reads["tail_src"] is not None:
            pools.copy(layer, reads["tail_src"], reads["tail_dst"])
        k_pool, v_pool = pools.paged_kv(layer)
        return paged_attention(
            mx.contiguous(q), k_pool, v_pool, table=reads["table"],
            used=reads["used"], cu_q=reads["cu_q"],
            max_used=reads["max_used"], scale=self.scale)

    def forward_chunk(self, chunk):
        """Return the final-normed hidden state of chunk.final_indices.

        The answers and the pools are handed to MLX for evaluation
        before this returns, so the chunk's reads are ordered before
        any later write to the pages it read.
        """
        mx = self.mx
        config = self.weights.config
        eps = config.rms_eps
        n_q, n_kv, dim = config.n_q, config.n_kv, config.head_dim
        q_width, k_width = n_q * dim, n_kv * dim
        reads = self._reads(chunk)
        positions = self._array(chunk.positions, mx.int32)
        hidden = self.weights.embed[self._array(chunk.input_ids, mx.int32)]
        n = hidden.shape[0]
        for index, layer in enumerate(self.weights.layers):
            x = mx.fast.rms_norm(hidden, layer.input_norm, eps)
            qkv = x @ layer.qkv.T
            q = mx.fast.rms_norm(
                qkv[:, :q_width].reshape(n, n_q, dim), layer.q_norm, eps)
            k = mx.fast.rms_norm(
                qkv[:, q_width:q_width + k_width].reshape(n, n_kv, dim),
                layer.k_norm, eps)
            v = mx.contiguous(qkv[:, q_width + k_width:]).reshape(n, n_kv, dim)
            attended = self._attention(
                index, self._rope(q, positions), self._rope(k, positions), v,
                reads)
            hidden = hidden + attended.reshape(n, q_width) @ layer.o_proj.T
            x = mx.fast.rms_norm(hidden, layer.post_norm, eps)
            gate_up = x @ layer.gate_up.T
            gate = gate_up[:, :config.intermediate]
            up = gate_up[:, config.intermediate:]
            hidden = hidden + (gate * mx.sigmoid(gate) * up) @ layer.down.T
        final = self._array(chunk.final_indices, mx.int32)
        normed = mx.fast.rms_norm(hidden[final], self.weights.norm, eps)
        mx.async_eval(normed, *self.pools.arrays())
        return normed
