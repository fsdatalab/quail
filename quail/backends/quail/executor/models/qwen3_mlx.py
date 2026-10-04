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

from quail.backends.quail.executor.mlx_device.kernels import (
    check_geometry,
    paged_attention,
)
from quail.backends.quail.executor.models.base import ModelPipeline


class MlxQwen3Pipeline(ModelPipeline):
    """Forward passes for Qwen3 weights held as MLX arrays.

    Args:
        model: The model's Qwen3Weights.
        arena: The KVArena, whose pools are MlxKVPools.
        spec: The model's ModelSpec; unused, the weights carry their shape.
    """

    tree_attention = False
    needs_pages = True
    gemm_warmup = False

    def __init__(self, model, arena, *, spec=None):
        import mlx.core as mx

        weights, pools = model, arena.pools
        config = weights.config
        expected = [(config.n_kv, config.head_dim)] * config.layers
        if arena.shapes != expected or arena.has_sliding:
            raise ValueError("the KV arena does not match the model's KV shape")
        if pools.dtype != weights.dtype:
            raise ValueError("the KV pools and the weights differ in dtype")
        check_geometry(config.n_q, config.n_kv, config.head_dim,
                       arena.page_tokens, weights.dtype)
        self.mx = mx
        self.weights = weights
        self.pools = pools
        self.scale = config.head_dim ** -0.5
        # MLX shapes are 32-bit, so a chunk needs rows x widest_row < 2^31.
        widest = max((config.n_q + 2 * config.n_kv) * config.head_dim,
                     2 * config.intermediate)
        self.max_chunk_tokens = (2**31 - 1) // widest

    def _array(self, values, dtype):
        """Return a staged array, or a host one, as an MLX array of a dtype."""
        if isinstance(values, self.mx.array):
            return values.astype(dtype)
        return self.mx.array(values, dtype=dtype)

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
        tail = unified["tail_src"] is not None
        return dict(
            src=(self._array(unified["src"], mx.int32)
                 if len(unified["src"]) else None),
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
