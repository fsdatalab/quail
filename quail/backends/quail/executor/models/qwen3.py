"""Qwen3 forward pass over vLLM's loaded Qwen3 module.

Layer order: input RMSNorm, fused QKV projection, per-head Q and K
RMSNorm with rotary, attention, output projection, post-attention
RMSNorm, gate_up projection, SiLU-gated product, down projection.
The Qwen3 rerankers and the Decision 2.0 backbone are this
architecture with bf16 weights and run through this file too.
"""

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.cuda_graphs import DecodeGraphs
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.cost.budgets import tree_attention_allowed


class Qwen3Pipeline(ModelPipeline):
    """Forward passes for Qwen3 checkpoints loaded by vLLM."""

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine):
        import torch

        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.norm
        attn = self.layers[0].self_attn
        fp8 = attn.qkv_proj.weight.dtype == torch.float8_e4m3fn
        self.engine = engine_class(
            arena, num_q_heads=attn.num_heads, num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=fp8, kernels=kernels)
        # The fused kernels compute element offsets in 32-bit ints,
        # so a chunk needs rows x widest_row < 2^31.
        widest = max(max(layer.self_attn.qkv_proj.weight.shape[0],
                         layer.mlp.gate_up_proj.weight.shape[0])
                     for layer in self.layers)
        self.max_chunk_tokens = (2**31 - 1) // widest
        self.tree_attention = spec is None or tree_attention_allowed(spec)
        self.decode_graphs = DecodeGraphs(self)

    def linears(self):
        # every layer has the same four shapes
        layer = self.layers[0]
        return (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                layer.mlp.gate_up_proj, layer.mlp.down_proj)

    def forward_chunk(self, chunk):
        chunk.meta["layer"] = 0
        hidden = self.embed(chunk.input_ids)
        residual = None
        for layer in self.layers:
            qkv = self._before_attention(layer, hidden, residual)
            if residual is None:
                residual = hidden
            hidden = self._attention(layer, qkv, chunk)
            hidden = self._after_attention(layer, hidden, residual)
        return self._final_norm(hidden, residual, chunk.final_indices)

    def _before_attention(self, layer, hidden, residual):
        """Return the layer's QKV projection; a given residual takes hidden first."""
        engine = self.engine
        if residual is None:
            q_in, q_scale = engine.norm_quant(hidden, layer.input_layernorm)
        else:
            q_in, q_scale = engine.norm_quant(hidden, layer.input_layernorm, residual)
        return engine.gemm(q_in, q_scale, layer.self_attn.qkv_proj)

    def _attention(self, layer, qkv, chunk):
        """Return the layer's attention output projection over the chunk's KV."""
        engine = self.engine
        attn = layer.self_attn
        q, k = engine.qk_norm_rope(qkv, chunk.positions, attn)
        v = qkv[:, (engine.num_q_heads + engine.num_kv_heads) * engine.head_dim:]
        o_in, o_scale = engine.attention(q, k, v, chunk)
        return engine.gemm(o_in, o_scale, attn.o_proj)

    def _after_attention(self, layer, hidden, residual):
        """Return the layer's MLP output; the residual takes hidden first."""
        engine = self.engine
        g_in, g_scale = engine.norm_quant(
            hidden, layer.post_attention_layernorm, residual)
        gate_up = engine.gemm(g_in, g_scale, layer.mlp.gate_up_proj)
        d_in, d_scale = engine.activation_quant(gate_up)
        return engine.gemm(d_in, d_scale, layer.mlp.down_proj)

    def _final_norm(self, hidden, residual, final):
        normed, _ = self.engine.fused_add_rms_norm(
            hidden.index_select(0, final), residual.index_select(0, final),
            self.final_norm)
        return normed

    # ---- the pieces between attention calls, for PieceGraphs ----------------

    @property
    def num_pieces(self) -> int:
        return len(self.layers) + 1

    def piece_widths(self) -> tuple:
        """Return the hidden width and the QKV projection's width."""
        layer = self.layers[0]
        return (self.embed.weight.shape[1],
                layer.self_attn.qkv_proj.weight.shape[0])

    def piece(self, index, ids, hidden, residual, qkv, out):
        """Run the work between attention calls index - 1 and index.

        Piece 0 embeds the ids and projects layer 0's QKV; piece i
        finishes layer i - 1 from its attention output in hidden and
        projects layer i's QKV; the last piece finishes the last layer
        into out. Every argument is a buffer of the same rows; residual
        is updated in place.
        """
        layers = self.layers
        if index == 0:
            embedded = self.embed(ids)
            residual.copy_(embedded)
            qkv.copy_(self._before_attention(layers[0], embedded, None))
            return
        after = self._after_attention(layers[index - 1], hidden, residual)
        if index < len(layers):
            qkv.copy_(self._before_attention(layers[index], after, residual))
        else:
            out.copy_(after)

    def attention_piece(self, index, qkv, chunk):
        """Return layer index's attention output projection, run eagerly."""
        return self._attention(self.layers[index], qkv, chunk)

    def final_piece(self, hidden, residual, final):
        """Return the final norm of the rows final selects."""
        return self._final_norm(hidden, residual, final)
