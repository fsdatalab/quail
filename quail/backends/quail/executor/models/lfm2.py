"""LFM2 forward pass over vLLM's loaded LFM2 text model.

Each layer is an operator RMSNorm, a token mixer, an FFN RMSNorm, and a
gated MLP. The mixer is a short convolution in 22 layers and grouped
query attention in 8. A short convolution projects to B, C, and x in
one GEMM, convolves B times x with a depthwise causal filter of three
taps, multiplies by C, and projects out. It keeps the last two columns
of its convolution input as the layer's state, which the chunk's
checkpoint phases (executor/gdn.py) save and restore. An attention
mixer norms and rotates q and k per head and runs paged attention.
"""

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.gdn import short_conv_core
from quail.backends.quail.executor.models.base import ModelPipeline


class Lfm2Pipeline(ModelPipeline):
    """Forward passes for LFM2 checkpoints loaded by vLLM."""

    needs_pages = True
    tree_attention = False
    warm_row_classes = False
    warm_joins = False

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine, use_gdn_kernels=True):
        text = getattr(model, "language_model", model)
        self.spec = spec
        self.arena = arena
        self.use_gdn_kernels = use_gdn_kernels
        self.vllm_config = getattr(model, "quail_vllm_config", None)
        self.layers = text.model.layers
        self.embed = text.model.embed_tokens
        self.final_norm = text.model.embedding_norm
        self.attention_layer = next(
            layer for layer in self.layers if hasattr(layer, "self_attn"))
        self.conv_layer = next(
            layer for layer in self.layers if hasattr(layer, "short_conv"))
        attn = self.attention_layer.self_attn
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=False, kernels=kernels)
        widest = max(
            self.conv_layer.short_conv.in_proj.weight.shape[0],
            attn.qkv_proj.weight.shape[0],
            self.conv_layer.feed_forward.w13.weight.shape[0])
        self.max_chunk_tokens = (2**31 - 1) // widest

    def linears(self):
        conv = self.conv_layer.short_conv
        attn = self.attention_layer.self_attn
        mlp = self.attention_layer.feed_forward
        return (conv.in_proj, conv.out_proj, attn.qkv_proj, attn.out_proj,
                mlp.w13, mlp.w2)

    def _linear(self, x, module):
        x_q, x_s = self.engine.quant(x)
        return self.engine.gemm(x_q, x_s, module)

    def _short_conv(self, mixer, x, phases, slot):
        projected = self._linear(x, mixer.in_proj)
        b, c, gated = projected.chunk(3, dim=-1)
        weight = mixer.conv.weight
        taps = weight.view(weight.shape[0], weight.shape[2])
        conv = short_conv_core(
            (b * gated).contiguous(), taps, phases, self.arena, slot,
            use_kernels=self.use_gdn_kernels)
        return self._linear(c * conv, mixer.out_proj)

    def _attention(self, attn, x, positions, chunk):
        qkv = self._linear(x, attn.qkv_proj)
        rows = qkv.shape[0]
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = attn.q_layernorm(
            q.reshape(rows, attn.num_heads, attn.head_dim).contiguous())
        k = attn.k_layernorm(
            k.reshape(rows, attn.num_kv_heads, attn.head_dim).contiguous())
        q, k = attn.rotary_emb(positions, q.reshape(rows, -1),
                               k.reshape(rows, -1))
        out = self.engine.attention_unified(
            q.view(rows, attn.num_heads, attn.head_dim),
            k.view(rows, attn.num_kv_heads, attn.head_dim),
            v.contiguous().view(rows, attn.num_kv_heads, attn.head_dim),
            chunk.meta)
        return self._linear(out, attn.out_proj)

    def _mlp(self, mlp, x):
        gate_up = self._linear(x, mlp.w13)
        d_in, d_scale = self.engine.activation_quant(gate_up)
        return self.engine.gemm(d_in, d_scale, mlp.w2)

    def forward_chunk(self, chunk):
        from contextlib import nullcontext

        from vllm.config import set_current_vllm_config

        context = (nullcontext() if self.vllm_config is None
                   else set_current_vllm_config(self.vllm_config))
        with context:
            return self._forward(chunk)

    def _forward(self, chunk):
        meta = chunk.meta
        meta["layer"] = 0
        phases = meta["gdn"]
        if phases is None:
            raise ValueError(
                "an LFM2 chunk needs its convolution phases; the "
                "arena has no checkpoint slots")
        hidden = self.embed(chunk.input_ids)
        residual = None
        slot = 0
        for layer in self.layers:
            if residual is None:
                residual = hidden
                x = layer.operator_norm(hidden)
            else:
                x, residual = layer.operator_norm(hidden, residual)
            if hasattr(layer, "short_conv"):
                x = self._short_conv(layer.short_conv, x, phases, slot)
                slot += 1
                # arena layers are indexed by model layer
                meta["layer"] += 1
            else:
                x = self._attention(layer.self_attn, x, chunk.positions, chunk)
            x, residual = layer.ffn_norm(x, residual)
            hidden = self._mlp(layer.feed_forward, x)
        final = chunk.final_indices
        normed, _ = self.final_norm(hidden.index_select(0, final),
                                    residual.index_select(0, final))
        return normed
