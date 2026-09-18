"""DiffusionGemma forward pass over vLLM's loaded module.

One pass per document: the prompt rows run in the model's encoder
mode (causal attention, KV written to the arena) and the canvas rows
that follow each answer cue run in its decoder mode (bidirectional
over the canvas, reading the whole prompt, KV never kept). The TRUE
and FALSE logits at the first canvas row are the answer, which is
what the model's own sampler commits on its first denoising step
when it is confident. There is no further denoising.

Layer order, from vLLM's Gemma4DecoderLayer: input RMSNorm, fused
QKV projection, per-head Q and K RMSNorm with rotary, unweighted V
RMSNorm, attention at softmax scale 1.0 (sliding window on the
sliding layers), output projection, post-attention RMSNorm, residual
add; pre-feedforward RMSNorm, the dense MLP and the routed experts
side by side with their own norms, post-feedforward RMSNorm,
residual add, per-layer scalar. Embeddings are scaled by
sqrt(hidden). Each canvas row's embedding passes through the
self-conditioning post-norm with a zero conditioning signal, as on
the sampler's first step.

The norms, projections, rotary modules, and expert kernels are the
loaded vLLM modules, called as they are: the checkpoint's fp8 weights
carry per-channel scales with per-token activation quantization,
which vLLM's own linear path handles. Only attention and packing are
Quail's. The engine's fused Qwen3 kernels are not used, so the
pipeline reports is_fp8 False and every chunk runs the unified
attention path.
"""

import numpy as np

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.models.base import ModelPipeline

# The canvas starts as random token ids, as vLLM's sampler starts it;
# one fixed draw serves every document so answers are reproducible.
CANVAS_SEED = 0


def canvas_token_ids(vocab: int, tokens: int, seed: int = CANVAS_SEED) -> tuple:
    """The fixed random token ids that fill every canvas."""
    rng = np.random.default_rng(seed)
    return tuple(int(i) for i in rng.integers(0, vocab, tokens))


def _init_moe_workspace():
    """Give vLLM's fused MoE kernels the scratch buffers its worker would.

    The buffers grow to the largest chunk on first use and stay
    allocated; nothing locks them.
    """
    import torch
    from vllm.v1.worker import workspace

    if not workspace.is_workspace_manager_initialized():
        workspace.init_workspace_manager(torch.device("cuda"))


class DiffusionGemmaPipeline(ModelPipeline):
    """Forward passes for DiffusionGemma checkpoints loaded by vLLM.

    The full-attention layers' 512-wide heads run vLLM's Triton paged
    attention kernel, so every chunk carries arena pages.

    Args:
        model: vLLM's DiffusionGemmaForConditionalGeneration module.
        arena: KVArena with one pool per layer at that layer's KV
            geometry (spec.kv_shapes).
        spec: The ModelSpec, for the vocabulary size and canvas length.
        kernels: Engine kernel set; only its KV scatter kernel runs here.
        engine_class: Engine class, replaceable by experiments.
    """

    needs_pages = True

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine):
        backbone = model.model
        self.layers = backbone.layers
        self.embed = backbone.embed_tokens
        self.normalizer = backbone.normalizer
        self.final_norm = backbone.norm
        self.canvas_norm = model.self_conditioning.post_norm
        self.vllm_config = model.quail_vllm_config
        self.window = backbone.config.sliding_window
        attn = self.layers[0].self_attn
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=False, kernels=kernels)
        self.canvas_ids = canvas_token_ids(spec.vocab, spec.canvas_tokens)
        _init_moe_workspace()
        # The engine's KV scatter kernel indexes rows in 32-bit ints.
        # vLLM keeps these fp8 weights transposed, so take the wider
        # side.
        widest = max(max(layer.self_attn.qkv_proj.weight.shape)
                     for layer in self.layers)
        self.max_chunk_tokens = (2**31 - 1) // widest

    def linears(self):
        layer = self.layers[0]
        return (layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                layer.mlp.gate_up_proj, layer.mlp.down_proj)

    def forward_chunk(self, chunk):
        from vllm.forward_context import set_forward_context

        meta = chunk.meta
        meta["layer"] = 0
        positions = chunk.positions
        n = chunk.input_ids.shape[0]
        hidden = self.embed(chunk.input_ids) * self.normalizer
        canvas = meta.get("canvas")
        if canvas is not None:
            rows = canvas["rows"]
            hidden.index_copy_(
                0, rows, self.canvas_norm(hidden.index_select(0, rows)))
        # the fused MoE kernels look their layer up in the forward
        # context
        with set_forward_context(None, self.vllm_config, num_tokens=n):
            for layer in self.layers:
                hidden = self._layer(layer, hidden, positions, meta)
        return self.final_norm(hidden.index_select(0, chunk.final_indices))

    def _attention(self, attn, hidden, positions, meta):
        n = hidden.shape[0]
        H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = attn.q_norm(q.unflatten(-1, (H, D))).flatten(-2, -1)
        k = attn.k_norm(k.unflatten(-1, (KH, D))).flatten(-2, -1)
        q, k = attn.rotary_emb(positions, q, k)
        v = attn.v_norm(v.unflatten(-1, (KH, D)))
        out = self.engine.attention_unified(
            q.reshape(n, H, D), k.reshape(n, KH, D).contiguous(),
            v.contiguous(), meta, softmax_scale=1.0,
            window=self.window if attn.is_sliding else None)
        out, _ = attn.o_proj(out)
        return out

    def _layer(self, layer, hidden, positions, meta):
        residual = hidden
        hidden = layer.input_layernorm(residual)
        hidden = self._attention(layer.self_attn, hidden, positions, meta)
        hidden = layer.post_attention_layernorm(hidden)
        hidden = hidden + residual
        residual = hidden
        hidden = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(hidden)
        if layer.enable_moe_block:
            dense = layer.post_feedforward_layernorm_1(hidden)
            routed = layer.pre_feedforward_layernorm_2(residual)
            routed = layer.moe(routed, layer.router(residual))
            routed = layer.post_feedforward_layernorm_2(routed)
            hidden = dense + routed
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = hidden + residual
        return hidden * layer.layer_scalar
