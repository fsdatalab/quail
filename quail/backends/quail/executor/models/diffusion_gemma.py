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

The projections and expert kernels are the loaded vLLM modules,
called as they are: the checkpoint's fp8 weights carry per-channel
scales with per-token activation quantization, which vLLM's own
linear path handles. The norms and rotary go straight to vLLM's CUDA
kernels through the engine, since the modules' own dispatch runs the
unfused PyTorch path on this build, ten times slower. Attention and
packing are Quail's. The engine's fused Qwen3 kernels are not used,
so the pipeline reports is_fp8 False and every chunk runs the unified
attention path.
"""

import numpy as np

from quail.backends.quail.executor.attention import WIDE_HEAD_KERNELS, Engine
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
        wide_head_kernel: Attention kernel for the 512-wide heads,
            "triton" (vLLM's unified attention) or "fa4".
    """

    needs_pages = True

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine, wide_head_kernel="triton", fused=True):
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
        if wide_head_kernel not in WIDE_HEAD_KERNELS:
            raise ValueError(
                f"wide_head_kernel must be one of {WIDE_HEAD_KERNELS}, "
                f"got {wide_head_kernel!r}")
        self.engine.wide_head_kernel = wide_head_kernel
        self._ones = {}
        for layer in self.layers:
            rope = layer.self_attn.rotary_emb
            # the rotary kernel reads the cache at the activations' dtype
            rope.cos_sin_cache = rope.cos_sin_cache.to(self.engine.torch.bfloat16)
            router = layer.router
            # the router's constant scale and learned per-dimension scale
            # fold into one vector
            router.quail_scale = (router.root_size.to(router.scale.dtype)
                                  * router.scale).detach()
        self.fused = fused
        if fused:
            self._fold_layer_scalars(model)
        self.canvas_ids = canvas_token_ids(spec.vocab, spec.canvas_tokens)
        # an empty canvas reads the answer at the prompt's last row
        if spec.canvas_tokens and not (
                0 <= spec.canvas_answer_row < spec.canvas_tokens):
            raise ValueError(
                f"canvas_answer_row {spec.canvas_answer_row} is outside "
                f"the {spec.canvas_tokens}-row canvas")
        self.canvas_answer_row = (spec.canvas_answer_row
                                  if spec.canvas_tokens else 0)
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

    def _norm(self, x, module):
        """One of the model's RMS norms, through the engine's kernel."""
        weight = module.weight if module.has_weight else self._ones_like(module, x)
        return self.engine.norm_rows(x, weight, module.variance_epsilon)

    def _ones_like(self, module, x):
        key = (module.hidden_size, x.dtype)
        if key not in self._ones:
            self._ones[key] = self.engine.torch.ones(
                module.hidden_size, dtype=x.dtype, device=x.device)
        return self._ones[key]

    def _fold_layer_scalars(self, model):
        """Fold each layer's output scalar into its post-feedforward norm.

        Layer l multiplies its output, the norm of its feedforward
        branch plus the residual, by s_l. Scaling that norm's weight by
        s_l leaves only the residual to scale, which the fused path does
        in place. The scalars are far below one (0.07 to 0.5), so a
        cumulative fold across layers would overflow the weights.
        """
        if getattr(model, "quail_scalars_folded", False):
            return
        for layer in self.layers:
            layer.post_feedforward_layernorm.weight.data.mul_(
                float(layer.layer_scalar))
        model.quail_scalars_folded = True

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
                0, rows, self._norm(hidden.index_select(0, rows), self.canvas_norm))
        # the fused MoE kernels look their layer up in the forward
        # context
        with set_forward_context(None, self.vllm_config, num_tokens=n):
            if self.fused:
                hidden = self._layers_fused(hidden, positions, meta)
            else:
                for layer in self.layers:
                    hidden = self._layer(layer, hidden, positions, meta)
        return self._norm(hidden.index_select(0, chunk.final_indices),
                          self.final_norm)

    # ---- the fused path ---------------------------------------------
    # The same arithmetic as _layer with the residual adds, the norms
    # that feed a linear, and that linear's input quantization fused,
    # and the per-head q and k norms fused with the rotary. The layer
    # scalars are folded into the post-feedforward norm weights at
    # build time.

    def _norm_quant(self, x, norm, residual=None):
        return self.engine.norm_quant_rows(x, norm.weight,
                                           norm.variance_epsilon, residual)

    def _layers_fused(self, hidden, positions, meta):
        engine = self.engine
        residual = hidden
        x_q, x_s = self._norm_quant(residual, self.layers[0].input_layernorm)
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            qkv = engine.fp8_linear(attn.qkv_proj, x_q, x_s)
            out = self._attention_fused(attn, qkv, positions, meta)
            out, _ = attn.o_proj(out)
            out = self._norm(out, layer.post_attention_layernorm)
            # residual becomes the attention sum; the feedforward reads
            # its norm
            x_q, x_s = self._norm_quant(out, layer.pre_feedforward_layernorm,
                                        residual)
            mlp = layer.mlp
            gate_up = engine.fp8_linear(mlp.gate_up_proj, x_q, x_s)
            dense, _ = mlp.down_proj(mlp.act_fn(gate_up))
            if layer.enable_moe_block:
                dense = self._norm(dense, layer.post_feedforward_layernorm_1)
                routed_in = self._norm(residual,
                                       layer.pre_feedforward_layernorm_2)
                router = layer.router
                logits, _ = router.proj(engine.norm_rows(
                    residual, router.quail_scale, router.norm.variance_epsilon))
                routed = layer.moe(routed_in, logits)
                routed = self._norm(routed, layer.post_feedforward_layernorm_2)
                # dense becomes the norm of the two branches' sum
                engine.fused_add_rms_norm(dense, routed,
                                          layer.post_feedforward_layernorm)
            else:
                dense = self._norm(dense, layer.post_feedforward_layernorm)
            # dense carries the layer scalar through its norm weight;
            # the residual takes it here
            residual.mul_(layer.layer_scalar)
            if index == last:
                return dense + residual
            # residual becomes the layer's output; the next layer reads
            # its norm
            x_q, x_s = self._norm_quant(
                dense, self.layers[index + 1].input_layernorm, residual)
        raise AssertionError("a model needs at least one layer")

    def _attention_fused(self, attn, qkv, positions, meta):
        n = qkv.shape[0]
        H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        rope = attn.rotary_emb
        q, k = self.engine.qk_norm_rope_heads(
            qkv, positions, n_q=H, n_kv=KH, head_dim=D,
            q_weight=attn.q_norm.weight, k_weight=attn.k_norm.weight,
            eps=attn.q_norm.variance_epsilon, cos_sin_cache=rope.cos_sin_cache)
        v = self._norm(qkv[:, (H + KH) * D:].reshape(n, KH, D), attn.v_norm)
        return self.engine.attention_unified(
            q.view(n, H, D), k.view(n, KH, D), v, meta, softmax_scale=1.0,
            window=self.window if attn.is_sliding else None)

    # ---- the reference path (fused=False) ---------------------------

    def _attention(self, attn, hidden, positions, meta):
        n = hidden.shape[0]
        H, KH, D = attn.num_heads, attn.num_kv_heads, attn.head_dim
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q = self._norm(q.reshape(n, H, D), attn.q_norm).view(n, H * D)
        k = self._norm(k.reshape(n, KH, D), attn.k_norm).view(n, KH * D)
        rope = attn.rotary_emb
        q, k = self.engine.rope_inplace(positions, q, k, D, rope.cos_sin_cache,
                                        rope.is_neox_style)
        v = self._norm(v.reshape(n, KH, D), attn.v_norm)
        out = self.engine.attention_unified(
            q.view(n, H, D), k.view(n, KH, D), v, meta, softmax_scale=1.0,
            window=self.window if attn.is_sliding else None)
        out, _ = attn.o_proj(out)
        return out

    def _router_logits(self, router, x):
        """The router's logits: unweighted norm, one scale, projection."""
        scaled = self._norm(x, router.norm) * router.quail_scale
        logits, _ = router.proj(scaled)
        return logits

    def _layer(self, layer, hidden, positions, meta):
        residual = hidden
        hidden = self._norm(residual, layer.input_layernorm)
        hidden = self._attention(layer.self_attn, hidden, positions, meta)
        hidden = self._norm(hidden, layer.post_attention_layernorm)
        hidden = hidden + residual
        residual = hidden
        hidden = self._norm(hidden, layer.pre_feedforward_layernorm)
        hidden = layer.mlp(hidden)
        if layer.enable_moe_block:
            dense = self._norm(hidden, layer.post_feedforward_layernorm_1)
            routed = self._norm(residual, layer.pre_feedforward_layernorm_2)
            routed = layer.moe(routed, self._router_logits(layer.router, residual))
            routed = self._norm(routed, layer.post_feedforward_layernorm_2)
            hidden = dense + routed
        hidden = self._norm(hidden, layer.post_feedforward_layernorm)
        hidden = hidden + residual
        return hidden * layer.layer_scalar
