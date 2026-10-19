"""Qwen hybrid forward pass over vLLM's loaded dense or MoE text model.

Layers alternate three Gated DeltaNet layers and one gated attention
layer. Each is input RMSNorm, token mixer, post-attention RMSNorm,
dense or expert feed-forward block. Both RMSNorms are Gemma-style (weight + 1).

A Gated DeltaNet mixer projects to q, k, v, and z in one GEMM and to
b and a in a small bf16 one, runs the convolution and recurrence from
the chunk's checkpoint phases (executor/gdn.py), norms the result
gated by z, and projects out. An attention mixer projects to the
queries with a per-head output gate, norms and rotates q and k, runs
paged attention, scales the result by sigmoid of the gate, and
projects out.
"""

from functools import partial

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.gdn import gdn_core
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.executor.moe import init_moe_workspace
from quail.cost.budgets import kernel_index_cap


class Qwen35Pipeline(ModelPipeline):
    """Forward passes for dense and MoE Qwen hybrid checkpoints."""

    # head dim 256 runs paged attention only, and a hybrid chunk reads
    # its documents from checkpoints, which the unified path plans
    needs_pages = True
    tree_attention = False
    warm_row_classes = False
    warm_joins = False
    gemm_warmup = False

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine):
        text = getattr(model, "language_model", model)
        self.spec = spec
        self.arena = arena
        # vLLM's norms and rotary dispatch read the current config
        self.vllm_config = getattr(model, "quail_vllm_config", None)
        self.layers = text.model.layers
        self.embed = text.model.embed_tokens
        self.final_norm = text.model.norm
        self.attention_layer = next(
            layer for layer in self.layers if hasattr(layer, "self_attn"))
        self.gdn_layer = next(
            layer for layer in self.layers if hasattr(layer, "linear_attn"))
        attn = self.attention_layer.self_attn
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=False, kernels=kernels)
        self.feed_forwards = tuple(
            layer.mlp if spec.experts else partial(self._mlp, layer.mlp)
            for layer in self.layers)
        if spec.experts:
            init_moe_workspace()
            self.warm_tokens += (4096, 8192, 16384)
            self.warm_row_classes = True
        self.max_chunk_tokens = kernel_index_cap(spec)

    def convolutions(self):
        """Return the convolution weights and activation used by this model."""
        conv = self.gdn_layer.linear_attn.conv1d
        weight = conv.weight.view(conv.weight.shape[0], conv.weight.shape[2])
        return ((weight, conv.bias, True),)

    def _linear(self, x, module):
        """Apply the projection using its loaded quantization method."""
        return module.quant_method.apply(module, x, getattr(module, "bias", None))

    def _gated_delta_net(self, mixer, x, phases, slot):
        mixed = self._linear(x, mixer.in_proj_qkvz)
        beta_alpha = self._linear(x, mixer.in_proj_ba)
        qkv, z = mixed.split([mixer.conv_dim, mixer.value_dim], dim=-1)
        b, a = beta_alpha.chunk(2, dim=-1)
        conv_weight = mixer.conv1d.weight
        tensors = {
            "qkv": qkv, "b": b, "a": a,
            "conv_weight": conv_weight.view(
                conv_weight.shape[0], conv_weight.shape[2]),
            "conv_bias": mixer.conv1d.bias,
            "A_log": mixer.A_log, "dt_bias": mixer.dt_bias,
        }
        core = gdn_core(self.spec, tensors, phases, self.arena, slot)
        rows, heads, width = core.shape
        gated = mixer.norm(core.reshape(-1, width),
                           z.reshape(-1, width))
        return self._linear(gated.reshape(rows, heads * width),
                            mixer.out_proj)

    def _attention(self, attn, x, positions, chunk):
        import torch

        qkv = self._linear(x, attn.qkv_proj)
        rows = qkv.shape[0]
        q_gate, k, v = qkv.split(
            [attn.q_size * 2, attn.kv_size, attn.kv_size], dim=-1)
        # per head: head_dim query columns, then head_dim gate columns
        q, gate = q_gate.view(rows, attn.num_heads, -1).chunk(2, dim=-1)
        q = attn.q_norm(q.reshape(rows, attn.num_heads, attn.head_dim))
        k = attn.k_norm(k.reshape(rows, attn.num_kv_heads, attn.head_dim))
        q, k = attn.rotary_emb(positions, q.reshape(rows, -1),
                               k.reshape(rows, -1))
        out = self.engine.attention_unified(
            q.view(rows, attn.num_heads, attn.head_dim),
            k.view(rows, attn.num_kv_heads, attn.head_dim),
            v.contiguous().view(rows, attn.num_kv_heads, attn.head_dim),
            chunk.meta)
        out = out * torch.sigmoid(gate.reshape(rows, -1))
        return self._linear(out, attn.o_proj)

    def _mlp(self, mlp, x):
        gate_up = self._linear(x, mlp.gate_up_proj)
        return self._linear(mlp.act_fn(gate_up), mlp.down_proj)

    def _norm(self, norm, x, residual):
        """One Gemma-style norm with the residual add, as (normed, residual)."""
        return self.engine.gemma_add_rms_norm(
            x, residual, norm.weight, norm.variance_epsilon)

    def forward_chunk(self, chunk):
        from contextlib import nullcontext

        from vllm.config import set_current_vllm_config
        from vllm.forward_context import set_forward_context

        context = (nullcontext() if self.vllm_config is None
                   else set_current_vllm_config(self.vllm_config))
        forward = (set_forward_context(
            None, self.vllm_config, num_tokens=chunk.input_ids.shape[0])
            if self.spec.experts else nullcontext())
        with context, forward:
            return self._forward(chunk)

    def _forward(self, chunk):
        meta = chunk.meta
        meta["layer"] = 0
        phases = meta["gdn"]
        if phases is None:
            raise ValueError(
                "a Qwen3.5 chunk needs its Gated DeltaNet phases; the "
                "arena has no checkpoint slots")
        hidden = self.embed(chunk.input_ids)
        residual = None
        slot = 0
        for layer, feed_forward in zip(self.layers, self.feed_forwards):
            x, residual = self._norm(layer.input_layernorm, hidden, residual)
            if hasattr(layer, "linear_attn"):
                x = self._gated_delta_net(layer.linear_attn, x, phases, slot)
                slot += 1
                # arena layers are indexed by model layer
                meta["layer"] += 1
            else:
                x = self._attention(layer.self_attn, x, chunk.positions, chunk)
            x, residual = self._norm(layer.post_attention_layernorm, x,
                                     residual)
            hidden = feed_forward(x)
        final = chunk.final_indices
        normed, _ = self._norm(self.final_norm, hidden.index_select(0, final),
                               residual.index_select(0, final))
        return normed
