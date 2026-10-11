"""Qwen3.5-family forward pass over vLLM's loaded module.

Three Gated DeltaNet linear-attention layers, then one gated
full-attention layer, repeated. A linear layer keeps one recurrent
state per sequence instead of KV rows; this pass runs every segment
from the zero state (saved state arrives with the state pool). The
norms, projections, rotary, and gated norm are vLLM's own modules, so
the arithmetic matches vLLM's. Quail's arena serves the full layers'
attention.
"""

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.model import text_model
from quail.backends.quail.executor.models.base import ModelPipeline
from quail.cost import budgets


class Qwen35Pipeline(ModelPipeline):
    """Forward passes for Qwen3.5-family checkpoints loaded by vLLM.

    Args:
        model: vLLM's Qwen3_5ForConditionalGeneration module, loaded
            with language_model_only.
        arena: KVArena with one pool per layer at that layer's KV
            geometry (spec.kv_shapes); the linear layers' pools are
            empty.
        spec: The ModelSpec, whose geometry must match the checkpoint's.
        engine_class: Engine class, replaceable by tests.
    """

    # the unified path's paged attention is the one path: the tree
    # merge takes 128-wide heads only, and canvas rows do not apply
    needs_pages = True
    tree_attention = False
    gemm_warmup = False

    def __init__(self, model, arena, *, spec, engine_class=Engine):
        backbone = text_model(model).model
        self.layers = backbone.layers
        self.embed = backbone.embed_tokens
        self.final_norm = backbone.norm
        full = [i for i, layer in enumerate(self.layers)
                if layer.layer_type == "full_attention"]
        expected = [i for i in range(spec.layers) if spec.is_full_layer(i)]
        if len(self.layers) != spec.layers or full != expected:
            raise ValueError(
                f"the loaded model's full-attention layers {full} differ "
                f"from the spec's {expected}")
        if any(getattr(layer, "layer_scale", False) for layer in self.layers):
            raise ValueError("layer scales are not supported")
        attn = self.layers[full[0]].self_attn
        loaded = (attn.num_heads, attn.num_kv_heads, attn.head_dim)
        if loaded != (spec.n_q, spec.n_kv, spec.d_head):
            raise ValueError(
                f"the loaded model's attention geometry (q heads, kv heads, "
                f"head dim) {loaded} differs from the spec's "
                f"{(spec.n_q, spec.n_kv, spec.d_head)}")
        if not attn.attn_output_gate:
            raise ValueError("the forward pass expects gated attention")
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb, fp8=False)
        self.max_chunk_tokens = budgets.kernel_index_cap(spec)

    def forward_chunk(self, chunk):
        hidden, residual = self.backbone_rows(chunk)
        final = chunk.final_indices
        normed, _ = self.final_norm(hidden.index_select(0, final),
                                    residual.index_select(0, final))
        return normed

    def backbone_rows(self, chunk):
        """Return (hidden, residual) after the last layer; their sum is its output.

        The layers keep vLLM's split: a layer returns its feedforward
        output and the residual it was added to, and the next layer's
        input norm sums them.
        """
        meta = chunk.meta
        positions = chunk.positions
        hidden = self.embed(chunk.input_ids)
        residual = None
        for index, layer in enumerate(self.layers):
            if residual is None:
                residual = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            if layer.layer_type == "full_attention":
                # the engine indexes the arena's pools by model layer
                meta["layer"] = index
                x = self._full_attention(layer.self_attn, x, positions, meta)
            else:
                x = self._linear_attention(layer.linear_attn, x, meta)
            x, residual = layer.post_attention_layernorm(x, residual)
            hidden = layer.mlp(x)
        return hidden, residual

    def _full_attention(self, attn, x, positions, meta):
        """Gated attention: per-head [q | gate], partial rotary, sigmoid gate."""
        torch = self.engine.torch
        n = x.shape[0]
        heads, kv_heads, dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        qkv, _ = attn.qkv_proj(x)
        q_gate, k, v = qkv.split([attn.q_size * 2, attn.kv_size, attn.kv_size],
                                 dim=-1)
        q, gate = torch.chunk(q_gate.view(n, heads, 2 * dim), 2, dim=-1)
        q = attn.q_norm(q.contiguous()).view(n, heads * dim)
        k = attn.k_norm(k.view(n, kv_heads, dim)).view(n, kv_heads * dim)
        q, k = attn.rotary_emb(positions, q, k)
        out = self.engine.attention_unified(
            q.view(n, heads, dim), k.view(n, kv_heads, dim).contiguous(),
            v.view(n, kv_heads, dim).contiguous(), meta)
        out = out * torch.sigmoid(gate.reshape(n, heads * dim))
        out, _ = attn.o_proj(out)
        return out

    def _linear_attention(self, linear, x, meta):
        """Gated DeltaNet over the chunk's segments, each from the zero state."""
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
        )
        from vllm.third_party.flash_linear_attention.ops import (
            chunk_gated_delta_rule,
            fused_post_conv_prep,
        )

        torch = self.engine.torch
        n = x.shape[0]
        cu = self._segment_bounds(meta)
        segments = cu.numel() - 1
        mixed_qkvz, _ = linear.in_proj_qkvz(x)
        ba, _ = linear.in_proj_ba(x)
        qkv_size = 2 * linear.key_dim + linear.value_dim
        mixed_qkv, z = mixed_qkvz.split([qkv_size, linear.value_dim], dim=-1)
        b, a = ba.chunk(2, dim=-1)
        weight = linear.conv1d.weight
        conv_dim, kernel = weight.shape[0], weight.shape[-1]
        # the conv kernel reads each segment's window from the slot its
        # index names and writes the final window back there; slot 0
        # is its null slot, so the segments take slots 1 and up
        windows = torch.zeros((segments + 1, conv_dim, kernel - 1),
                              dtype=mixed_qkv.dtype, device=x.device)
        conv_out = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1), weight.view(conv_dim, kernel),
            linear.conv1d.bias, activation=linear.activation,
            conv_states=windows,
            has_initial_state=torch.zeros(segments, dtype=torch.bool,
                                          device=x.device),
            cache_indices=torch.arange(1, segments + 1, dtype=torch.int32,
                                       device=x.device),
            query_start_loc=cu).transpose(0, 1).contiguous()
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_out, a=a.contiguous(), b=b.contiguous(),
            A_log=linear.A_log, dt_bias=linear.dt_bias,
            num_k_heads=linear.num_k_heads, head_k_dim=linear.head_k_dim,
            head_v_dim=linear.head_v_dim, apply_l2norm=True,
            output_g_exp=False)
        core, _ = chunk_gated_delta_rule(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=None,
            output_final_state=False, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False)
        head = linear.head_v_dim
        gated = linear.norm(core.reshape(-1, head), z.reshape(-1, head))
        out, _ = linear.out_proj(gated.reshape(n, -1))
        return out

    def _segment_bounds(self, meta):
        """Cumulative row bounds of the chunk's sequences, int32 on the device.

        One segment per unified sequence; a chunk without arena pages
        falls back to its causal segments.
        """
        torch = self.engine.torch
        unified = meta["unified"]
        cu = unified["cu_q"] if unified is not None else meta["cu_a"]
        return cu.to(dtype=torch.int32)
