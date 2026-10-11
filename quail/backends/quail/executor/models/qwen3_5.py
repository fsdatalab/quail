"""Qwen3.5-family forward pass over vLLM's loaded module.

Three Gated DeltaNet linear-attention layers, then one gated
full-attention layer, repeated. A linear layer keeps one recurrent
state per sequence instead of KV rows: the chunk's state plan
(meta["state"], from pack_chunk) says which slot of the arena's state
pool each segment starts from and which it saves to, wave by wave.
The norms, projections, rotary, and gated norm are vLLM's own
modules, so the arithmetic matches vLLM's. Quail's arena serves the
full layers' attention and the linear layers' state.
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
        self.arena = arena
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
                x = self._linear_attention(layer.linear_attn, x, meta, index)
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

    def _linear_attention(self, linear, x, meta, layer):
        """Gated DeltaNet over the chunk's state segments, wave by wave."""
        torch = self.engine.torch
        n = x.shape[0]
        mixed_qkvz, _ = linear.in_proj_qkvz(x)
        ba, _ = linear.in_proj_ba(x)
        qkv_size = 2 * linear.key_dim + linear.value_dim
        mixed_qkv, z = mixed_qkvz.split([qkv_size, linear.value_dim], dim=-1)
        b, a = ba.chunk(2, dim=-1)
        plan = meta.get("state")
        if plan is None:
            # an arena without state slots: every sequence starts from
            # the zero state and saves nothing
            waves, pools = [self._zero_wave(meta)], None
        else:
            waves, pools = plan["waves"], self.arena.state_pools(layer)
        core = torch.empty((n, linear.value_dim), dtype=x.dtype, device=x.device)
        for wave in waves:
            out = self._wave(linear, wave, mixed_qkv, a, b, pools)
            if wave["rows"] is None:
                core = out
            else:
                core.index_copy_(0, wave["rows"], out)
        head = linear.head_v_dim
        gated = linear.norm(core.reshape(-1, head), z.reshape(-1, head))
        out, _ = linear.out_proj(gated.reshape(n, -1))
        return out

    def _wave(self, linear, wave, mixed_qkv, a, b, pools):
        """One conv call and one delta-rule call over a wave's rows.

        Returns the wave's rows of the core output, (rows, value_dim).
        """
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_fn,
        )
        from vllm.third_party.flash_linear_attention.ops import (
            chunk_gated_delta_rule,
            fused_post_conv_prep,
        )

        torch = self.engine.torch
        rows, m = wave["rows"], wave["n"]
        if rows is not None:
            mixed_qkv = mixed_qkv.index_select(0, rows)
            a, b = a.index_select(0, rows), b.index_select(0, rows)
        weight = linear.conv1d.weight
        conv_dim, kernel = weight.shape[0], weight.shape[-1]
        # the conv kernel reads each sequence's window from the slot
        # its index names and writes the final window back there; slot
        # 0 is its null slot, so the sequences take slots 1 and up of
        # a scratch copy, and the pool sees only the windows that save
        windows = torch.zeros((m + 1, conv_dim, kernel - 1),
                              dtype=mixed_qkv.dtype, device=mixed_qkv.device)
        if pools is None:
            initial = None
        else:
            s_pool, conv_pool = pools
            windows[1:].copy_(conv_pool.index_select(0, wave["init"]))
            initial = s_pool.index_select(0, wave["init"])
        conv_out = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1), weight.view(conv_dim, kernel),
            linear.conv1d.bias, activation=linear.activation,
            conv_states=windows, has_initial_state=wave["has_init"],
            cache_indices=torch.arange(1, m + 1, dtype=torch.int32,
                                       device=mixed_qkv.device),
            query_start_loc=wave["cu"]).transpose(0, 1).contiguous()
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=conv_out, a=a.contiguous(), b=b.contiguous(),
            A_log=linear.A_log, dt_bias=linear.dt_bias,
            num_k_heads=linear.num_k_heads, head_k_dim=linear.head_k_dim,
            head_v_dim=linear.head_v_dim, apply_l2norm=True,
            output_g_exp=False)
        saving = wave["save_index"] is not None
        out, final = chunk_gated_delta_rule(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=initial,
            output_final_state=saving, cu_seqlens=wave["cu"],
            use_qk_l2norm_in_kernel=False)
        if saving:
            index, slots = wave["save_index"], wave["save_slots"]
            s_pool.index_copy_(0, slots, final.index_select(0, index)
                               .to(s_pool.dtype))
            conv_pool.index_copy_(0, slots,
                                  windows[1:].index_select(0, index))
        return out.squeeze(0).reshape(out.shape[1], -1)

    def _zero_wave(self, meta):
        """The one wave of a chunk without a state plan: every sequence from zero."""
        torch = self.engine.torch
        unified = meta["unified"]
        cu = (unified["cu_q"] if unified is not None else meta["cu_a"]).to(
            dtype=torch.int32)
        m = cu.numel() - 1
        return dict(rows=None, cu=cu, n=m, init=None,
                    has_init=torch.zeros(m, dtype=torch.bool, device=cu.device),
                    save_index=None, save_slots=None)
