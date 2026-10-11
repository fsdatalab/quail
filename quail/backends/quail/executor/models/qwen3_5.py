"""Qwen3.5 forward pass over vLLM's loaded Qwen3.5 text model.

Layers alternate three Gated DeltaNet layers and one gated attention
layer. Each is input RMSNorm, token mixer, post-attention RMSNorm,
gated MLP. Both RMSNorms are Gemma-style (weight + 1).

A Gated DeltaNet mixer projects to q, k, v, and z in one GEMM and to
b and a in a small bf16 one, runs the convolution and recurrence from
the chunk's checkpoint phases (executor/gdn.py), norms the result
gated by z, and projects out. An attention mixer projects to the
queries with a per-head output gate, norms and rotates q and k, runs
paged attention, scales the result by sigmoid of the gate, and
projects out.
"""

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.gdn import gdn_core
from quail.backends.quail.executor.models.base import ModelPipeline


class Qwen35Pipeline(ModelPipeline):
    """Forward passes for Qwen3.5 checkpoints loaded by vLLM."""

    # head dim 256 runs paged attention only, and a hybrid chunk reads
    # its documents from checkpoints, which the unified path plans
    needs_pages = True
    tree_attention = False
    warm_row_classes = False
    warm_joins = False

    def __init__(self, model, arena, *, spec, kernels="quail",
                 engine_class=Engine, use_gdn_kernels=True):
        import torch

        text = getattr(model, "language_model", model)
        self.spec = spec
        self.arena = arena
        self.use_gdn_kernels = use_gdn_kernels
        self.layers = text.model.layers
        self.embed = text.model.embed_tokens
        self.final_norm = text.model.norm
        self.attention_layer = next(
            layer for layer in self.layers if hasattr(layer, "self_attn"))
        self.gdn_layer = next(
            layer for layer in self.layers if hasattr(layer, "linear_attn"))
        attn = self.attention_layer.self_attn
        fp8 = attn.qkv_proj.weight.dtype == torch.float8_e4m3fn
        self.engine = engine_class(
            arena, n_q=attn.num_heads, n_kv=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=fp8, kernels=kernels)
        # The fused kernels compute element offsets in 32-bit ints, so
        # a chunk needs rows x widest_row < 2^31.
        widest = max(
            self.gdn_layer.linear_attn.in_proj_qkvz.weight.shape[0],
            attn.qkv_proj.weight.shape[0],
            self.attention_layer.mlp.gate_up_proj.weight.shape[0])
        self.max_chunk_tokens = (2**31 - 1) // widest

    def linears(self):
        gdn = self.gdn_layer.linear_attn
        attn = self.attention_layer.self_attn
        mlp = self.attention_layer.mlp
        return (gdn.in_proj_qkvz, gdn.out_proj, attn.qkv_proj, attn.o_proj,
                mlp.gate_up_proj, mlp.down_proj)

    def _linear(self, x, module):
        """One projection of x, quantized when the weights are fp8."""
        x_q, x_s = self.engine.quant(x)
        return self.engine.gemm(x_q, x_s, module)

    def _gated_delta_net(self, mixer, x, phases, slot):
        import torch

        mixed = self._linear(x, mixer.in_proj_qkvz)
        # in_proj_ba is bf16 in the fp8 checkpoint too
        beta_alpha = torch.nn.functional.linear(x, mixer.in_proj_ba.weight)
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
        core = gdn_core(self.spec, tensors, phases, self.arena, slot,
                        use_kernels=self.use_gdn_kernels)
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
        d_in, d_scale = self.engine.activation_quant(gate_up)
        return self.engine.gemm(d_in, d_scale, mlp.down_proj)

    def forward_chunk(self, chunk):
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
        for layer in self.layers:
            if residual is None:
                residual = hidden
                x = layer.input_layernorm(hidden)
            else:
                x, residual = layer.input_layernorm(hidden, residual)
            if hasattr(layer, "linear_attn"):
                x = self._gated_delta_net(layer.linear_attn, x, phases, slot)
                slot += 1
                # arena layers are indexed by model layer
                meta["layer"] += 1
            else:
                x = self._attention(layer.self_attn, x, chunk.positions, chunk)
            x, residual = layer.post_attention_layernorm(x, residual)
            hidden = self._mlp(layer.mlp, x)
        final = chunk.final_indices
        normed, _ = self.final_norm(hidden.index_select(0, final),
                                    residual.index_select(0, final))
        return normed
