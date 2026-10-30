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
from quail.backends.quail.executor.models.base import ModelPipeline, overlaid_rows
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
            arena, num_q_heads=attn.num_heads, num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim, rotary=attn.rotary_emb,
            fp8=False, kernels=kernels)
        # each Gated DeltaNet layer's index among them, its state slot
        gdn_layers = [index for index, layer in enumerate(self.layers)
                      if hasattr(layer, "linear_attn")]
        self.gdn_slots = {index: slot for slot, index in enumerate(gdn_layers)}
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

    def _gdn_inputs(self, mixer, x):
        """Project rows to the convolution input, the gate z, and b and a."""
        mixed = self._linear(x, mixer.in_proj_qkvz)
        beta_alpha = self._linear(x, mixer.in_proj_ba)
        qkv, z = mixed.split([mixer.conv_dim, mixer.value_dim], dim=-1)
        b, a = beta_alpha.chunk(2, dim=-1)
        return dict(qkv=qkv, z=z, b=b, a=a)

    def _gdn_core(self, mixer, inputs, phases, slot, output=None):
        """Run the convolution and recurrence of one Gated DeltaNet layer."""
        conv_weight = mixer.conv1d.weight
        tensors = {
            "qkv": inputs["qkv"], "b": inputs["b"], "a": inputs["a"],
            "conv_weight": conv_weight.view(
                conv_weight.shape[0], conv_weight.shape[2]),
            "conv_bias": mixer.conv1d.bias,
            "A_log": mixer.A_log, "dt_bias": mixer.dt_bias,
        }
        return gdn_core(self.spec, tensors, phases, self.arena, slot,
                        output=output)

    def _gdn_output(self, mixer, core, z):
        """Norm the recurrence output gated by z, then project it out."""
        rows, heads, width = core.shape
        gated = mixer.norm(core.reshape(-1, width), z.reshape(-1, width))
        return self._linear(gated.reshape(rows, heads * width), mixer.out_proj)

    def _attention_inputs(self, attn, x, positions):
        """Project rows to normed, rotated q and k, v, and the output gate."""
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
        return dict(q=q, k=k, v=v.contiguous(), gate=gate.reshape(rows, -1))

    def _attention_core(self, attn, inputs, meta, output=None):
        """Paged attention over the rows and the arena, as (rows, heads * dim)."""
        rows = inputs["q"].shape[0]
        return self.engine.attention_unified(
            inputs["q"].view(rows, attn.num_heads, attn.head_dim),
            inputs["k"].view(rows, attn.num_kv_heads, attn.head_dim),
            inputs["v"].view(rows, attn.num_kv_heads, attn.head_dim),
            meta, output=None if output is None else output.view(
                rows, attn.num_heads, attn.head_dim))

    def _attention_output(self, attn, out, gate):
        """Scale attention by sigmoid of its gate, then project it out."""
        import torch

        return self._linear(out * torch.sigmoid(gate), attn.o_proj)

    def _mixer(self, index, x, positions, chunk):
        """One layer's token mixer over the rows, eagerly."""
        layer = self.layers[index]
        if hasattr(layer, "linear_attn"):
            inputs = self._gdn_inputs(layer.linear_attn, x)
            core = self._gdn_core(layer.linear_attn, inputs, chunk.meta["gdn"],
                                  self.gdn_slots[index])
            return self._gdn_output(layer.linear_attn, core, inputs["z"])
        inputs = self._attention_inputs(layer.self_attn, x, positions)
        out = self._attention_core(layer.self_attn, inputs, chunk.meta)
        return self._attention_output(layer.self_attn, out, inputs["gate"])

    def _mlp(self, mlp, x):
        gate_up = self._linear(x, mlp.gate_up_proj)
        return self._linear(mlp.act_fn(gate_up), mlp.down_proj)

    def _norm(self, norm, x, residual):
        """One Gemma-style norm with the residual add, as (normed, residual)."""
        return self.engine.gemma_add_rms_norm(
            x, residual, norm.weight, norm.variance_epsilon)

    def forward_context(self, num_tokens):
        """Enter vLLM's config, and for experts its forward context and stream."""
        from contextlib import ExitStack

        stack = ExitStack()
        if self.vllm_config is not None:
            from vllm.config import set_current_vllm_config

            stack.enter_context(set_current_vllm_config(self.vllm_config))
        if self.spec.experts:
            import torch
            from vllm.forward_context import set_forward_context

            stack.enter_context(set_forward_context(
                None, self.vllm_config, num_tokens=num_tokens))
            # Register Quail's stream before vLLM can lazily select a different one.
            stack.enter_context(torch.cuda.stream(torch.cuda.current_stream()))
        return stack

    def forward_chunk(self, chunk):
        with self.forward_context(chunk.input_ids.shape[0]):
            return self._forward(chunk)

    def _forward(self, chunk):
        if chunk.meta["gdn"] is None:
            raise ValueError(
                "a Qwen3.5 chunk needs its Gated DeltaNet phases; the "
                "arena has no checkpoint slots")
        hidden = self.embed(chunk.input_ids)
        residual = None
        for index, layer in enumerate(self.layers):
            x, residual = self._norm(layer.input_layernorm, hidden, residual)
            # arena layers are indexed by model layer
            chunk.meta["layer"] = index
            x = self._mixer(index, x, chunk.positions, chunk)
            x, residual = self._norm(layer.post_attention_layernorm, x,
                                     residual)
            hidden = self.feed_forwards[index](x)
        final = chunk.final_indices
        if self.spec.sae is not None:
            # the SAE reads the residual stream after the last layer: the
            # MLP output not yet added (hidden) plus the running residual
            if self.forward_hook is not None:
                stream = hidden + residual
                self.forward_hook(stream, chunk)
                return stream.index_select(0, final)
            return (hidden.index_select(0, final)
                    + residual.index_select(0, final))
        normed, _ = self._norm(self.final_norm, hidden.index_select(0, final),
                               residual.index_select(0, final))
        return normed

    def boundary(self, num_tokens, device):
        """Each mixer kind's inputs, and its output before the out projection.

        A pass runs one layer at a time, so a Gated DeltaNet layer's
        tensors and an attention layer's share storage: the mixer
        inputs, the output gates (z and gate), which the next piece
        reads, and the mixer outputs.
        """
        import torch

        spec, attn = self.spec, self.attention_layer.self_attn
        q_width = attn.num_heads * attn.head_dim
        kv_width = attn.num_kv_heads * attn.head_dim
        value_width = spec.gdn_value_heads * spec.gdn_value_dim
        (qkv,), (q, k, v) = overlaid_rows(
            torch, num_tokens, device,
            [self.gdn_layer.linear_attn.conv_dim], [q_width, kv_width, kv_width])
        (z,), (gate,) = overlaid_rows(
            torch, num_tokens, device, [value_width], [q_width])
        (core,), (attention,) = overlaid_rows(
            torch, num_tokens, device, [value_width], [q_width])
        (b, a), = overlaid_rows(torch, num_tokens, device,
                                [spec.gdn_value_heads, spec.gdn_value_heads])
        return dict(qkv=qkv, z=z, b=b, a=a,
                    core=core.view(num_tokens, spec.gdn_value_heads,
                                   spec.gdn_value_dim),
                    q=q, k=k, v=v, gate=gate, attention=attention)

    def piece(self, index, buffers):
        """Finish the previous layer, then project the next mixer's inputs.

        The previous mixer's output projection and MLP, then the next
        layer's norm and input projections; an attention layer's KV is
        written here, inside the graph.
        """
        if index == 0:
            hidden, residual = self.embed(buffers.ids), None
        else:
            layer = self.layers[index - 1]
            if hasattr(layer, "linear_attn"):
                x = self._gdn_output(layer.linear_attn, buffers.core, buffers.z)
            else:
                x = self._attention_output(
                    layer.self_attn, buffers.attention, buffers.gate)
            x, residual = self._norm(layer.post_attention_layernorm, x,
                                     buffers.residual)
            hidden = self.feed_forwards[index - 1](x)
        if index == len(self.layers):
            normed, _ = self._norm(self.final_norm, hidden, residual)
            buffers.output.copy_(normed)
            return
        layer = self.layers[index]
        x, residual = self._norm(layer.input_layernorm, hidden, residual)
        buffers.residual.copy_(residual)
        if hasattr(layer, "linear_attn"):
            inputs = self._gdn_inputs(layer.linear_attn, x)
        else:
            inputs = self._attention_inputs(layer.self_attn, x, buffers.positions)
        for name, tensor in inputs.items():
            getattr(buffers, name).copy_(tensor)
        if not hasattr(layer, "linear_attn"):
            buffers.write_kv(self.engine, buffers.k, buffers.v, index)

    def mixer(self, index, buffers, chunk):
        """The layer's recurrence or attention over real rows, into buffers."""
        n = chunk.tokens
        layer = self.layers[index]
        if hasattr(layer, "linear_attn"):
            inputs = dict(qkv=buffers.qkv[:n], b=buffers.b[:n], a=buffers.a[:n])
            self._gdn_core(layer.linear_attn, inputs, chunk.meta["gdn"],
                           self.gdn_slots[index], output=buffers.core[:n])
        else:
            inputs = dict(q=buffers.q[:n], k=buffers.k[:n], v=buffers.v[:n])
            self._attention_core(layer.self_attn, inputs, chunk.meta,
                                 output=buffers.attention[:n])
