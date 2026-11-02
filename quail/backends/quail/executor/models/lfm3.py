"""LFM3 forward pass over the text weights of an LFM3 checkpoint.

Each layer is an operator RMSNorm, a token mixer, a post-operator
RMSNorm, the residual add, then an FFN RMSNorm, a gated MLP, and a
post-FFN RMSNorm before the second residual add. The mixer is a short
convolution of seven taps, or grouped query attention with per-head q
and k norms and a sigmoid gate per head on its output. Attention
layers alternate sliding (a window of `sliding_window` tokens, rotary
over the whole head) and full (rotary over the first half of the
head's frequency pairs, the rest left unrotated).

vLLM has no LFM3 model, so load_lfm3 reads the checkpoint's
language-model tensors into plain objects that hold a weight, which is
all the engine's bf16 GEMM reads.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.models import lfm3_kernels as kernels
from quail.backends.quail.executor.models.base import ModelPipeline

TEXT_PREFIX = "model.language_model."


def is_lfm3(path) -> bool:
    """Whether a checkpoint directory holds an LFM3 or LFM3-VL model."""
    config = Path(path) / "config.json"
    return (config.is_file() and json.loads(config.read_text()).get(
        "model_type") in ("lfm3", "lfm3_vl"))


def text_config(path) -> dict:
    """The language model's config, from an LFM3 or LFM3-VL config.json."""
    config = json.loads((Path(path) / "config.json").read_text())
    return config.get("text_config", config)


def rotary_tables(torch, config, layer_type, max_positions, device):
    """Cosine and sine of each position's angles for one attention layer type.

    LFM3 keeps the full head's frequency progression and zeroes the
    frequencies past partial_rotary_factor, so those pairs stay
    unrotated. The tables are rounded to bf16, as the checkpoint's own
    code rounds them before the fp32 rotation.

    Returns:
        (cos, sin), each (max_positions, head_dim // 2) float32.
    """
    params = config["rope_parameters"][layer_type]
    if params.get("rope_type", "default") != "default":
        raise ValueError(f"LFM3 rotary type {params['rope_type']!r} is not "
                         f"supported")
    dim = config["head_dim"]
    rotated = int(params.get("partial_rotary_factor", 1.0) * dim) // 2
    inverse = 1.0 / params["rope_theta"] ** (
        torch.arange(0, 2 * rotated, 2, dtype=torch.float32) / dim)
    inverse = torch.cat((inverse, torch.zeros(dim // 2 - rotated)))
    angles = torch.arange(max_positions, dtype=torch.float32)[:, None] * inverse
    return tuple(t.to(torch.bfloat16).float().to(device)
                 for t in (angles.cos(), angles.sin()))


class Lfm3Model:
    """The loaded text model: embedding, layers, final norm, and tied head.

    Attributes:
        model: Namespace with embed_tokens (the embedding weight),
            layers, and embedding_norm.
        lm_head: Namespace holding the output head's weight, the
            embedding's. retain_answer_head keeps the rows it needs and
            clears this attribute.
        config: The language model's config.
        rotary: Per attention layer type, its (cos, sin) tables.
    """

    def __init__(self, embed, layers, final_norm, config, rotary):
        self.model = SimpleNamespace(embed_tokens=embed, layers=layers,
                                     embedding_norm=final_norm)
        self.lm_head = SimpleNamespace(weight=embed)
        self.config = config
        self.rotary = rotary

    def register_buffer(self, name, tensor, **_):
        """Keep a tensor as an attribute, as torch.nn.Module does."""
        setattr(self, name, tensor)


def load_lfm3(path, device="cuda", max_positions=None):
    """Load an LFM3 checkpoint's language model onto the GPU in bf16.

    The vision tower and projector of an LFM3-VL checkpoint are not
    read. q, k, v, and the output gate share one weight, and so do the
    MLP's w1 and w3, so each runs as one GEMM.

    Args:
        path: The local checkpoint directory.
        device: Where the weights go.
        max_positions: Rotary table length; the config's
            max_position_embeddings when None.

    Raises:
        ValueError: The checkpoint has a tensor the model does not use,
            lacks one it needs, or uses a feature this pass lacks.
    """
    import torch
    from safetensors import safe_open

    config = text_config(path)
    if config.get("conv_bias"):
        raise ValueError("an LFM3 convolution with bias is not supported")
    if not config.get("use_sandwich_norm", False):
        raise ValueError("an LFM3 model without sandwich norms is not "
                         "supported")
    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith(TEXT_PREFIX):
                    tensors[name[len(TEXT_PREFIX):]] = f.get_tensor(name)

    def take(name):
        try:
            tensor = tensors.pop(name)
        except KeyError:
            raise ValueError(f"the checkpoint has no {name}") from None
        return tensor.to(device=device, dtype=torch.bfloat16)

    eps = config["norm_eps"]

    def norm(name):
        return SimpleNamespace(weight=take(name), variance_epsilon=eps)

    def linear(*names):
        weights = [take(name) for name in names]
        return SimpleNamespace(
            weight=weights[0] if len(weights) == 1 else torch.cat(weights))

    layers = []
    for index, kind in enumerate(config["layer_types"]):
        p = f"layers.{index}."
        layer = SimpleNamespace(
            kind=kind,
            operator_norm=norm(p + "operator_norm.weight"),
            post_operator_norm=norm(p + "post_operator_norm.weight"),
            ffn_norm=norm(p + "ffn_norm.weight"),
            post_ffn_norm=norm(p + "post_ffn_norm.weight"),
            w13=linear(p + "feed_forward.w1.weight", p + "feed_forward.w3.weight"),
            w2=linear(p + "feed_forward.w2.weight"))
        if kind == "conv":
            taps = take(p + "conv.conv.weight")
            layer.in_proj = linear(p + "conv.in_proj.weight")
            layer.out_proj = linear(p + "conv.out_proj.weight")
            layer.taps = taps.view(taps.shape[0], taps.shape[2]).contiguous()
        else:
            a = p + "self_attn."
            layer.qkv_gate = linear(a + "q_proj.weight", a + "k_proj.weight",
                                    a + "v_proj.weight", a + "gate_proj.weight")
            layer.o_proj = linear(a + "o_proj.weight")
            layer.q_norm = norm(a + "q_norm.weight")
            layer.k_norm = norm(a + "k_norm.weight")
        layers.append(layer)
    embed = take("embed_tokens.weight")
    final_norm = norm("embedding_norm.weight")
    if tensors:
        raise ValueError(f"the LFM3 pass does not use {sorted(tensors)[:5]}")
    positions = max_positions or config["max_position_embeddings"]
    rotary = {kind: rotary_tables(torch, config, kind, positions, device)
              for kind in set(config["layer_types"]) - {"conv"}}
    return Lfm3Model(embed, layers, final_norm, config, rotary)


class Lfm3Pipeline(ModelPipeline):
    """Forward passes for LFM3 checkpoints loaded by load_lfm3, under set attention.

    Every sequence in a chunk is one pass: a document and its questions,
    packed whole, with no KV kept. The convolution and sliding-window
    layers run causally through each sequence from zero state. On the
    full-attention layers a sequence with one question reads itself in
    both directions, as d1's set attention reads a lone question.
    """

    needs_pages = False
    tree_attention = False
    warm_row_classes = False
    warm_joins = False
    whole_sequences = True

    def __init__(self, model, arena, *, spec, engine_class=Engine):
        if spec.attention_mask != "set":
            raise ValueError(
                f"{spec.name}: the LFM3 pass runs set attention only, not "
                f"{spec.attention_mask!r}")
        config = model.config
        loaded = (len(model.model.layers), config["hidden_size"],
                  config["num_attention_heads"], config["num_key_value_heads"],
                  config["head_dim"], config.get("sliding_window", 0),
                  config["conv_L_cache"])
        expected = (spec.layers, spec.hidden, spec.num_q_heads,
                    spec.num_kv_heads, spec.d_head, spec.sliding_window,
                    spec.conv_width)
        if loaded != expected:
            raise ValueError(
                f"the loaded model's geometry (layers, hidden, q heads, kv "
                f"heads, head dim, window, conv width) {loaded} differs from "
                f"the spec's {expected}")
        kinds = tuple("short_conv" if kind == "conv" else kind
                      for kind in config["layer_types"])
        if kinds != spec.layer_kinds:
            raise ValueError("the loaded model's layer kinds differ from "
                             "the spec's")
        self.spec = spec
        self.arena = arena
        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.final_norm = model.model.embedding_norm
        self.window = spec.sliding_window
        self.num_heads = spec.num_q_heads
        self.num_kv_heads = spec.num_kv_heads
        self.head_dim = spec.d_head
        self.engine = engine_class(
            arena, num_q_heads=spec.num_q_heads, num_kv_heads=spec.num_kv_heads,
            head_dim=spec.d_head, rotary=None, fp8=False)
        # per attention layer kind, its [cos | sin] table, as the fused
        # q/k norm and rotary kernel reads it
        self.cos_sin = {kind: self.engine.torch.cat(tables, -1).contiguous()
                        for kind, tables in model.rotary.items()}
        attention = next(layer for layer in self.layers if layer.kind != "conv")
        conv = next(layer for layer in self.layers if layer.kind == "conv")
        self.attention_layer, self.conv_layer = attention, conv
        widest = max(conv.in_proj.weight.shape[0],
                     attention.qkv_gate.weight.shape[0],
                     conv.w13.weight.shape[0])
        self.max_chunk_tokens = (2**31 - 1) // widest

    def linears(self):
        conv, attention = self.conv_layer, self.attention_layer
        return (conv.in_proj, conv.out_proj, attention.qkv_gate,
                attention.o_proj, conv.w13, conv.w2)

    def convolutions(self):
        """Return the convolution weights and activation used by this model."""
        return ((self.conv_layer.taps, None, False),)

    def _linear(self, x, module):
        return self.engine.gemm(x, None, module)

    def _norm(self, x, norm):
        return self.engine.norm_rows(x, norm.weight, norm.variance_epsilon)

    def _short_conv(self, layer, x, starts):
        conv = kernels.gated_conv(self._linear(x, layer.in_proj), layer.taps,
                                  starts)
        return self._linear(conv, layer.out_proj)

    def _attention(self, layer, x, positions, meta):
        rows = x.shape[0]
        H, KH, D = self.num_heads, self.num_kv_heads, self.head_dim
        projected = self._linear(x, layer.qkv_gate)
        q, k = self.engine.qk_norm_rope_heads(
            projected, positions, num_q_heads=H, num_kv_heads=KH, head_dim=D,
            q_weight=layer.q_norm.weight, k_weight=layer.k_norm.weight,
            eps=layer.q_norm.variance_epsilon,
            cos_sin_cache=self.cos_sin[layer.kind])
        q, k = q.view(rows, H, D), k.view(rows, KH, D)
        v = projected[:, (H + KH) * D:(H + 2 * KH) * D].contiguous().view(
            rows, KH, D)
        cu, longest = meta["cu_a"], meta["max_a"]
        if layer.kind == "sliding_attention":
            out, _ = self.engine._fa(q, k, v, cu, cu, longest, longest,
                                     causal=True, window=(self.window - 1, 0))
        elif meta.get("set_rows") is None:
            out, _ = self.engine._fa(q, k, v, cu, cu, longest, longest,
                                     causal=False)
        else:
            out = self._set_attention(q, k, v, meta)
        out = kernels.gate_heads(out.reshape(rows, H * D),
                                 projected[:, (H + 2 * KH) * D:])
        return self._linear(out, layer.o_proj)

    def _set_attention(self, q, k, v, meta):
        """Full attention under set attention, for passes of several questions.

        The state reads its whole sequence. Each question reads its
        state and itself; the two reads merge by log-sum-exp.
        """
        torch, fa, plan = self.engine.torch, self.engine._fa, meta["set_rows"]
        out = torch.empty_like(q)
        state, asked = plan["state"], plan["asked"]
        state_out, _ = fa(q.index_select(0, state), k, v, plan["cu_state"],
                          meta["cu_a"], plan["max_state"], meta["max_a"],
                          causal=False)
        out.index_copy_(0, state, state_out)
        q_asked = q.index_select(0, asked)
        read = plan["read_state"]
        over_state, lse_state = fa(
            q_asked, k.index_select(0, read), v.index_select(0, read),
            plan["cu_asked"], plan["cu_read"], plan["max_asked"],
            plan["max_read"], causal=False)
        own, lse_own = fa(
            q_asked, k.index_select(0, asked), v.index_select(0, asked),
            plan["cu_own"], plan["cu_own"], plan["max_own"], plan["max_own"],
            causal=False)
        return kernels.merge_questions(
            out, asked, over_state, lse_state.contiguous(), own,
            lse_own.contiguous())

    def _plan_set_rows(self, chunk, meta):
        """Row indices and bounds of the set attention calls, or None.

        A chunk whose every sequence is one question reads each sequence
        in both directions with one call; it needs no plan.
        """
        import numpy as np

        passes = meta.get("passes", ())
        if not any(len(p) == 3 and len(p[2]) > 1 for p in passes):
            return None
        state, asked, read = [], [], []
        cu_state, cu_asked, cu_read, cu_own = [0], [0], [0], [0]
        for entry in passes:
            if len(entry) == 2:
                (first, state_end), questions = entry, ()
            else:
                first, state_end, questions = entry
            state.append(np.arange(first, state_end))
            cu_state.append(cu_state[-1] + state_end - first)
            if not questions:
                continue
            total = int(sum(questions))
            asked.append(np.arange(state_end, state_end + total))
            read.append(np.arange(first, state_end))
            cu_asked.append(cu_asked[-1] + total)
            cu_read.append(cu_read[-1] + state_end - first)
            for length in questions:
                cu_own.append(cu_own[-1] + int(length))
        torch, device = self.engine.torch, chunk.input_ids.device

        def rows(parts):
            return torch.from_numpy(np.concatenate(parts).astype(np.int64)).to(device)

        def bounds(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        def widest(values):
            return int(np.diff(values).max())

        return dict(
            state=rows(state), asked=rows(asked), read_state=rows(read),
            cu_state=bounds(cu_state), cu_asked=bounds(cu_asked),
            cu_read=bounds(cu_read), cu_own=bounds(cu_own),
            max_state=widest(cu_state), max_asked=widest(cu_asked),
            max_read=widest(cu_read), max_own=widest(cu_own))

    def _mlp(self, layer, x):
        gate_up = self._linear(x, layer.w13)
        d_in, d_scale = self.engine.activation_quant(gate_up)
        return self.engine.gemm(d_in, d_scale, layer.w2)

    def forward_chunk(self, chunk):
        import torch.nn.functional as F

        meta = chunk.meta
        if meta.get("unified") is not None or meta.get("reads") is not None:
            raise ValueError("an LFM3 chunk packs whole sequences without "
                             "arena pages")
        lengths = tuple(end - first for _, first, _, end in chunk.row_spans)
        if sum(lengths) != chunk.tokens:
            raise ValueError("an LFM3 chunk holds whole sequences only")
        starts = kernels.sequence_starts(self.engine.torch, lengths,
                                         chunk.input_ids.device)
        meta["set_rows"] = self._plan_set_rows(chunk, meta)
        hidden = F.embedding(chunk.input_ids, self.embed)
        x = self._norm(hidden, self.layers[0].operator_norm)
        for index, layer in enumerate(self.layers):
            if layer.kind == "conv":
                y = self._short_conv(layer, x, starts)
            else:
                y = self._attention(layer, x, chunk.positions, meta)
            x = kernels.post_norm_add_norm(
                y, hidden, layer.post_operator_norm.weight, layer.ffn_norm.weight,
                layer.ffn_norm.variance_epsilon)
            following = (self.layers[index + 1].operator_norm.weight
                         if index + 1 < len(self.layers) else None)
            x = kernels.post_norm_add_norm(
                self._mlp(layer, x), hidden, layer.post_ffn_norm.weight,
                following, layer.post_ffn_norm.variance_epsilon)
        final = hidden.index_select(0, chunk.final_indices)
        return self._norm(final, self.final_norm)
