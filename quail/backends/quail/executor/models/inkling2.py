"""Inkling 2 Mini forward pass over the text weights of its checkpoint.

The model has 24 layers. Layers 6 to 17 run three times with shared
weights, so a token makes 48 layer visits; each visit keeps its own KV
and convolution state, so the arena and the checkpoint slots index
visits, not layers. Passes 2 and 3 restart from the state before the
loop and read messages from the previous pass.

Each visit is a block attention residual into the attention input, the
attention, a short convolution of its output, a block attention residual
into the expert input, the experts, and a short convolution of their
output added to the residual stream. A block attention residual mixes
the completed six-layer blocks and the running partial block with a
learned query instead of one running sum.

Attention has 24 query heads, 8 KV heads of 128, a learned relative
bias over the last 1,024 distances, a sink logit per token and head,
and a sigmoid output gate. Keys and values pass a short convolution
before the cache: static taps on the 20 sliding-window layers, static
plus per-token taps on the 4 full-attention layers. The experts are
128 softcapped SwiGLU experts, 8 per token: NVFP4 in 20 layers and
MXFP8 in the last 4, with quantized activations as the checkpoint
declares.

vLLM 0.29 has no Inkling model, so load_inkling2 reads the checkpoint
itself.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quail.backends.quail.executor.attention import Engine
from quail.backends.quail.executor.models import inkling2_kernels as kernels
from quail.backends.quail.executor.models.base import ModelPipeline

ROOT = "model.llm."
# the parts of the checkpoint the text pass does not read
SKIPPED_PARTS = ("model.visual.",)
# the convolution state columns of one visit: key, value, attention
# output, expert output
STATE_CHANNELS = {"k": (0, 1024), "v": (1024, 2048), "attn": (2048, 4352),
                  "mlp": (4352, 6656)}
# the message gates of a loop layer, by checkpoint controller name
GATES = (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp"))


def is_inkling2(path) -> bool:
    """Whether a checkpoint directory holds an Inkling 2 looped model."""
    config = Path(path) / "config.json"
    return (config.is_file() and json.loads(config.read_text()).get(
        "model_type") == "inkling2_loop")


def text_config(path) -> dict:
    return json.loads((Path(path) / "config.json").read_text())["text_config"]


def visit_layers(config) -> list[int]:
    """The layer of each visit, in run order: 6 + 3 * 12 + 6 visits."""
    loop = config["loop_config"]
    start, end = loop["core_start"], loop["core_end"]
    core = list(range(start, end))
    return (list(range(start)) + core * loop["num_passes"]
            + list(range(end, config["num_hidden_layers"])))


def visit_kinds(config) -> list[str]:
    local = set(config["local_layer_ids"])
    return ["sliding_attention" if layer in local else "full_attention"
            for layer in visit_layers(config)]


class Inkling2Model:
    """The loaded text model: embedding, layers, and output head.

    Attributes:
        lm_head: Namespace holding the output head's weight;
            retain_answer_head keeps the rows it needs.
        quail_logit_softcap: The cap of the output logits.
    """

    def __init__(self, embed, embed_norm, layers, final_norm, output_query,
                 head, config):
        self.embed = embed
        self.embed_norm = embed_norm
        self.layers = layers
        self.final_norm = final_norm
        self.output_query = output_query
        self.lm_head = SimpleNamespace(weight=head)
        self.config = config
        self.quail_logit_softcap = float(config["final_logit_softcapping"])

    def register_buffer(self, name, tensor, **_):
        setattr(self, name, tensor)


def _dequantize_mxfp8(torch, weight, scale):
    """bf16 values of e4m3 weights with a power-of-two scale per 32 columns."""
    experts, rows, columns = weight.shape
    out = torch.empty((experts, rows, columns), dtype=torch.bfloat16,
                      device=weight.device)
    for first in range(0, experts, 8):
        part = slice(first, first + 8)
        values = weight[part].float().view(*scale[part].shape, 32)
        values = values * torch.exp2(scale[part].float() - 127)[..., None]
        out[part] = values.view(-1, rows, columns).to(torch.bfloat16)
    return out


def load_inkling2(path, device="cuda"):
    """Load an Inkling 2 Mini checkpoint's language model onto the GPU.

    Raises:
        ValueError: The checkpoint lacks a tensor the pass reads.
    """
    import torch
    from safetensors import safe_open
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    config = text_config(path)
    hidden = config["hidden_size"]
    scale = hidden ** -0.5
    files = {}
    for file in sorted(Path(path).glob("model*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as f:
            files.update(dict.fromkeys(
                (name for name in f.keys()
                 if not name.startswith(SKIPPED_PARTS)), file))
    handles = {}

    def take(name, dtype=None):
        full = ROOT + name
        if full not in files:
            raise ValueError(f"the checkpoint is missing {full}")
        file = files[full]
        if file not in handles:
            handles[file] = safe_open(str(file), framework="pt",
                                      device=str(device))
        tensor = handles[file].get_tensor(full)
        return tensor if dtype is None else tensor.to(dtype)

    fp32 = torch.float32
    local = set(config["local_layer_ids"])
    loop = config["loop_config"]
    start, end = loop["core_start"], loop["core_end"]
    queries = take("middle_cycle_cross_pass_queries", fp32) * scale
    kv_queries = take("middle_cycle_cross_pass_kv_queries", fp32) * scale
    layers = []
    for index in range(config["num_hidden_layers"]):
        def own(name, index=index):
            return f"layers.{index}.{name}"

        is_local = index in local
        layer = SimpleNamespace(index=index, is_local=is_local,
                                boundary=index % config[
                                    "n_layers_per_attn_res_block"] == 0)
        q_side = [take(own("attn.wq_du.weight")),
                  take(own("attn.gate_attn.weight")),
                  take(own("attn.wr_du.weight")),
                  take(own("attn.attn_sink_proj.weight"))]
        k_side = [take(own("attn.wk_dv.weight"))]
        v_side = [take(own("attn.wv_dv.weight"))]
        if is_local:
            layer.k_conv = take(own("attn.k_sconv.weight"), fp32).view(-1, 4)
            layer.v_conv = take(own("attn.v_sconv.weight"), fp32).view(-1, 4)
            layer.attn_norm = take(own("attn_norm.weight")).view(1, hidden)
        else:
            k_side.append(take(own("attn.k_sconv.weight_proj")))
            v_side.append(take(own("attn.v_sconv.weight_proj")))
            layer.k_conv = take(own("attn.k_sconv.static_w"), fp32)
            layer.v_conv = take(own("attn.v_sconv.static_w"), fp32)
            layer.attn_norm = torch.stack(
                [take(own(f"{part}_input_norm.weight"))
                 for part in ("q", "k", "v")])
        widths = [[w.shape[0] for w in side] for side in (q_side, k_side,
                                                          v_side)]
        layer.w_all = torch.cat(q_side + k_side + v_side).contiguous()
        layer.side_rows = np.cumsum([0] + [sum(w) for w in widths]).tolist()
        layer.q_widths, layer.k_widths, layer.v_widths = widths
        layer.wo = take(own("attn.wo_ud.weight"))
        layer.q_scale = take(own("attn.q_l2_clip.scale"))
        layer.rel_scale = take(own("attn.rel_l2_clip.scale"))
        layer.rel_proj = take(own("attn.rel_logits_proj.proj")).contiguous()
        layer.attn_conv = take(own("attn_sconv.weight"), fp32).view(-1, 4)
        layer.mlp_conv = take(own("mlp_sconv.weight"), fp32).view(-1, 4)
        layer.mlp_norm = take(own("mlp_norm.weight")).view(1, hidden)
        layer.attn_query = (take(f"attn_res.attn_queries.{index - 1}", fp32)
                            .view(-1, hidden) * scale if index else None)
        layer.mlp_query = (take(f"attn_res.mlp_queries.{index}", fp32)
                           .view(1, hidden) * scale)
        layer.router = take(own("mlp.gate.weight")).contiguous()
        layer.router_bias = take(own("mlp.gate.bias"), fp32)
        w13 = take(own("mlp.experts.w13_weight"))
        w2 = take(own("mlp.experts.w2_weight"))
        w13_scale = take(own("mlp.experts.w13_weight_scale"))
        w2_scale = take(own("mlp.experts.w2_weight_scale"))
        layer.nvfp4 = w13.dtype == torch.uint8
        if layer.nvfp4:
            layer.w13, layer.w2 = w13.contiguous(), w2.contiguous()
            layer.w13_scale = swizzle_blockscale(w13_scale).contiguous()
            layer.w2_scale = swizzle_blockscale(w2_scale).contiguous()
            for name in ("w13", "w2"):
                amax = take(own(f"mlp.experts.{name}_weight_amax"), fp32)
                denominator = take(
                    own(f"mlp.experts.{name}_weight_outer_scale_denominator"),
                    fp32)
                setattr(layer, f"{name}_alpha",
                        (amax / denominator).contiguous())
        else:
            layer.w13 = _dequantize_mxfp8(torch, w13, w13_scale)
            layer.w2 = _dequantize_mxfp8(torch, w2, w2_scale)
            del w13, w2
        layer.gates = None
        if start <= index < end:
            position = index - start
            message = {"q": queries[2 * position],
                       "k": kv_queries[2 * position],
                       "v": kv_queries[2 * position + 1],
                       "mlp": queries[2 * position + 1]}
            layer.gates = {}
            for name, controller in GATES:
                down = take(own(f"controllers.{controller}.down_weight"))
                forget = take(own(f"controllers.{controller}.forget_weight"))
                gain = take(own(f"controllers.{controller}.gain_weight"))
                layer.gates[name] = SimpleNamespace(
                    down=down.contiguous(),
                    gains=torch.cat([forget, gain]).contiguous(),
                    query=message[name].contiguous())
        layers.append(layer)
    embed = take("embed.weight")
    embed_norm = take("embed_norm.weight")
    final_norm = take("norm.weight").view(1, hidden)
    output_query = take("attn_res.output_query", fp32).view(1, hidden) * scale
    head = take("unembed.weight")[:config["unpadded_vocab_size"]].contiguous()
    handles.clear()
    return Inkling2Model(embed, embed_norm, layers, final_norm, output_query,
                         head, config)


def conv_taps(phases, rows):
    """Resolve every convolution tap of a chunk to a row, zero, or saved state.

    The checkpoint phases run in order and a phase reads the slots the
    ones before it wrote, so a slot written in this chunk holds the
    last three inputs of the sequence that wrote it. Following that
    through the phases maps each tap to one source, and one launch per
    convolution covers every phase.

    Args:
        phases: The chunk's checkpoint phases (gdn.plan_phases).
        rows: Rows in the chunk.

    Returns:
        (taps, write_slots, write_sources): (rows, 3) int32 sources of
        each row's three earlier inputs, oldest first; the slots this
        chunk leaves state in; and (slots, 3) int32 sources of each
        slot's new columns. A source is a row (>= 0), zero (-1), or
        column j of a slot's saved state (-(2 + 3 * slot + j)).
    """
    taps = np.full((rows, 3), kernels.ZERO_TAP, dtype=np.int32)
    current = {}
    window = np.arange(3)
    for phase in phases:
        if phase is None:
            continue
        reads = phase.host_read_slots
        writes = phase.host_write_slots
        pending = []
        first = 0
        for index, length in enumerate(phase.lengths):
            seq_rows = phase.host_rows[first:first + length].astype(np.int32)
            first += length
            if reads is None:
                history = np.full(3, kernels.ZERO_TAP, dtype=np.int32)
            else:
                slot = int(reads[index])
                history = current.get(slot)
                if history is None:
                    history = -(2 + 3 * slot + window).astype(np.int32)
            sources = np.concatenate([history, seq_rows])
            if phase.output:
                taps[seq_rows] = sources[np.arange(length)[:, None]
                                         + window[None, :]]
            if writes is not None:
                pending.append((int(writes[index]), sources[-3:].copy()))
        for slot, sources in pending:
            current[slot] = sources
    slots = np.fromiter(current, dtype=np.int64, count=len(current))
    sources = (np.stack([current[int(s)] for s in slots]) if len(slots)
               else np.zeros((0, 3), dtype=np.int32))
    return taps, slots, sources


class Inkling2Pipeline(ModelPipeline):
    """Forward passes for Inkling 2 Mini checkpoints loaded by load_inkling2.

    Arena layers and checkpoint state layers are the 48 visits in run
    order. Every visit's KV lives in the full pool; a sliding-window
    visit reads only its last 1,024 keys.
    """

    needs_pages = True
    tree_attention = False
    warm_row_classes = False
    warm_joins = False
    gemm_warmup = False

    def __init__(self, model, arena, *, spec, engine_class=Engine):
        import torch

        config = model.config
        visits = visit_layers(config)
        if (len(visits), config["hidden_size"], config["num_attention_heads"],
                config["num_key_value_heads"], config["head_dim"]) != (
                spec.layers, spec.hidden, spec.num_q_heads,
                spec.num_kv_heads, spec.d_head):
            raise ValueError(
                f"the loaded model's geometry (visits, hidden, q heads, kv "
                f"heads, head dim) differs from {spec.name}'s spec")
        if tuple(visit_kinds(config)) != tuple(spec.layer_kinds):
            raise ValueError("the loaded model's visit kinds differ from "
                             "the spec's")
        self.torch = torch
        self.spec = spec
        self.arena = arena
        self.model = model
        self.config = config
        self.visits = visits
        self.loop = config["loop_config"]
        self.hidden = config["hidden_size"]
        self.window = config["sliding_window_size"]
        self.eps = config["rms_norm_eps"]
        self.engine = engine_class(
            arena, num_q_heads=spec.num_q_heads,
            num_kv_heads=spec.num_kv_heads, head_dim=spec.d_head,
            rotary=None, fp8=False)
        self.max_chunk_tokens = (2**31 - 1) // (8 * 2 * self.hidden)
        # whether torch.mm takes out_dtype for bf16 inputs; probed once
        self._fp32_out = None

    # ---- small pieces ---------------------------------------------

    def _gate(self, gate, raw, messages, weight):
        """A loop message gate, then the RMS norm with weight, in bf16."""
        torch = self.torch
        prev_block, prev_out = messages
        inputs = kernels.gate_inputs(torch, raw, prev_block, prev_out,
                                     gate.query, self.eps)
        # the down output is rounded to bf16 and the gain weights are bf16
        # values, so a bf16 GEMM accumulating in fp32 forms the same products
        down = torch.mm(inputs, gate.down.t())
        gains = self._mm_fp32(down, gate.gains)
        return kernels.gate_apply(torch, raw, prev_block, prev_out,
                                  gate.query, gains, weight, self.eps)

    def _write_state(self, x, taps, state, part):
        """Save the new convolution columns of the slots this chunk writes."""
        slots, sources = taps["write_slots"], taps["write_sources"]
        if slots.numel() == 0:
            return
        first, last = STATE_CHANNELS[part]
        view = state[:, :, first:last]
        view[slots] = kernels.gather_state(self.torch, x, sources, view)

    def _mm_fp32(self, x, weight):
        """Return x @ weight.T for bf16 operands, accumulated in fp32.

        Products of two bf16 values are exact in fp32, so the fp32 GEMM
        fallback forms the same products.
        """
        torch = self.torch
        if self._fp32_out is None:
            try:
                torch.mm(x[:1], weight.t(), out_dtype=torch.float32)
                self._fp32_out = True
            except (TypeError, RuntimeError):
                self._fp32_out = False
        if self._fp32_out:
            return torch.mm(x, weight.t(), out_dtype=torch.float32)
        return torch.mm(x.float(), weight.float().t())

    def _router(self, layer, x):
        torch = self.torch
        logits = self._mm_fp32(x, layer.router)
        scores = logits.sigmoid()
        ids = torch.topk(scores + layer.router_bias,
                         self.config["num_experts_per_tok"], dim=-1,
                         sorted=False).indices
        weights = scores.gather(1, ids)
        weights = (weights / (weights.sum(-1, keepdim=True) + 1e-20)
                   * self.config["route_scale"])
        return ids.to(torch.int32), weights

    def _experts(self, layer, x):
        """The routed experts over bf16 rows, as bf16 rows."""
        from vllm import _custom_ops as ops

        torch = self.torch
        tokens = x.shape[0]
        if tokens == 0:
            return torch.empty_like(x)
        ids, weights = self._router(layer, x)
        experts = self.config["n_routed_experts"]
        topk = ids.shape[1]
        rows = tokens * topk
        hidden, inter = self.hidden, self.config["intermediate_size"]
        device = x.device
        offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
        sf_offsets = torch.empty(experts + 1, dtype=torch.int32, device=device)
        sizes1 = torch.empty((experts, 3), dtype=torch.int32, device=device)
        sizes2 = torch.empty((experts, 3), dtype=torch.int32, device=device)
        a_map = torch.empty(rows, dtype=torch.int32, device=device)
        c_map = torch.empty(rows, dtype=torch.int32, device=device)
        ops.get_cutlass_moe_mm_data(ids, offsets, sizes1, sizes2, a_map,
                                    c_map, experts, inter, hidden, sf_offsets)
        c_long = c_map.long()
        expert = torch.empty(rows, dtype=torch.int32, device=device)
        expert[c_long] = ids.flatten()
        gamma = torch.empty(rows, dtype=torch.float32, device=device)
        gamma[c_long] = weights.flatten()
        sf_rows = rows + 128 * experts
        caps = dict(gate_cap=self.config["mlp_gate_softcap"],
                    up_cap=self.config["mlp_up_softcap"])
        if layer.nvfp4:
            a_q, a_sf, a_scale = kernels.quantize_tokens(
                torch, x, c_map, ids, offsets, sf_offsets, nvfp4=True,
                sf_rows=sf_rows)
            gate_up = torch.empty((rows, 2 * inter), dtype=torch.bfloat16,
                                  device=device)
            ops.cutlass_fp4_moe_mm(gate_up, a_q, layer.w13, a_sf,
                                   layer.w13_scale, layer.w13_alpha, sizes1,
                                   offsets[:-1], sf_offsets[:-1])
            y_q, y_sf, y_scale = kernels.swiglu_quantize(
                torch, gate_up, a_scale, gamma, expert, offsets, sf_offsets,
                nvfp4=True, sf_rows=sf_rows, **caps)
            down = torch.empty((rows, hidden), dtype=torch.bfloat16,
                               device=device)
            ops.cutlass_fp4_moe_mm(down, y_q, layer.w2, y_sf, layer.w2_scale,
                                   layer.w2_alpha, sizes2, offsets[:-1],
                                   sf_offsets[:-1])
            return kernels.combine(torch, down, y_scale, c_map, tokens, topk)
        a, _, _ = kernels.quantize_tokens(torch, x, c_map, ids, offsets,
                                          sf_offsets, nvfp4=False,
                                          sf_rows=sf_rows)
        ends = offsets[1:]
        gate_up = torch._grouped_mm(a, layer.w13.transpose(1, 2), offs=ends)
        y, _, _ = kernels.swiglu_quantize(
            torch, gate_up, None, gamma, expert, offsets, sf_offsets,
            nvfp4=False, sf_rows=sf_rows, **caps)
        down = torch._grouped_mm(y, layer.w2.transpose(1, 2), offs=ends)
        return kernels.combine(torch, down, None, c_map, tokens, topk)

    # ---- one visit --------------------------------------------------

    def _attention_inputs(self, layer, h, blocks, num_blocks, messages):
        """The normed (q, k, v) inputs, one shared tensor when they agree."""
        torch = self.torch
        if layer.index == 0:
            return (kernels.rms_norm(torch, h, layer.attn_norm[0], self.eps),)
        if messages is None:
            mixed = kernels.mix(torch, blocks, num_blocks, h,
                                layer.attn_query, weight=layer.attn_norm,
                                eps=self.eps)
            return tuple(mixed)
        raw = kernels.mix(torch, blocks, num_blocks, h, layer.attn_query,
                          eps=self.eps)
        inputs = []
        for part, (name, _) in enumerate(GATES[:3]):
            source = raw[0] if layer.is_local else raw[part]
            weight = layer.attn_norm[0 if layer.is_local else part]
            inputs.append(self._gate(layer.gates[name], source, messages,
                                     weight))
        return tuple(inputs)

    def _projections(self, layer, inputs):
        """(q side, k side, v side) projection outputs."""
        torch = self.torch
        if len(inputs) == 1:
            out = torch.mm(inputs[0], layer.w_all.t())
            bounds = layer.side_rows
            return tuple(out[:, bounds[i]:bounds[i + 1]] for i in range(3))
        bounds = layer.side_rows
        return tuple(
            torch.mm(inputs[i], layer.w_all[bounds[i]:bounds[i + 1]].t())
            for i in range(3))

    def _visit(self, visit, layer, residual, blocks, num_blocks, chunk, plan,
               messages):
        torch, spec, config = self.torch, self.spec, self.config
        n = residual.shape[0]
        heads, kv_heads, dim = spec.num_q_heads, spec.num_kv_heads, spec.d_head
        rank = config["d_rel"]
        inputs = self._attention_inputs(layer, residual, blocks, num_blocks,
                                        messages)
        q_side, k_side, v_side = self._projections(layer, inputs)
        q_width, gate_width, rel_width, _ = layer.q_widths
        log_scale = (None if layer.is_local else
                     (config["log_scaling_alpha"],
                      float(config["log_scaling_n_floor"])))
        q = kernels.clip_heads(torch, q_side[:, :q_width], layer.q_scale,
                               chunk.positions, heads=heads, head_dim=dim,
                               clip=config["qk_l2_clip_threshold"],
                               log_scale=log_scale)
        rel_first = q_width + gate_width
        rel = kernels.clip_heads(
            torch, q_side[:, rel_first:rel_first + rel_width],
            layer.rel_scale, chunk.positions, heads=heads, head_dim=rank,
            clip=config["rel_l2_clip_threshold"], log_scale=log_scale)
        sink = q_side[:, rel_first + rel_width:]
        state = self.arena.conv_state[:, visit]
        kv_width = kv_heads * dim
        taps = plan["taps"]
        k_in, v_in = k_side[:, :kv_width], v_side[:, :kv_width]
        dynamic = not layer.is_local
        k3 = kernels.kv_conv(
            torch, k_in, taps, state[:, :, 0:1024], layer.k_conv,
            heads=kv_heads, head_dim=dim,
            dynamic=k_side[:, kv_width:] if dynamic else None, is_key=True,
            clip=config["qk_l2_clip_threshold"], eps=self.eps)
        v3 = kernels.kv_conv(
            torch, v_in, taps, state[:, :, 1024:2048], layer.v_conv,
            heads=kv_heads, head_dim=dim,
            dynamic=v_side[:, kv_width:] if dynamic else None, is_key=False,
            clip=0.0, eps=self.eps)
        self._write_state(k_in, plan, state, "k")
        self._write_state(v_in, plan, state, "v")
        meta = chunk.meta
        meta["layer"] = visit
        self.engine.write_kv(k3, v3, meta)
        kp, vp = self.arena.paged_kv(visit)
        unified = meta["unified"]
        attn = torch.empty((n, heads, dim), dtype=torch.bfloat16,
                           device=residual.device)
        kernels.rel_attention(
            torch, q, kp, vp, rel, sink, layer.rel_proj, attn,
            plan["tiles"], unified["cu_q"], unified["used"], unified["table"],
            window=self.window if layer.is_local else 0,
            block_rows=plan["block_rows"],
            num_warps=8 if plan["block_rows"] >= 64 else 4,
            num_stages=3 if plan["block_rows"] >= 64 else 2)
        gated = kernels.gate_attention(
            torch, attn.view(n, heads * dim), q_side[:, q_width:rel_first])
        projected = torch.mm(gated, layer.wo.t())
        attn_out = kernels.short_conv(torch, projected, taps,
                                      state[:, :, 2048:4352], layer.attn_conv)
        self._write_state(projected, plan, state, "attn")
        if layer.boundary:
            blocks[num_blocks].copy_(residual)
            num_blocks += 1
            partial, addend = attn_out, None
        else:
            partial, addend = residual, attn_out
        if messages is None:
            mlp_in = kernels.mix(torch, blocks, num_blocks, partial,
                                 layer.mlp_query, addend=addend,
                                 weight=layer.mlp_norm, eps=self.eps)[0]
        else:
            raw = kernels.mix(torch, blocks, num_blocks, partial,
                              layer.mlp_query, addend=addend, eps=self.eps)[0]
            mlp_in = self._gate(layer.gates["mlp"], raw, messages,
                                layer.mlp_norm[0])
        expert_out = self._experts(layer, mlp_in)
        if layer.boundary:
            out = kernels.short_conv(torch, expert_out, taps,
                                     state[:, :, 4352:6656], layer.mlp_conv,
                                     residual=attn_out)
        else:
            out = kernels.short_conv(torch, expert_out, taps,
                                     state[:, :, 4352:6656], layer.mlp_conv,
                                     residual=residual, addend=attn_out)
        self._write_state(expert_out, plan, state, "mlp")
        return out, num_blocks

    # ---- the chunk --------------------------------------------------

    def _plan(self, chunk):
        """The chunk's convolution taps and attention tiles, on the GPU."""
        torch = self.torch
        meta = chunk.meta
        if meta.get("gdn") is None or meta.get("unified") is None:
            raise ValueError("an Inkling chunk needs arena pages and "
                             "checkpoint phases")
        device = chunk.input_ids.device
        taps, slots, sources = conv_taps(meta["gdn"], chunk.tokens)
        unified = meta["unified"]
        group = self.spec.num_q_heads // self.spec.num_kv_heads
        # 16-row tiles for answer steps, 128-row tiles for prefill
        block_rows = 16 if unified["max_q"] <= 5 else 128
        seqs, firsts = kernels.attention_tiles(unified["cu_q_host"],
                                               block_rows // group)

        def gpu(array, dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=device, dtype=dtype)

        return dict(taps=gpu(taps, torch.int32),
                    write_slots=gpu(slots, torch.int64),
                    write_sources=gpu(sources, torch.int32),
                    tiles=(gpu(seqs, torch.int32), gpu(firsts, torch.int32)),
                    block_rows=block_rows)

    def forward_chunk(self, chunk):
        torch = self.torch
        model = self.model
        plan = self._plan(chunk)
        n = chunk.tokens
        h = kernels.rms_norm(
            torch, torch.nn.functional.embedding(chunk.input_ids, model.embed),
            model.embed_norm, self.eps)
        blocks = torch.empty((4, n, self.hidden), dtype=torch.bfloat16,
                             device=h.device)
        start, end = self.loop["core_start"], self.loop["core_end"]
        passes = self.loop["num_passes"]
        num_blocks = 0
        visit = 0
        for layer in model.layers[:start]:
            h, num_blocks = self._visit(visit, layer, h, blocks, num_blocks,
                                        chunk, plan, None)
            visit += 1
        before_h, before_blocks = h, num_blocks
        messages = None
        for loop_pass in range(passes):
            if loop_pass:
                # the next pass rewrites the blocks it reads its messages from
                messages = (blocks[before_blocks + 1].clone(), h)
                h, num_blocks = before_h, before_blocks
            for layer in model.layers[start:end]:
                h, num_blocks = self._visit(visit, layer, h, blocks,
                                            num_blocks, chunk, plan, messages)
                visit += 1
        for layer in model.layers[end:]:
            h, num_blocks = self._visit(visit, layer, h, blocks, num_blocks,
                                        chunk, plan, None)
            visit += 1
        final = chunk.final_indices
        return kernels.mix(torch, blocks.index_select(1, final), num_blocks,
                           h.index_select(0, final), model.output_query,
                           weight=model.final_norm, eps=self.eps)[0]
