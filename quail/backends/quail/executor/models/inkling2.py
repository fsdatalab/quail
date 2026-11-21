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
from quail.backends.quail.executor.cuda_graphs import PiecewiseBuffers
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

    Args:
        path: The local checkpoint directory.
        device: Where the weights go.

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
            gates = {}
            for name, controller in GATES:
                down = take(own(f"controllers.{controller}.down_weight"))
                forget = take(own(f"controllers.{controller}.forget_weight"))
                gain = take(own(f"controllers.{controller}.gain_weight"))
                gates[name] = (down, torch.cat([forget, gain]),
                               message[name])

            def stacked(names, gates=gates):
                # the gates of names, batched: (gates, ...) weights, already
                # transposed for x @ w
                return SimpleNamespace(
                    down_t=torch.stack([gates[n][0].t() for n in names]),
                    gains_t=torch.stack([gates[n][1].t() for n in names]),
                    query=torch.stack([gates[n][2] for n in names]))

            layer.gates = {"qkv": stacked(("q", "k", "v")),
                           "mlp": stacked(("mlp",))}
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


def visit_schedule(layers, loop):
    """Each visit in run order, and the completed blocks after the last.

    A visit names its layer, its index among the visits (its arena
    layer and state layer), the completed blocks before it, whether it
    reads loop messages, and whether it saves the state before the loop
    ("save") or restarts a pass from it ("restart").

    Returns:
        (visits, final completed blocks).
    """
    start, end = loop["core_start"], loop["core_end"]
    visits = []
    num_blocks = 0
    before = None

    def add(layer, *, looped=False, enter=None):
        nonlocal num_blocks
        visits.append(SimpleNamespace(index=len(visits), layer=layer,
                                      blocks=num_blocks, looped=looped,
                                      enter=enter))
        if layer.boundary:
            num_blocks += 1

    for layer in layers[:start]:
        add(layer)
    for loop_pass in range(loop["num_passes"]):
        for position, layer in enumerate(layers[start:end]):
            enter = None
            if position == 0:
                if loop_pass == 0:
                    before = num_blocks
                    enter = "save"
                else:
                    num_blocks = before
                    enter = "restart"
            add(layer, looped=loop_pass > 0, enter=enter)
    for layer in layers[end:]:
        add(layer)
    return visits, num_blocks


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
        self.schedule, self.final_blocks = visit_schedule(model.layers,
                                                          self.loop)
        # the piecewise graphs capture one piece per visit, plus the last
        self.layers = self.schedule
        self.final_norm = SimpleNamespace(weight=model.final_norm)

    # ---- small pieces ---------------------------------------------

    def _gates(self, gates, hidden, messages, weight):
        """Loop message gates, each then RMS-normed with its weight, in bf16.

        Args:
            gates: Batched gate weights (loader's stacked).
            hidden: (rows, width) shared by every gate, or (gates, rows,
                width).
            messages: (previous pass's last block, its output).
            weight: (gates, width) RMS norm weights.

        Returns:
            (gates, rows, width) gated inputs.
        """
        torch = self.torch
        prev_block, prev_out = messages
        inputs = kernels.gate_inputs(torch, hidden, prev_block, prev_out,
                                     gates.query, self.eps)
        # the down output is rounded to bf16 and the gain weights are bf16
        # values, so a bf16 GEMM accumulating in fp32 forms the same products
        down = torch.bmm(inputs, gates.down_t)
        gains = self._mm_fp32(down, gates.gains_t)
        return kernels.gate_apply(torch, hidden, prev_block, prev_out,
                                  gates.query, gains, weight, self.eps)

    def _save_state(self, x, run, state, part):
        """Save the new convolution columns of the slots this chunk writes."""
        first, last = STATE_CHANNELS[part]
        kernels.save_state(self.torch, x, run.write_slots, run.write_sources,
                           state[:, :, first:last])

    def _mm_fp32(self, x, weight):
        """Return x @ weight for bf16 operands, accumulated in fp32.

        A 2-D x multiplies a 2-D weight; a 3-D x a batch of weights.
        Products of two bf16 values are exact in fp32, so the fp32 GEMM
        fallback forms the same products.
        """
        torch = self.torch
        product = torch.mm if x.dim() == 2 else torch.bmm
        if self._fp32_out is None:
            try:
                product(x[..., :1, :], weight, out_dtype=torch.float32)
                self._fp32_out = True
            except (TypeError, RuntimeError):
                self._fp32_out = False
        if self._fp32_out:
            return product(x, weight, out_dtype=torch.float32)
        return product(x.float(), weight.float())

    def _router(self, layer, x):
        torch = self.torch
        logits = self._mm_fp32(x, layer.router.t())
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
        if layer.is_local:
            # one input feeds the query, key, and value gates
            hidden = raw[0]
            weight = layer.attn_norm.expand(3, -1).contiguous()
        else:
            hidden, weight = raw, layer.attn_norm
        return tuple(self._gates(layer.gates["qkv"], hidden, messages,
                                 weight))

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

    def _keep(self, run, name, value):
        """Set a tensor the next piece or token mixer reads.

        Captured pieces read and write fixed buffers, so a buffer is
        copied into; an eager pass keeps the tensor itself.
        """
        target = getattr(run, name, None)
        if isinstance(run, PiecewiseBuffers):
            target.copy_(value)
            return target
        setattr(run, name, value)
        return value

    def _pre(self, visit, run):
        """A visit's work up to attention: inputs, projections, keys, values.

        Leaves the queries, relative states, sink logits, and output gates
        for the token mixer and the next piece, and writes the visit's KV.
        """
        torch, spec, config = self.torch, self.spec, self.config
        layer = visit.layer
        heads, kv_heads, dim = spec.num_q_heads, spec.num_kv_heads, spec.d_head
        if visit.enter == "save":
            self._keep(run, "before", run.residual)
        elif visit.enter == "restart":
            # the next pass rewrites the block its messages read
            self._keep(run, "prev_block",
                       run.blocks[:, visit.blocks + 1].clone())
            self._keep(run, "prev_out", run.residual)
            self._keep(run, "residual", run.before)
        messages = (run.prev_block, run.prev_out) if visit.looped else None
        inputs = self._attention_inputs(layer, run.residual, run.blocks,
                                        visit.blocks, messages)
        q_side, k_side, v_side = self._projections(layer, inputs)
        q_width, gate_width, rel_width, _ = layer.q_widths
        rel_first = q_width + gate_width
        sink_first = rel_first + rel_width
        log_scale = (None if layer.is_local else
                     (config["log_scaling_alpha"],
                      float(config["log_scaling_n_floor"])))
        self._keep(run, "q3", kernels.clip_heads(
            torch, q_side[:, :q_width], layer.q_scale, run.positions,
            heads=heads, head_dim=dim, clip=config["qk_l2_clip_threshold"],
            log_scale=log_scale))
        self._keep(run, "rel", kernels.clip_heads(
            torch, q_side[:, rel_first:sink_first], layer.rel_scale,
            run.positions, heads=heads, head_dim=config["d_rel"],
            clip=config["rel_l2_clip_threshold"], log_scale=log_scale))
        self._keep(run, "sink", q_side[:, sink_first:sink_first + heads])
        self._keep(run, "gate", q_side[:, q_width:rel_first])
        state = self.arena.conv_state[:, visit.index]
        kv_width = kv_heads * dim
        k_in, v_in = k_side[:, :kv_width], v_side[:, :kv_width]
        dynamic = not layer.is_local
        k3 = kernels.kv_conv(
            torch, k_in, run.taps, state[:, :, 0:1024], layer.k_conv,
            heads=kv_heads, head_dim=dim,
            dynamic=k_side[:, kv_width:] if dynamic else None, is_key=True,
            clip=config["qk_l2_clip_threshold"], eps=self.eps)
        v3 = kernels.kv_conv(
            torch, v_in, run.taps, state[:, :, 1024:2048], layer.v_conv,
            heads=kv_heads, head_dim=dim,
            dynamic=v_side[:, kv_width:] if dynamic else None, is_key=False,
            clip=0.0, eps=self.eps)
        self._save_state(k_in, run, state, "k")
        self._save_state(v_in, run, state, "v")
        run.write_kv(self.engine, k3, v3, visit.index)

    def _post(self, visit, run):
        """A visit's work after attention: output, experts, residual stream."""
        torch, spec = self.torch, self.spec
        layer = visit.layer
        n = run.residual.shape[0]
        width = spec.num_q_heads * spec.d_head
        state = self.arena.conv_state[:, visit.index]
        gated = kernels.gate_attention(torch, run.attn.view(n, width),
                                       run.gate)
        projected = torch.mm(gated, layer.wo.t())
        attn_out = kernels.short_conv(torch, projected, run.taps,
                                      state[:, :, 2048:4352], layer.attn_conv)
        self._save_state(projected, run, state, "attn")
        num_blocks = visit.blocks
        if layer.boundary:
            run.blocks[:, num_blocks].copy_(run.residual)
            num_blocks += 1
            partial, addend = attn_out, None
        else:
            partial, addend = run.residual, attn_out
        if visit.looped:
            raw = kernels.mix(torch, run.blocks, num_blocks, partial,
                              layer.mlp_query, addend=addend, eps=self.eps)[0]
            mlp_in = self._gates(layer.gates["mlp"], raw,
                                 (run.prev_block, run.prev_out),
                                 layer.mlp_norm)[0]
        else:
            mlp_in = kernels.mix(torch, run.blocks, num_blocks, partial,
                                 layer.mlp_query, addend=addend,
                                 weight=layer.mlp_norm, eps=self.eps)[0]
        expert_out = self._experts(layer, mlp_in)
        if layer.boundary:
            out = kernels.short_conv(torch, expert_out, run.taps,
                                     state[:, :, 4352:6656], layer.mlp_conv,
                                     residual=attn_out)
        else:
            out = kernels.short_conv(torch, expert_out, run.taps,
                                     state[:, :, 4352:6656], layer.mlp_conv,
                                     residual=run.residual, addend=attn_out)
        self._save_state(expert_out, run, state, "mlp")
        self._keep(run, "residual", out)

    # ---- pieces between token mixers ----------------------------------

    def boundary(self, num_tokens, device):
        """The tensors pieces and token mixers pass to each other, by row."""
        torch, spec = self.torch, self.spec
        heads, dim = spec.num_q_heads, spec.d_head

        def rows(*shape, dtype=torch.bfloat16, fill=None):
            tensor = torch.zeros((num_tokens, *shape), dtype=dtype,
                                 device=device)
            return tensor if fill is None else tensor.fill_(fill)

        return dict(
            blocks=rows(4, self.hidden), before=rows(self.hidden),
            prev_block=rows(self.hidden), prev_out=rows(self.hidden),
            q3=rows(heads, dim), rel=rows(heads, self.config["d_rel"]),
            sink=rows(heads), gate=rows(heads * dim), attn=rows(heads, dim),
            # padding rows read no taps and write no state
            taps=rows(3, dtype=torch.int32, fill=kernels.ZERO_TAP),
            write_slots=rows(dtype=torch.int64, fill=-1),
            write_sources=rows(3, dtype=torch.int32, fill=kernels.ZERO_TAP))

    def piece(self, index, buffers):
        """The previous visit's work after attention, then the next's up to it."""
        torch, model = self.torch, self.model
        if index == 0:
            embedded = torch.nn.functional.embedding(buffers.ids, model.embed)
            self._keep(buffers, "residual", kernels.rms_norm(
                torch, embedded, model.embed_norm, self.eps))
        else:
            self._post(self.schedule[index - 1], buffers)
        if index == len(self.schedule):
            blocks, residual = buffers.blocks, buffers.residual
            final = getattr(buffers, "final", None)
            if final is not None:
                blocks = blocks.index_select(0, final)
                residual = residual.index_select(0, final)
            self._keep(buffers, "output", kernels.mix(
                torch, blocks, self.final_blocks, residual,
                model.output_query, weight=model.final_norm,
                eps=self.eps)[0])
            return
        self._pre(self.schedule[index], buffers)

    def mixer(self, index, buffers, chunk):
        """Visit index's attention over the chunk's real rows."""
        torch = self.torch
        layer = self.schedule[index].layer
        plan = chunk.meta["inkling_plan"]
        unified = chunk.meta["unified"]
        n = chunk.tokens
        kp, vp = self.arena.paged_kv(index)
        if not isinstance(buffers, PiecewiseBuffers):
            buffers.attn = torch.empty_like(buffers.q3)
        prefill = plan["block_rows"] >= 64
        kernels.rel_attention(
            torch, buffers.q3[:n], kp, vp, buffers.rel[:n], buffers.sink[:n],
            layer.rel_proj, buffers.attn[:n], plan["tiles"], unified["cu_q"],
            unified["used"], unified["table"],
            window=self.window if layer.is_local else 0,
            max_keys=unified["max_used"], block_rows=plan["block_rows"],
            block_keys=64 if prefill else 32,
            num_warps=8 if prefill else 4, num_stages=3 if prefill else 2)

    def stage(self, buffers, chunk):
        """Copy the chunk's convolution taps and state writes into buffers.

        A None chunk leaves every row reading zero taps and writing no
        state.
        """
        n = writes = 0
        if chunk is not None:
            plan = self._plan(chunk)
            n, writes = chunk.tokens, plan["write_slots"].shape[0]
            buffers.taps[:n].copy_(plan["taps"])
            buffers.write_slots[:writes].copy_(plan["write_slots"])
            buffers.write_sources[:writes].copy_(plan["write_sources"])
        # a padding write's sources are still read before its slot masks it
        buffers.taps[n:].fill_(kernels.ZERO_TAP)
        buffers.write_slots[writes:].fill_(-1)
        buffers.write_sources[writes:].fill_(kernels.ZERO_TAP)

    # ---- the chunk --------------------------------------------------

    def _plan(self, chunk):
        """The chunk's convolution taps, state writes, and attention tiles.

        Also kept in chunk.meta for the token mixers.
        """
        torch = self.torch
        meta = chunk.meta
        if meta.get("gdn") is None or meta.get("unified") is None:
            raise ValueError("an Inkling chunk needs arena pages and "
                             "checkpoint phases")
        device = chunk.input_ids.device
        taps, slots, sources = conv_taps(meta["gdn"], chunk.tokens)
        unified = meta["unified"]
        group = self.spec.num_q_heads // self.spec.num_kv_heads
        # 16-row tiles for answer steps, 64-row tiles for prefill
        block_rows = 16 if unified["max_q"] <= 5 else 64
        seqs, firsts = kernels.attention_tiles(unified["cu_q_host"],
                                               block_rows // group)

        def gpu(array, dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(
                device=device, dtype=dtype)

        plan = dict(taps=gpu(taps, torch.int32),
                    write_slots=gpu(slots, torch.int64),
                    write_sources=gpu(sources, torch.int32),
                    tiles=(gpu(seqs, torch.int32), gpu(firsts, torch.int32)),
                    block_rows=block_rows)
        meta["inkling_plan"] = plan
        return plan

    def forward_chunk(self, chunk):
        """Run the chunk eagerly through the same pieces and token mixers."""
        plan = self._plan(chunk)
        meta = chunk.meta
        n = chunk.tokens

        def write_kv(engine, k, v, layer):
            meta["layer"] = layer
            engine.write_kv(k, v, meta)

        run = SimpleNamespace(
            ids=chunk.input_ids, positions=chunk.positions,
            blocks=self.torch.empty((n, 4, self.hidden),
                                    dtype=self.torch.bfloat16,
                                    device=chunk.input_ids.device),
            taps=plan["taps"], write_slots=plan["write_slots"],
            write_sources=plan["write_sources"], final=chunk.final_indices,
            write_kv=write_kv)
        for index in range(len(self.schedule) + 1):
            self.piece(index, run)
            if index < len(self.schedule):
                self.mixer(index, run, chunk)
        return run.output
