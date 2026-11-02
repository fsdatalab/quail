"""A plain PyTorch LFM3 forward pass, the reference for Quail's LFM3 kernels.

It runs one pass at a time with ordinary operations: the convolution as
shifted sums, each norm on its own, rotary in float32, and attention
under an explicit mask. Under set attention the full layers let the
state rows read every row and each question read the state and itself;
the sliding layers are causal within their window. It rounds to bf16
where the checkpoint's own code does.
"""

import torch


def _norm(x, norm):
    scaled = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                     + norm.variance_epsilon)
    return scaled.to(torch.bfloat16) * norm.weight


def _rotate(x, cos, sin):
    first, second = x.float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    return torch.cat((first * cos - second * sin, second * cos + first * sin),
                     -1).to(torch.bfloat16)


def _conv(layer, x):
    b, c, gated = (x @ layer.in_proj.weight.T).chunk(3, dim=-1)
    inputs = (b * gated).float()
    width = layer.taps.shape[1]
    padded = torch.cat((inputs.new_zeros(width - 1, inputs.shape[1]), inputs))
    rows = inputs.shape[0]
    conv = sum(padded[tap:tap + rows] * layer.taps[:, tap].float()
               for tap in range(width))
    return (c * conv.to(torch.bfloat16)) @ layer.out_proj.weight.T


def _attention(layer, x, cos, sin, allowed, spec):
    rows = x.shape[0]
    heads, kv_heads, dim = spec.num_q_heads, spec.num_kv_heads, spec.d_head
    q, k, v, gate = (x @ layer.qkv_gate.weight.T).split(
        [heads * dim, kv_heads * dim, kv_heads * dim, heads], dim=-1)
    q = _rotate(_norm(q.reshape(rows, heads, dim), layer.q_norm), cos, sin)
    k = _rotate(_norm(k.reshape(rows, kv_heads, dim), layer.k_norm), cos, sin)
    v = v.reshape(rows, kv_heads, dim)
    repeat = heads // kv_heads
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) / dim ** 0.5
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    out = torch.einsum("hts,shd->thd", weights, v.float()).to(torch.bfloat16)
    out = out * torch.sigmoid(gate)[:, :, None]
    return out.reshape(rows, heads * dim) @ layer.o_proj.weight.T


def masks(groups, window):
    """The full layers' set mask and the sliding layers' causal window mask.

    Args:
        groups: Each row's question, numbered from 1, or -1 in the state.
        window: The sliding window, in rows.
    """
    group = torch.tensor(groups, device="cuda")
    rows = torch.arange(len(groups), device="cuda")
    state = group == -1
    full = state[:, None] | state[None, :] | (group[:, None] == group[None, :])
    if not (group > 0).any() or int(group.max()) == 1:
        full = torch.ones_like(full)
    behind = rows[:, None] - rows[None, :]
    sliding = (behind >= 0) & (behind < window)
    return full, sliding


@torch.inference_mode()
def reference_hidden(model, spec, ids, groups):
    """The final-normed hidden state of every row of one pass.

    Args:
        model: The loaded LFM3 model (load_lfm3), in bf16.
        spec: Its ModelSpec.
        ids: The pass's token ids.
        groups: Each row's question, numbered from 1, or -1 in the state.
    """
    full, sliding = masks(groups, spec.sliding_window)
    positions = torch.arange(len(ids), device="cuda")
    hidden = model.model.embed_tokens[torch.tensor(ids, device="cuda")]
    for layer in model.model.layers:
        x = _norm(hidden, layer.operator_norm)
        if layer.kind == "conv":
            y = _conv(layer, x)
        else:
            cos, sin = (table[positions] for table in model.rotary[layer.kind])
            allowed = full if layer.kind == "full_attention" else sliding
            y = _attention(layer, x, cos, sin, allowed, spec)
        hidden = hidden + _norm(y, layer.post_operator_norm)
        gate, up = (_norm(hidden, layer.ffn_norm) @ layer.w13.weight.T).chunk(2, -1)
        y = (torch.nn.functional.silu(gate) * up) @ layer.w2.weight.T
        hidden = hidden + _norm(y, layer.post_ffn_norm)
    return _norm(hidden, model.model.embedding_norm)
