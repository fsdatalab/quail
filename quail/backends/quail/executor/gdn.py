"""Gated DeltaNet layers over packed rows, with per-sequence checkpoints.

A chunk's rows form sequences: a document's fresh prefix, and each
question that follows a document. A prefix starts from zero state and
leaves a checkpoint (the recurrent matrix and the convolution columns)
in its document's slot. A question starts from that checkpoint and
leaves nothing, so any number of questions read one document without
rewinding.

Each phase below runs every sequence of one kind in one pass over its
rows. Prefixes run first: a question's start state is read after they
have written their slots.

The reference functions are plain torch and define the maths. The
recurrence can also run vLLM's chunked kernel on the same tensors; the
convolution always runs the reference.
"""

from dataclasses import dataclass

import numpy as np

L2_EPS = 1e-6


@dataclass
class GdnPhase:
    """Sequences of one kind inside a chunk.

    Attributes:
        rows: Chunk rows of every sequence, back to back, on the GPU.
        cu_seqlens: Sequence bounds over rows, int32, on the GPU.
        lengths: Rows per sequence, on the host.
        read_slots: Checkpoint slot each sequence starts from, or None
            when every sequence starts from zero state.
        write_slots: Checkpoint slot each sequence leaves its final
            state in, or None when none is kept.
    """

    rows: object
    cu_seqlens: object
    lengths: tuple
    read_slots: object = None
    write_slots: object = None


def plan_phases(arena, groups, stage):
    """Lay a chunk's groups out as the two phases of every Gated DeltaNet layer.

    Allocates a checkpoint slot for each fresh document and records the
    length its checkpoint covers. A kept document must be asked at
    exactly that length, since its state cannot rewind.

    Args:
        arena: KVArena with checkpoint slots.
        groups: One mapping per group in chunk order, with key, fresh,
            f, start, row0, prefix_end, keeps, and suffix_spans (the
            begin and end rows of each suffix).
        stage: Callable (name, values, dtype) that copies host values to
            the GPU.

    Returns:
        (prefix phase, suffix phase); either is None when it has no
        sequences.

    Raises:
        ValueError: A group cannot run on a state that does not rewind.
        ArenaFullError: Every checkpoint slot belongs to a pinned key.
    """
    import torch

    from quail.backends.quail.executor.chunk import ArenaFullError

    prefix_rows, prefix_lengths, prefix_slots = [], [], []
    suffix_rows, suffix_lengths, suffix_slots = [], [], []
    for group in groups:
        key = group["key"]
        if group["keeps"]:
            raise ValueError(
                f"group {key!r}: a hybrid model keeps no suffix KV, since "
                f"its state cannot rewind past the document")
        if group["fresh"]:
            if group["start"]:
                raise ValueError(
                    f"group {key!r}: a hybrid model cannot borrow a prefix, "
                    f"which needs the state at the shared point")
            slot = arena.checkpoint_slot(key)
            if slot is None:
                raise ArenaFullError(
                    f"no free checkpoint slot for {key!r}: all "
                    f"{arena.n_state_slots} belong to documents in use")
            arena.record_checkpoint(key, group["f"])
            first, last = group["row0"], group["prefix_end"]
            if last > first:
                prefix_rows.append(np.arange(first, last, dtype=np.int64))
                prefix_lengths.append(last - first)
                prefix_slots.append(slot)
        else:
            slot = arena.accounting.slots.get(key)
            if slot is None or arena.checkpoint_tokens(key) != group["f"]:
                raise ValueError(
                    f"group {key!r}: asked at {group['f']} tokens, but its "
                    f"checkpoint covers {arena.checkpoint_tokens(key)}")
        begins, ends = group["suffix_spans"]
        for begin, end in zip(begins.tolist(), ends.tolist()):
            suffix_rows.append(np.arange(begin, end, dtype=np.int64))
            suffix_lengths.append(end - begin)
            suffix_slots.append(arena.accounting.slots[key])

    def phase(name, rows, lengths, slots, *, read):
        if not lengths:
            return None
        cu = np.concatenate([[0], np.cumsum(lengths)])
        slot_tensor = stage(f"gdn_{name}_slots", slots, torch.int64)
        return GdnPhase(
            rows=stage(f"gdn_{name}_rows", np.concatenate(rows), torch.int64),
            cu_seqlens=stage(f"gdn_{name}_cu", cu, torch.int32),
            lengths=tuple(lengths),
            read_slots=slot_tensor if read else None,
            write_slots=None if read else slot_tensor)

    return (phase("prefix", prefix_rows, prefix_lengths, prefix_slots,
                  read=False),
            phase("suffix", suffix_rows, suffix_lengths, suffix_slots,
                  read=True))


# ---- reference maths --------------------------------------------------

def l2_normalize(x):
    """Scale the last dimension of x to unit length."""
    import torch

    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + L2_EPS)


def causal_conv_reference(x, weight, lengths, history=None, bias=None):
    """Depthwise causal convolution then SiLU, per sequence of packed rows.

    Args:
        x: (rows, channels) inputs of the sequences back to back.
        weight: (channels, width) filter taps; the last tap is the
            current row.
        lengths: Rows per sequence.
        history: (sequences, width - 1, channels) input columns before
            each sequence, oldest first; None means zeros.
        bias: Optional (channels,) bias.

    Returns:
        (output with x's shape, tails), where tails is
        (sequences, width - 1, channels): each sequence's last width - 1
        input columns, which the next row would read as its history.
    """
    import torch
    import torch.nn.functional as F

    rows, channels = x.shape
    width = weight.shape[1]
    lengths_t = torch.as_tensor(list(lengths), device=x.device)
    starts = torch.cumsum(lengths_t, 0) - lengths_t
    sequence = torch.repeat_interleave(
        torch.arange(len(lengths), device=x.device), lengths_t)
    offset = torch.arange(rows, device=x.device) - starts[sequence]
    out = torch.zeros((rows, channels), dtype=torch.float32, device=x.device)
    for back in range(width):
        tap = weight[:, width - 1 - back].float()
        source = (torch.arange(rows, device=x.device) - back).clamp(min=0)
        column = x[source].float()
        before = offset < back
        if history is None:
            column = torch.where(before[:, None], 0.0, column)
        elif before.any():
            past = history[sequence[before], width - 1 - back + offset[before]]
            column[before] = past.float()
        out += column * tap
    if bias is not None:
        out += bias.float()
    tails = _conv_tails(x, lengths_t, starts, history, width)
    return F.silu(out).to(x.dtype), tails


def _conv_tails(x, lengths, starts, history, width):
    """Each sequence's last width - 1 input columns, oldest first."""
    import torch

    count = lengths.shape[0]
    index = lengths[:, None] + torch.arange(width - 1, device=x.device)
    in_history = index < width - 1
    row = (starts[:, None] + index - (width - 1)).clamp(min=0)
    tails = x[row.clamp(max=x.shape[0] - 1)]
    pad = (history if history is not None
           else torch.zeros((count, width - 1, x.shape[1]), dtype=x.dtype,
                            device=x.device))
    old = pad.gather(1, index.clamp(max=width - 2)[:, :, None].expand(
        -1, -1, x.shape[1]))
    return torch.where(in_history[:, :, None], old.to(x.dtype), tails)


def gated_delta_reference(q, k, v, g, beta, state, lengths, scale):
    """Gated delta rule over packed sequences, one token at a time.

    Per token: S <- exp(g) * S (I - beta k k^T) + beta v k^T, and the
    output is S q. Queries and keys are L2-normalized and the query is
    multiplied by scale first.

    Args:
        q: (rows, key heads, key dim).
        k: (rows, key heads, key dim).
        v: (rows, value heads, value dim).
        g: (rows, value heads) log of the decay.
        beta: (rows, value heads) write strength.
        state: (sequences, value heads, value dim, key dim), float32;
            updated in place to each sequence's final state.
        lengths: Rows per sequence.
        scale: Query multiplier after normalization.

    Returns:
        (rows, value heads, value dim) outputs, in v's dtype.
    """
    import torch

    value_heads = v.shape[1]
    repeat = value_heads // q.shape[1]
    q = l2_normalize(q).repeat_interleave(repeat, dim=1) * scale
    k = l2_normalize(k).repeat_interleave(repeat, dim=1)
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    decay = g.float().exp()
    row = 0
    for index, length in enumerate(lengths):
        s = state[index]
        for t in range(row, row + length):
            s = s * decay[t][:, None, None]
            predicted = (s * k[t][:, None, :]).sum(-1)
            delta = (v[t].float() - predicted) * beta[t].float()[:, None]
            s = s + delta[:, :, None] * k[t][:, None, :]
            out[t] = (s * q[t][:, None, :]).sum(-1)
        state[index] = s
        row += length
    return out.to(v.dtype)


# ---- kernel path ------------------------------------------------------

def gated_delta_kernel(q, k, v, g, beta, state, cu_seqlens, scale):
    """The same rule through vLLM's chunked kernel; state is updated in place.

    The kernel normalizes q and k itself and takes key heads fewer than
    value heads without repeating them.
    """
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

    out, final = chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=state, output_final_state=True,
        cu_seqlens=cu_seqlens.long(), scale=scale,
        use_qk_l2norm_in_kernel=True)
    state.copy_(final)
    return out[0]


# ---- the layer --------------------------------------------------------

def gated_rms_norm(x, gate, weight, eps):
    """RMS-normalize the last dimension of x, scale by weight, gate by SiLU."""
    import torch
    import torch.nn.functional as F

    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                     + eps)
    scaled = weight * normed.to(x.dtype)
    return (scaled.float() * F.silu(gate.float())).to(x.dtype)


def gdn_core(geometry, tensors, phases, arena, layer_slot, *, use_kernels):
    """Run one Gated DeltaNet layer's convolution and recurrence.

    Args:
        geometry: The model spec, for its head counts and dimensions.
        tensors: Dict with qkv (rows, conv channels), b, a, conv_weight,
            conv_bias, A_log, and dt_bias.
        phases: (prefix phase, suffix phase) from plan_phases.
        arena: KVArena holding the checkpoint pools.
        layer_slot: Index of this layer among the Gated DeltaNet layers.
        use_kernels: Whether the recurrence runs vLLM's kernel.

    Returns:
        (rows, value heads, value dim) recurrence outputs.
    """
    import torch
    import torch.nn.functional as F

    spec = geometry
    qkv = tensors["qkv"]
    rows = qkv.shape[0]
    out = torch.zeros((rows, spec.gdn_value_heads, spec.gdn_value_dim),
                      dtype=qkv.dtype, device=qkv.device)
    key_width = spec.gdn_key_heads * spec.gdn_key_dim
    beta = tensors["b"].sigmoid()
    gate = (-tensors["A_log"].float().exp()
            * F.softplus(tensors["a"].float() + tensors["dt_bias"].float()))
    scale = spec.gdn_key_dim ** -0.5
    for phase in phases:
        if phase is None:
            continue
        mixed = qkv.index_select(0, phase.rows)
        history = (None if phase.read_slots is None
                   else arena.conv_state[phase.read_slots, layer_slot])
        mixed, tails = causal_conv_reference(
            mixed, tensors["conv_weight"], phase.lengths, history,
            tensors["conv_bias"])
        if phase.write_slots is not None:
            arena.conv_state[phase.write_slots, layer_slot] = tails
        q, k, v = mixed.split([key_width, key_width, mixed.shape[1]
                               - 2 * key_width], dim=-1)
        q = q.reshape(-1, spec.gdn_key_heads, spec.gdn_key_dim)
        k = k.reshape(-1, spec.gdn_key_heads, spec.gdn_key_dim)
        v = v.reshape(-1, spec.gdn_value_heads, spec.gdn_value_dim)
        g = gate.index_select(0, phase.rows)
        b = beta.index_select(0, phase.rows)
        if phase.read_slots is None:
            state = torch.zeros(
                (len(phase.lengths), spec.gdn_value_heads,
                 spec.gdn_value_dim, spec.gdn_key_dim),
                dtype=torch.float32, device=qkv.device)
        else:
            state = arena.gdn_state[phase.read_slots, layer_slot]
        if use_kernels:
            core = gated_delta_kernel(q, k, v, g, b, state,
                                      phase.cu_seqlens, scale)
        else:
            core = gated_delta_reference(q, k, v, g, b, state,
                                         phase.lengths, scale)
        if phase.write_slots is not None:
            arena.gdn_state[phase.write_slots, layer_slot] = state
        out.index_copy_(0, phase.rows, core)
    return out
