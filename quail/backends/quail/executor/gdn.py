"""Gated DeltaNet layers over packed rows, with per-sequence checkpoints.

A chunk's rows form sequences: a document's fresh prefix, and each
question that follows a document. A prefix starts from zero state and
leaves a checkpoint (the recurrent matrix and the convolution columns)
in its document's slot. A question starts from that checkpoint and
leaves nothing, so any number of questions read one document without
rewinding. A chain's retained tokens, such as a stage frame, advance
the checkpoint in place, so the next stage asks at the longer length.

Each phase runs every sequence of one kind in one pass over its rows,
in the order plan_phases documents: a phase reads the slots the ones
before it wrote.

The reference functions are plain torch and define the maths. The
convolution and the recurrence can also run vLLM's fused kernels on the
same tensors.
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
        output: Whether the sequences' outputs feed the next layer;
            False for rows another phase already produced.
    """

    rows: object
    cu_seqlens: object
    lengths: tuple
    read_slots: object = None
    write_slots: object = None
    output: bool = True


def plan_phases(arena, groups, stage):
    """Lay a chunk's groups out as the phases of every Gated DeltaNet layer.

    Each layer runs the phases in this order, and a phase reads the
    slots the ones before it wrote:

    1. prefix: a fresh document's rows from zero state, writing its slot.
    2. continuation: the retained tokens of a group whose whole first
       suffix is kept (a stage frame), from its slot back into it. A
       later group of the same key in the chunk reads the extended slot.
    3. suffix: every question from its document's slot, writing nothing.
    4. extension: the leading retained tokens of a suffix that also
       holds a question, from the slot back into it. The suffix ran
       from the state before them, so the slot advances after it.

    Allocates a checkpoint slot for each fresh document and records the
    tokens it covers. Retained suffix tokens advance a second slot, so
    the document-end checkpoint survives for an operator that asks at
    the document's length after the KV rewinds. A group must be asked
    at one of the two lengths exactly, since a state cannot rewind.

    Args:
        arena: KVArena with checkpoint slots.
        groups: One mapping per group in chunk order, with key, fresh,
            f, start, row0, prefix_end, keeps, kept, and suffix_spans
            (the begin and end rows of each suffix).
        stage: Callable (name, values, dtype) that copies host values to
            the GPU.

    Returns:
        (prefix, continuation, suffix, extension) phases; each is None
        when it has no sequences.

    Raises:
        ValueError: A group borrows a prefix, keeps more tokens than its
            first suffix holds, or follows a group of its key that
            extends the checkpoint after its suffix.
        ArenaFullError: Every checkpoint slot belongs to a pinned key.
    """
    import torch

    from quail.backends.quail.executor.chunk import ArenaFullError

    lists = {name: ([], [], []) for name in
             ("prefix", "continuation", "suffix", "extension")}

    def add(name, rows, slot):
        rows_list, lengths, slots = lists[name]
        rows_list.append(rows)
        lengths.append(len(rows))
        slots.append(slot)

    extended_after = set()
    for group in groups:
        key = group["key"]
        kept = group.get("kept", 0)
        if group["start"]:
            raise ValueError(
                f"group {key!r}: a hybrid model cannot borrow a prefix, "
                f"which needs the state at the shared point")
        begins, ends = (spans.tolist() for spans in group["suffix_spans"])
        if kept and (not begins or kept > ends[0] - begins[0]):
            raise ValueError(
                f"group {key!r}: keeps {kept} suffix tokens, more than "
                f"its first suffix holds")
        if group["fresh"]:
            source = arena.checkpoint_slot(key)
            if source is None:
                raise ArenaFullError(
                    f"no free checkpoint slot for {key!r}: all "
                    f"{arena.n_state_slots} belong to documents in use")
            first, last = group["row0"], group["prefix_end"]
            if last > first:
                add("prefix", np.arange(first, last, dtype=np.int64), source)
            arena.record_checkpoint(key, group["f"])
        else:
            source = None
            if arena.checkpoint_tokens(key) == group["f"]:
                source = arena.accounting.slots.get(key)
            else:
                extension = arena.extension(key)
                if extension is not None and extension[1] == group["f"]:
                    source = extension[0]
            if source is None:
                extension = arena.extension(key)
                covered = [arena.checkpoint_tokens(key)]
                if extension is not None:
                    covered.append(extension[1])
                raise ValueError(
                    f"group {key!r}: asked at {group['f']} tokens, but its "
                    f"checkpoints cover {covered}")
            if key in extended_after:
                raise ValueError(
                    f"group {key!r}: follows a group of its key whose "
                    f"suffix runs before its checkpoint extends")
        pure = bool(kept) and len(begins) == 1 and kept == ends[0] - begins[0]
        target = source
        if kept:
            # the document-end checkpoint stays for a later operator
            # that asks at the document's length; the retained suffix
            # advances a second slot
            held = arena.extension(key)
            if held is not None and held[0] == source:
                target = source
            else:
                target = arena.extension_slot(key)
                if target is None:
                    raise ArenaFullError(
                        f"no free checkpoint slot to extend {key!r}")
            arena.record_extension(key, group["f"] + kept)
            head = np.arange(begins[0], begins[0] + kept, dtype=np.int64)
            name = "continuation" if pure else "extension"
            lists[name][0].append(head)
            lists[name][1].append(len(head))
            lists[name][2].append((source, target))
            if not pure:
                extended_after.add(key)
        if pure:
            continue
        for begin, end in zip(begins, ends):
            add("suffix", np.arange(begin, end, dtype=np.int64), source)

    def phase(name, *, read, write, output=True):
        rows, lengths, slots = lists[name]
        if not lengths:
            return None
        cu = np.concatenate([[0], np.cumsum(lengths)])
        if name in ("continuation", "extension"):
            reads = [pair[0] for pair in slots]
            writes = [pair[1] for pair in slots]
        else:
            reads = writes = slots
        return GdnPhase(
            rows=stage(f"gdn_{name}_rows", np.concatenate(rows), torch.int64),
            cu_seqlens=stage(f"gdn_{name}_cu", cu, torch.int32),
            lengths=tuple(lengths),
            read_slots=(stage(f"gdn_{name}_read", reads, torch.int64)
                        if read else None),
            write_slots=(stage(f"gdn_{name}_write", writes, torch.int64)
                         if write else None),
            output=output)

    return (phase("prefix", read=False, write=True),
            phase("continuation", read=True, write=True),
            phase("suffix", read=True, write=False),
            phase("extension", read=True, write=True, output=False))


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
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule,
    )

    out, final = chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=state, output_final_state=True,
        cu_seqlens=cu_seqlens.long(), scale=scale,
        use_qk_l2norm_in_kernel=True)
    state.copy_(final)
    return out[0]


_CONV_KERNEL = None


def _conv_triton_kernel():
    """Build the convolution kernel once; triton is imported on first use."""
    global _CONV_KERNEL
    if _CONV_KERNEL is not None:
        return _CONV_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def conv_silu(x_ptr, w_ptr, bias_ptr, hist_ptr, seq_ptr, within_ptr,
                  out_ptr, channels, stride_x, stride_hs, stride_ht,
                  stride_o, W: tl.constexpr, BLOCK: tl.constexpr,
                  HAS_HIST: tl.constexpr, HAS_BIAS: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < channels
        sequence = tl.load(seq_ptr + row)
        within = tl.load(within_ptr + row)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for back in tl.static_range(W):
            tap = tl.load(w_ptr + cols * W + (W - 1 - back), mask=mask,
                          other=0.0).to(tl.float32)
            inside = (within - back) >= 0
            value = tl.load(x_ptr + (row - back) * stride_x + cols,
                            mask=mask & inside, other=0.0).to(tl.float32)
            if HAS_HIST:
                past = tl.load(
                    hist_ptr + sequence * stride_hs
                    + (W - 1 - back + within) * stride_ht + cols,
                    mask=mask & (within - back < 0), other=0.0
                ).to(tl.float32)
                value = value + past
            acc += value * tap
        if HAS_BIAS:
            acc += tl.load(bias_ptr + cols, mask=mask,
                           other=0.0).to(tl.float32)
        out = acc * tl.sigmoid(acc)
        tl.store(out_ptr + row * stride_o + cols,
                 out.to(out_ptr.dtype.element_ty), mask=mask)

    _CONV_KERNEL = conv_silu
    return _CONV_KERNEL


def causal_conv_kernel(x, weight, phase, history, bias=None):
    """The same convolution as one Triton kernel over the packed rows.

    Args:
        x: (rows, channels) inputs of the phase's sequences back to back.
        weight: (channels, width) filter taps; the last tap is the
            current row.
        phase: The GdnPhase whose lengths bound the sequences.
        history: (sequences, width - 1, channels) input columns before
            each sequence, or None for zeros.
        bias: Optional (channels,) bias.

    Returns:
        (output with x's shape, tails), as causal_conv_reference does.
    """
    import torch

    rows, channels = x.shape
    width = weight.shape[1]
    device = x.device
    lengths = torch.as_tensor(phase.lengths, device=device)
    starts = torch.cumsum(lengths, 0) - lengths
    sequence = torch.repeat_interleave(
        torch.arange(len(phase.lengths), device=device), lengths)
    within = (torch.arange(rows, device=device) - starts[sequence]).int()
    out = torch.empty_like(x)
    x = x.contiguous()
    weight = weight.contiguous()
    block = 1024
    has_history = history is not None
    hist = history if has_history else x
    _conv_triton_kernel()[(rows, triton_cdiv(channels, block))](
        x, weight, bias if bias is not None else x, hist,
        sequence.int(), within, out, channels, x.stride(0),
        hist.stride(0) if has_history else 0,
        hist.stride(1) if has_history else 0, out.stride(0),
        W=width, BLOCK=block, HAS_HIST=has_history,
        HAS_BIAS=bias is not None, num_warps=4)
    tails = _conv_tails(x, lengths, starts, history, width)
    return out, tails


def triton_cdiv(value, block):
    """Blocks of `block` covering `value` elements."""
    return -(-value // block)


# a CUDA launch has at most this many blocks along its second grid axis,
# and the kernel spends one per sequence and value head
GRID_LIMIT = 65_535


def _kernel_in_batches(spec, q, k, v, g, beta, state, phase, scale):
    """Run the chunked kernel over as many sequences as one launch holds.

    Each batch updates its slice of `state` in place.
    """
    import torch

    count = len(phase.lengths)
    per_launch = max(1, GRID_LIMIT // spec.gdn_value_heads)
    if count <= per_launch:
        return gated_delta_kernel(q, k, v, g, beta, state,
                                  phase.cu_seqlens, scale)
    starts = [0]
    for length in phase.lengths:
        starts.append(starts[-1] + length)
    pieces = []
    for first in range(0, count, per_launch):
        last = min(count, first + per_launch)
        low, high = starts[first], starts[last]
        pieces.append(gated_delta_kernel(
            q[low:high], k[low:high], v[low:high], g[low:high],
            beta[low:high], state[first:last],
            phase.cu_seqlens[first:last + 1] - low, scale))
    return torch.cat(pieces)


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
        if use_kernels:
            mixed, tails = causal_conv_kernel(
                mixed, tensors["conv_weight"], phase, history,
                tensors["conv_bias"])
        else:
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
            core = _kernel_in_batches(spec, q, k, v, g, b, state, phase, scale)
        else:
            core = gated_delta_reference(q, k, v, g, b, state,
                                         phase.lengths, scale)
        if phase.write_slots is not None:
            arena.gdn_state[phase.write_slots, layer_slot] = state
        if phase.output:
            out.index_copy_(0, phase.rows, core)
    return out
