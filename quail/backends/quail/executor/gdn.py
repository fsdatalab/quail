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

The convolution and recurrence run vLLM's kernels.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from quail.cost.budgets import CHECKPOINT_PASS_QUESTIONS


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
        conv_metadata: vLLM convolution metadata shared by all layers.
            A phase consistently starts from zero or from saved state.
        state_rows: For a phase of one-token sequences, the arena's
            state row of each write slot (the slot plus one); the
            recurrent kernel updates those rows in place. None runs the
            chunked kernel on gathered state.
        copy_reads: Read slots whose state first moves to a different
            write slot, or None.
        copy_writes: The write slots those states move to, or None.
    """

    rows: object
    cu_seqlens: object
    lengths: tuple
    read_slots: object = None
    write_slots: object = None
    output: bool = True
    conv_metadata: object = None
    state_rows: object = None
    copy_reads: object = None
    copy_writes: object = None


def plan_phases(arena, groups, stage):
    """Lay a chunk's groups out as the phases of every Gated DeltaNet layer.

    Each layer runs the phases in this order, and a phase reads the
    slots the ones before it wrote:

    1. level 0: a fresh document's rows from zero state, writing its
       slot, and its state after each point a borrower starts from.
    2. levels 1 and up: rows that start from a slot a lower level wrote,
       in this chunk or an earlier one. A document that borrows a
       prefix starts from its parent's state at the shared point; the
       retained tokens of a group whose whole first suffix is kept (a
       stage frame) run from its slot back into it.
    3. suffix: every question from its document's slot, writing nothing.
    4. extension: the leading retained tokens of a suffix that also
       holds a question, from the slot back into it. The suffix ran
       from the state before them, so the slot advances after it.

    Allocates a checkpoint slot for each fresh document and records the
    tokens it covers, and a slot for each shared point. Retained suffix
    tokens advance a second slot, so the document-end checkpoint
    survives for an operator that asks at the document's length after
    the KV rewinds. A group must be asked at one of the two lengths
    exactly, since a state cannot rewind.

    Args:
        arena: KVArena with checkpoint slots.
        groups: One mapping per group in chunk order, with key, fresh,
            f, start, row0, prefix_end, keeps, kept, suffix_spans (the
            begin and end rows of each suffix), borrow_key (the parent
            of a borrowed prefix), and share_points (the token counts
            a borrower starts from).
        stage: Callable (name, values, dtype) that copies host values to
            the GPU.

    Returns:
        The phases in run order, each a GdnPhase.

    Raises:
        BorrowStateMissingError: A group borrows a prefix no state covers.
        ValueError: A group keeps more
            tokens than its first suffix holds, follows a group of
            its key that extends the checkpoint after its suffix,
            advances a checkpoint in place that a question of the chunk
            reads at its current length, is an empty document, or has
            a suffix that holds no tokens.
        ArenaFullError: Every checkpoint slot belongs to a pinned key.
    """
    undo = []
    # a later group's slot allocation must not evict a parent that an
    # earlier group reads, or the two would share one slot
    parents = [group["borrow_key"] for group in groups
               if group["fresh"] and group["start"]
               and group.get("borrow_key") is not None]
    arena.read_state(parents)
    try:
        return _plan_phases(arena, groups, stage, undo)
    except Exception:
        # a failed plan leaves no record or slot for a state it never
        # computed, so a retry after the chunk splits asks at the old
        # length and no borrower reads a share slot nothing wrote
        accounting = arena.accounting
        for kind, key, *rest in reversed(undo):
            if kind == "extension":
                had_slot, previous = rest
                if had_slot:
                    accounting.ext_tokens[key] = previous
                else:
                    accounting.drop_extension(key)
            elif kind == "checkpoint":
                had_slot, previous = rest
                if previous is None:
                    accounting.slot_tokens.pop(key, None)
                else:
                    accounting.slot_tokens[key] = previous
                if not had_slot and key in accounting.slots:
                    accounting.free_slots.append(accounting.slots.pop(key))
            else:
                slot = accounting.share_slots.pop((key, rest[0]), None)
                if slot is not None:
                    accounting.free_slots.append(slot)
        raise
    finally:
        arena.done_reading(parents)


def _plan_phases(arena, groups, stage, undo):
    """Plan a chunk's phases, logging each record and slot it takes in `undo`."""
    import torch

    from quail.backends.quail.executor.chunk import (
        ArenaFullError,
        BorrowStateMissingError,
    )

    # level 0 runs from zero state; level n >= 1 reads slots that a lower
    # level wrote in this chunk or an earlier one
    levels = {}
    ready = {}
    lists = {name: ([], [], []) for name in ("suffix", "extension")}

    def segment(rows, read, write):
        """Queue rows that start from slot `read` and leave state in `write`."""
        level = 0 if read is None else max(1, ready.get(read, -1) + 1)
        rows_list, lengths, slots = levels.setdefault(level, ([], [], []))
        rows_list.append(rows)
        lengths.append(len(rows))
        slots.append((read, write))
        ready[write] = level

    def add(name, rows, slot):
        rows_list, lengths, slots = lists[name]
        rows_list.append(rows)
        lengths.append(len(rows))
        slots.append(slot)

    extended_after = set()
    # slots a question of this chunk reads at their current length
    suffix_slots = set()
    for group in groups:
        key = group["key"]
        kept = group.get("kept", 0)
        start = group["start"]
        if start and not group["fresh"]:
            raise ValueError(
                f"group {key!r}: a resident document cannot start at "
                f"token {start}")
        begins, ends = (spans.tolist() for spans in group["suffix_spans"])
        if kept and (not begins or kept > ends[0] - begins[0]):
            raise ValueError(
                f"group {key!r}: keeps {kept} suffix tokens, more than "
                f"its first suffix holds")
        if group["fresh"]:
            if group["f"] == 0:
                raise ValueError(
                    f"group {key!r}: an empty document has no state to "
                    f"checkpoint")
            had_slot = arena.has_checkpoint(key)
            previous = arena.accounting.slot_tokens.get(key)
            source = arena.checkpoint_slot(key)
            if source is None:
                raise ArenaFullError(
                    f"no free checkpoint slot for {key!r}: all "
                    f"{arena.num_state_slots} belong to documents in use")
            undo.append(("checkpoint", key, had_slot, previous))
            first, last = group["row0"], group["prefix_end"]
            read = None
            if start:
                parent = group.get("borrow_key")
                read = (None if parent is None
                        else arena.shared_state(parent, start))
                if read is None:
                    raise BorrowStateMissingError(
                        key, f"group {key!r}: borrows {start} tokens, but "
                        f"no state of the parent covers them")
                if last <= first:
                    raise ValueError(
                        f"group {key!r}: borrows its whole prefix")
            points = sorted({t for t in group.get("share_points", ())
                             if start < t < group["f"]})
            bounds = [start] + points + [group["f"]]
            for low, high in zip(bounds, bounds[1:]):
                if high == group["f"]:
                    write = source
                else:
                    existed = (key, high) in arena.accounting.share_slots
                    write = arena.share_slot(key, high)
                    if write is None:
                        raise ArenaFullError(
                            f"no free checkpoint slot to share {key!r} "
                            f"at {high} tokens")
                    if not existed:
                        undo.append(("share", key, high))
                if last > first:
                    segment(np.arange(first + low - start,
                                      first + high - start, dtype=np.int64),
                            read, write)
                read = write
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
            if pure and target == source and source in suffix_slots:
                raise ValueError(
                    f"group {key!r}: advances a checkpoint in place that "
                    f"a question of this chunk reads at {group['f']} tokens")
            undo.append(("extension", key, held is not None,
                         None if held is None else held[1]))
            arena.record_extension(key, group["f"] + kept)
            head = np.arange(begins[0], begins[0] + kept, dtype=np.int64)
            if pure:
                segment(head, source, target)
            else:
                rows_list, lengths, slots = lists["extension"]
                rows_list.append(head)
                lengths.append(len(head))
                slots.append((source, target))
                extended_after.add(key)
        if pure:
            continue
        for begin, end in zip(begins, ends):
            # an empty sequence would repeat a cu_seqlens entry
            if end <= begin:
                raise ValueError(f"group {key!r}: a suffix holds no tokens")
            add("suffix", np.arange(begin, end, dtype=np.int64), source)
            suffix_slots.add(source)

    def build(name, rows, lengths, reads, writes, output=True):
        cu = np.concatenate([[0], np.cumsum(lengths)])
        return GdnPhase(
            rows=stage(f"gdn_{name}_rows", np.concatenate(rows), torch.int64),
            cu_seqlens=stage(f"gdn_{name}_cu", cu, torch.int32),
            lengths=tuple(lengths),
            read_slots=(None if reads is None else
                        stage(f"gdn_{name}_read", reads, torch.int64)),
            write_slots=(None if writes is None else
                         stage(f"gdn_{name}_write", writes, torch.int64)),
            output=output)

    def build_decode(name, rows, slots):
        reads = np.array([pair[0] for pair in slots], dtype=np.int64)
        writes = np.array([pair[1] for pair in slots], dtype=np.int64)
        phase = build(name, rows, [1] * len(rows), reads, writes)
        moved = reads != writes
        phase.state_rows = stage(f"gdn_{name}_state_rows", writes + 1,
                                 torch.int64)
        if moved.any():
            phase.copy_reads = stage(f"gdn_{name}_copy_read", reads[moved],
                                     torch.int64)
            phase.copy_writes = stage(f"gdn_{name}_copy_write", writes[moved],
                                      torch.int64)
        return phase

    phases = []
    for level in sorted(levels):
        rows, lengths, slots = levels[level]
        if level == 0:
            phases.append(build("level0", rows, lengths, None,
                                [pair[1] for pair in slots]))
            continue
        # a one-token sequence is an answer's decode step
        single = [index for index, length in enumerate(lengths) if length == 1]
        rest = [index for index, length in enumerate(lengths) if length != 1]
        if rest:
            phases.append(build(
                f"level{level}", [rows[i] for i in rest],
                [lengths[i] for i in rest], [slots[i][0] for i in rest],
                [slots[i][1] for i in rest]))
        if single:
            phases.append(build_decode(
                f"level{level}_decode", [rows[i] for i in single],
                [slots[i] for i in single]))
    rows, lengths, slots = lists["suffix"]
    if lengths:
        phases.append(build("suffix", rows, lengths, slots, None))
    rows, lengths, slots = lists["extension"]
    if lengths:
        phases.append(build("extension", rows, lengths,
                            [pair[0] for pair in slots],
                            [pair[1] for pair in slots], output=False))
    return tuple(phases)


# ---- kernel path ------------------------------------------------------

def gdn_prefill_backend():
    """Return "flashinfer" or "triton" for the chunked recurrence.

    QUAIL_GDN_PREFILL is "auto" (default), "triton", or "flashinfer".
    "auto" picks FlashInfer on Hopper, as vLLM's layer does.
    """
    import os

    import torch

    requested = os.environ.get("QUAIL_GDN_PREFILL", "auto")
    if requested == "triton":
        return "triton"
    if requested not in ("flashinfer", "auto"):
        raise ValueError(f"QUAIL_GDN_PREFILL={requested!r} is not "
                         "triton, flashinfer, or auto")
    if torch.cuda.get_device_capability() == (9, 0):
        return "flashinfer"
    if requested == "flashinfer":
        raise ValueError("FlashInfer GDN prefill needs a Hopper GPU")
    return "triton"


def _flashinfer_delta_rule(q, k, v, g, beta, state, cu_seqlens,
                          output_final_state):
    """The same rule through FlashInfer's prefill kernel, as vLLM calls it."""
    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    result = chunk_gated_delta_rule(
        q=q.contiguous(), k=k.contiguous(),
        v=v.contiguous(), g=torch.exp(g.float()).contiguous(),
        beta=beta.float().contiguous(), initial_state=state.float(),
        output_final_state=output_final_state, cu_seqlens=cu_seqlens.long())
    return result if output_final_state else (result, None)


def gated_delta_kernel(q, k, v, g, beta, state, cu_seqlens, scale, *,
                       output_final_state):
    """Return the recurrence outputs and optional final state.

    Inputs q and k are already normalized; g is the log decay. The
    kernel takes key heads fewer than value heads without repeating
    them. The backend is chosen by gdn_prefill_backend().
    """
    if gdn_prefill_backend() == "flashinfer":
        return _flashinfer_delta_rule(q, k, v, g, beta, state,
                                      cu_seqlens, output_final_state)
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule,
    )

    out, final = chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens.long(), scale=scale,
        use_qk_l2norm_in_kernel=False)
    return out[0], final


def causal_conv_kernel(x, weight, phase, history, bias=None, activate=True):
    """Run vLLM's convolution and return outputs and updated private history.

    Args:
        x: Packed inputs with shape (rows, channels).
        weight: Convolution weights with shape (channels, width).
        phase: Sequence lengths and cached convolution metadata.
        history: Private state with shape (capacity, width - 1, channels),
            updated in place. None starts each sequence from zero.
        bias: Optional per-channel bias.
        activate: Whether SiLU follows the convolution.

    Returns:
        The outputs and each sequence's final convolution state.
    """
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    if phase.conv_metadata is None:
        bounds = torch.tensor([0, *np.cumsum(phase.lengths)], dtype=torch.int32)
        nums, batches, offsets = compute_causal_conv1d_metadata(
            bounds, device=x.device)
        count = len(phase.lengths)
        phase.conv_metadata = SimpleNamespace(
            nums_dict=nums, batch_ptr=batches, token_chunk_offset_ptr=offsets,
            cache_indices=torch.arange(count, dtype=torch.int32, device=x.device),
            has_initial_state=torch.full(
                (count,), history is not None, dtype=torch.bool, device=x.device))
    metadata = phase.conv_metadata
    if history is None:
        capacity = 1 << (len(phase.lengths) - 1).bit_length()
        history = x.new_empty((capacity, weight.shape[1] - 1, x.shape[1]))
    out = causal_conv1d_fn(
        x.T, weight, bias, conv_states=history.transpose(1, 2),
        query_start_loc=phase.cu_seqlens, cache_indices=metadata.cache_indices,
        has_initial_state=metadata.has_initial_state,
        # Slot zero holds the first sequence; no private slot is reserved.
        null_block_id=None,
        activation="silu" if activate else None, metadata=metadata)
    return out.T, history[:len(phase.lengths)]


def _conv_history(arena, phase, layer_slot):
    """Gather a layer's convolution history into private state."""
    import torch

    if phase.read_slots is None:
        return None
    count = len(phase.lengths)
    # vLLM specializes on capacity; powers of two bound the compiled variants.
    capacity = 1 << (count - 1).bit_length()
    source = arena.conv_state[:, layer_slot]
    history = source.new_empty((capacity, *source.shape[1:]))
    torch.index_select(source, 0, phase.read_slots, out=history[:count])
    return history


# a CUDA launch has at most this many blocks along its second grid axis,
# and the kernel spends one per sequence and value head
GRID_LIMIT = 65_535
# A launch holds the state it gathered and the final state the kernel
# writes: two copies per sequence of one layer. The pass budget reserves
# one copy for each of CHECKPOINT_PASS_QUESTIONS questions, so a launch
# takes half that many sequences.
SEQUENCES_PER_LAUNCH = CHECKPOINT_PASS_QUESTIONS // 2


def _kernel_in_batches(spec, q, k, v, g, beta, load, store, phase, scale):
    """Run the chunked kernel over as many sequences as one launch holds.

    A launch gathers its sequences' state with load(first, last).
    When store is supplied, store(first, last, state) saves the final
    state. Temporary state is bounded by the launch size.
    """
    import torch

    count = len(phase.lengths)
    per_launch = max(1, min(GRID_LIMIT // spec.gdn_value_heads,
                            SEQUENCES_PER_LAUNCH))
    starts = [0]
    for length in phase.lengths:
        starts.append(starts[-1] + length)
    pieces = []
    for first in range(0, count, per_launch):
        last = min(count, first + per_launch)
        low, high = starts[first], starts[last]
        state = load(first, last)
        out, final = gated_delta_kernel(
            q[low:high], k[low:high], v[low:high], g[low:high],
            beta[low:high], state,
            phase.cu_seqlens[first:last + 1] - low, scale,
            output_final_state=store is not None)
        pieces.append(out)
        if store is not None:
            store(first, last, final)
        del state, final
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def _recurrent_decode(arena, phase, layer_slot, q, k, v, g, beta, scale):
    """Advance each one-token sequence's state in its arena row, in place.

    A sequence that reads another slot than it writes first copies that
    state into its write slot. Returns (sequences, value heads, value
    dim) outputs.
    """
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    if phase.copy_reads is not None:
        arena.gdn_state[phase.copy_writes, layer_slot] = (
            arena.gdn_state[phase.copy_reads, layer_slot])
    out, _ = fused_recurrent_gated_delta_rule(
        q[None], k[None], v[None], g[None], beta[None], scale=scale,
        initial_state=arena.gdn_state_rows[:, layer_slot],
        inplace_final_state=True, cu_seqlens=phase.cu_seqlens,
        ssm_state_indices=phase.state_rows)
    return out[0]


# ---- the layer --------------------------------------------------------

def short_conv_core(inputs, weight, phases, arena, layer_slot):
    """Run one short convolution layer's causal convolution.

    Args:
        inputs: (rows, channels) gated input, B times x.
        weight: (channels, width) filter taps; the last is the current row.
        phases: The chunk's phases from plan_phases.
        arena: KVArena holding the checkpoint pools.
        layer_slot: Index of this layer among the state layers.

    Returns:
        (rows, channels) convolution outputs, with no activation.
    """
    import torch

    out = torch.zeros_like(inputs)
    for phase in phases:
        if phase is None:
            continue
        mixed = inputs.index_select(0, phase.rows)
        history = _conv_history(arena, phase, layer_slot)
        mixed, tails = causal_conv_kernel(
            mixed, weight, phase, history, None, activate=False)
        if phase.write_slots is not None:
            arena.conv_state[phase.write_slots, layer_slot] = tails
        if phase.output:
            out.index_copy_(0, phase.rows, mixed)
    return out


def gdn_core(geometry, tensors, phases, arena, layer_slot, *, output=None):
    """Run one Gated DeltaNet layer's convolution and recurrence.

    Args:
        geometry: The model spec, for its head counts and dimensions.
        tensors: Dict with qkv (rows, conv channels), b, a, conv_weight,
            conv_bias, A_log, and dt_bias.
        phases: The chunk's phases from plan_phases.
        arena: KVArena holding the checkpoint pools.
        layer_slot: Index of this layer among the Gated DeltaNet layers.
        output: Optional (rows, value heads, value dim) tensor to write.

    Returns:
        (rows, value heads, value dim) recurrence outputs.
    """
    import torch
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    spec = geometry
    qkv = tensors["qkv"]
    rows = qkv.shape[0]
    out = (output.zero_() if output is not None else
           torch.zeros((rows, spec.gdn_value_heads, spec.gdn_value_dim),
                       dtype=qkv.dtype, device=qkv.device))
    scale = spec.gdn_key_dim ** -0.5
    for phase in phases:
        if phase is None:
            continue
        mixed = qkv.index_select(0, phase.rows)
        history = _conv_history(arena, phase, layer_slot)
        mixed, tails = causal_conv_kernel(
            mixed, tensors["conv_weight"], phase, history,
            tensors["conv_bias"])
        if phase.write_slots is not None:
            arena.conv_state[phase.write_slots, layer_slot] = tails
        q, k, v, g, b = fused_post_conv_prep(
            conv_output=mixed,
            a=tensors["a"].index_select(0, phase.rows),
            b=tensors["b"].index_select(0, phase.rows),
            A_log=tensors["A_log"], dt_bias=tensors["dt_bias"],
            num_k_heads=spec.gdn_key_heads, head_k_dim=spec.gdn_key_dim,
            head_v_dim=spec.gdn_value_dim, apply_l2norm=True,
            output_g_exp=False)
        if phase.state_rows is not None:
            core = _recurrent_decode(arena, phase, layer_slot, q, k, v, g, b,
                                     scale)
            if phase.output:
                out.index_copy_(0, phase.rows, core)
            continue

        def load(first, last, phase=phase):
            if phase.read_slots is None:
                return torch.zeros(
                    (last - first, spec.gdn_value_heads,
                     spec.gdn_value_dim, spec.gdn_key_dim),
                    dtype=torch.float32, device=qkv.device)
            return arena.gdn_state[phase.read_slots[first:last], layer_slot]

        def store(first, last, state, phase=phase):
            arena.gdn_state[phase.write_slots[first:last], layer_slot] = state

        core = _kernel_in_batches(
            spec, q, k, v, g, b, load,
            store if phase.write_slots is not None else None, phase, scale)
        if phase.output:
            out.index_copy_(0, phase.rows, core)
    return out
