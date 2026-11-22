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

from quail.common.budgets import CHECKPOINT_PASS_QUESTIONS


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
    contiguous_from_zero: bool = False
    contiguous_span: tuple | None = None
    read_slots_i32: object = None
    write_slots_i32: object = None
    unique_writes: bool = False
    same_unique_slots: bool = False
    moves_in_place: bool = False


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

    def _can_run_transient(group):
        if (not group["fresh"] or not group.get("transient_state")
                or group["start"] or group.get("kept") or group["f"] <= 0
                or group.get("share_points")):
            return False
        begins, ends = group["suffix_spans"]
        return (len(begins) == 1
                and int(begins[0]) == group["prefix_end"]
                and int(ends[0]) > int(begins[0]) > group["row0"])

    all_transient = bool(groups) and all(_can_run_transient(g) for g in groups)
    extended_after = set()
    # slots a question of this chunk reads at their current length
    suffix_slots = set()
    for group in (() if all_transient else groups):
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

    def _contiguous_span(segments):
        if not segments or not len(segments[0]):
            return None
        first = int(segments[0][0])
        cursor = first
        for part in segments:
            length = len(part)
            if not length or int(part[0]) != cursor:
                return None
            if length > 1 and (int(part[-1]) != cursor + length - 1
                               or not np.array_equal(
                                   part, np.arange(cursor, cursor + length,
                                                   dtype=np.int64))):
                return None
            cursor += length
        return (first, cursor) if cursor > first else None

    def build(name, rows, lengths, reads, writes, output=True):
        cu = np.concatenate([[0], np.cumsum(lengths)])
        reads_arr = None if reads is None else np.asarray(reads, dtype=np.int64)
        writes_arr = None if writes is None else np.asarray(writes, dtype=np.int64)
        unique_writes = (writes_arr is not None
                         and len(np.unique(writes_arr)) == len(writes_arr))
        same_unique = (unique_writes and reads_arr is not None
                       and np.array_equal(reads_arr, writes_arr))
        # each moved state copies into its write slot before the phase runs
        # in place, so no write slot may hold another sequence's read
        moved = None
        if unique_writes and reads_arr is not None and not same_unique:
            owner = {int(slot): index for index, slot in enumerate(writes_arr)}
            if all(owner.get(int(slot), index) == index
                   for index, slot in enumerate(reads_arr)):
                moved = reads_arr != writes_arr
        span = _contiguous_span(rows)
        phase = GdnPhase(
            rows=stage(f"gdn_{name}_rows", np.concatenate(rows), torch.int64),
            cu_seqlens=stage(f"gdn_{name}_cu", cu, torch.int32),
            lengths=tuple(lengths),
            read_slots=(None if reads_arr is None else
                        stage(f"gdn_{name}_read", reads_arr, torch.int64)),
            write_slots=(None if writes_arr is None else
                         stage(f"gdn_{name}_write", writes_arr, torch.int64)),
            output=output,
            contiguous_from_zero=span is not None and span[0] == 0,
            contiguous_span=span,
            read_slots_i32=(None if reads_arr is None else stage(
                f"gdn_{name}_read32", reads_arr.astype(np.int32), torch.int32)),
            write_slots_i32=(None if writes_arr is None else stage(
                f"gdn_{name}_write32", writes_arr.astype(np.int32), torch.int32)),
            unique_writes=unique_writes,
            same_unique_slots=same_unique,
            moves_in_place=moved is not None)
        if moved is not None and moved.any():
            phase.copy_reads = stage(f"gdn_{name}_copy_read", reads_arr[moved],
                                     torch.int64)
            phase.copy_writes = stage(f"gdn_{name}_copy_write",
                                      writes_arr[moved], torch.int64)
        return phase

    def build_decode(name, rows, slots):
        reads = np.array([pair[0] for pair in slots], dtype=np.int64)
        writes = np.array([pair[1] for pair in slots], dtype=np.int64)
        phase = build(name, rows, [1] * len(rows), reads, writes)
        moved = reads != writes
        phase.state_rows = stage(f"gdn_{name}_state_rows", writes + 1,
                                 torch.int64)
        if moved.any() and phase.copy_reads is None:
            phase.copy_reads = stage(f"gdn_{name}_copy_read", reads[moved],
                                     torch.int64)
            phase.copy_writes = stage(f"gdn_{name}_copy_write", writes[moved],
                                      torch.int64)
        return phase

    if all_transient:
        segments = [np.arange(g["row0"], int(g["suffix_spans"][1][0]),
                              dtype=np.int64) for g in groups]
        lengths = [len(part) for part in segments]
        return (build("level0", segments, lengths, None, None),)

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
    "auto" picks FlashInfer on Hopper and Blackwell, as vLLM's layer does.
    """
    import os

    import torch

    requested = os.environ.get("QUAIL_GDN_PREFILL", "auto")
    if requested == "triton":
        return "triton"
    if requested not in ("flashinfer", "auto"):
        raise ValueError(f"QUAIL_GDN_PREFILL={requested!r} is not "
                         "triton, flashinfer, or auto")
    capability = torch.cuda.get_device_capability()
    cuda_major = int((torch.version.cuda or "0").split(".")[0])
    if capability == (9, 0) or (capability[0] == 10 and cuda_major >= 13):
        return "flashinfer"
    if requested == "flashinfer":
        raise ValueError(
            "FlashInfer GDN prefill needs a Hopper GPU or a Blackwell GPU "
            "with CUDA 13 or newer")
    return "triton"


def _flashinfer_delta_rule(q, k, v, g, beta, state, cu_seqlens,
                           output_final_state, *, g_is_exp=False,
                           output=None, output_state=None, state_indices=None):
    """The same rule through FlashInfer's prefill kernel."""
    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    decay = g.float().contiguous() if g_is_exp else torch.exp(g.float()).contiguous()
    initial = None if state is None else (
        state if state.dtype == torch.float32 else state.float())
    kwargs = {}
    if output is not None:
        kwargs["output"] = output
    if output_state is not None:
        kwargs["output_state"] = output_state
    if state_indices is not None:
        kwargs["state_indices"] = state_indices
    # FlashInfer 0.6.18 passes the bounds unconverted to the context-parallel
    # kernel it picks for few sequences: int32 on SM100, int64 on SM90
    major = torch.cuda.get_device_capability(q.device)[0] if q.is_cuda else 0
    bounds = cu_seqlens.int() if major == 10 else cu_seqlens.long()
    result = chunk_gated_delta_rule(
        q=q.contiguous(), k=k.contiguous(),
        v=v.contiguous(), g=decay,
        beta=beta.float().contiguous(), initial_state=initial,
        output_final_state=output_final_state, cu_seqlens=bounds,
        **kwargs)
    return result if output_final_state else (result, None)


def gated_delta_kernel(q, k, v, g, beta, state, cu_seqlens, scale, *,
                       output_final_state, g_is_exp=False):
    """Return the recurrence outputs and optional final state.

    Inputs q and k are already normalized; g is the log decay unless
    g_is_exp is True. The kernel takes key heads fewer than value heads
    without repeating them. The backend is chosen by gdn_prefill_backend().
    """
    if gdn_prefill_backend() == "flashinfer":
        return _flashinfer_delta_rule(
            q, k, v, g, beta, state, cu_seqlens, output_final_state,
            g_is_exp=g_is_exp)
    from vllm.third_party.flash_linear_attention.ops import (
        chunk_gated_delta_rule,
    )

    if g_is_exp:
        g = g.log()
    out, final = chunk_gated_delta_rule(
        q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None],
        initial_state=state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens.long(), scale=scale,
        use_qk_l2norm_in_kernel=False)
    return out[0], final


def causal_conv_kernel(x, weight, phase, history, bias=None, activate=True,
                       *, state_pool=None):
    """Run vLLM's convolution and return outputs and updated private history.

    Args:
        x: Packed inputs with shape (rows, channels).
        weight: Convolution weights with shape (channels, width).
        phase: Sequence lengths and cached convolution metadata.
        history: Private state with shape (capacity, width - 1, channels),
            updated in place. None starts each sequence from zero.
        bias: Optional per-channel bias.
        activate: Whether SiLU follows the convolution.
        state_pool: Optional (slots, width - 1, channels) arena slice
            updated directly through phase.write_slots_i32.

    Returns:
        The outputs and each sequence's final convolution state (None when
        state_pool is updated in place).
    """
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    has_initial = history is not None or (
        state_pool is not None and phase.read_slots is not None)
    if phase.conv_metadata is None:
        bounds = torch.tensor([0, *np.cumsum(phase.lengths)], dtype=torch.int32)
        nums, batches, offsets = compute_causal_conv1d_metadata(
            bounds, device=x.device)
        count = len(phase.lengths)
        cache_indices = (
            phase.write_slots_i32 if state_pool is not None
            else torch.arange(count, dtype=torch.int32, device=x.device))
        phase.conv_metadata = SimpleNamespace(
            nums_dict=nums, batch_ptr=batches, token_chunk_offset_ptr=offsets,
            cache_indices=cache_indices,
            has_initial_state=torch.full(
                (count,), has_initial, dtype=torch.bool, device=x.device))
    metadata = phase.conv_metadata
    if state_pool is not None:
        conv_states = state_pool.transpose(1, 2)
    else:
        if history is None:
            capacity = 1 << (len(phase.lengths) - 1).bit_length()
            history = x.new_empty((capacity, weight.shape[1] - 1, x.shape[1]))
        conv_states = history.transpose(1, 2)
    out = causal_conv1d_fn(
        x.T, weight, bias, conv_states=conv_states,
        query_start_loc=phase.cu_seqlens, cache_indices=metadata.cache_indices,
        has_initial_state=metadata.has_initial_state,
        # Slot zero holds the first sequence; no private slot is reserved.
        null_block_id=None,
        activation="silu" if activate else None, metadata=metadata)
    return out.T, (None if state_pool is not None else history[:len(phase.lengths)])


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


def _kernel_in_batches(spec, q, k, v, g, beta, load, store, phase, scale, *,
                       g_is_exp=False):
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
    extra = {"g_is_exp": True} if g_is_exp else {}
    for first in range(0, count, per_launch):
        last = min(count, first + per_launch)
        low, high = starts[first], starts[last]
        state = load(first, last)
        out, final = gated_delta_kernel(
            q[low:high], k[low:high], v[low:high], g[low:high],
            beta[low:high], state,
            phase.cu_seqlens[first:last + 1] - low, scale,
            output_final_state=store is not None, **extra)
        pieces.append(out)
        if store is not None:
            store(first, last, final)
        del state, final
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def _pool_delta_rule(spec, q, k, v, g, beta, arena, phase, layer_slot, *,
                     output=None):
    """Run FlashInfer GDN prefill against the arena state pool."""
    import torch

    count = len(phase.lengths)
    has_reads = phase.read_slots is not None
    has_writes = phase.write_slots is not None
    is_sm100 = q.is_cuda and torch.cuda.get_device_capability(q.device)[0] == 10
    direct = (is_sm100 and not has_reads
              and (not has_writes or phase.unique_writes))
    per_launch = max(
        1,
        GRID_LIMIT // spec.gdn_value_heads if direct
        else min(GRID_LIMIT // spec.gdn_value_heads, SEQUENCES_PER_LAUNCH))
    pool = arena.gdn_state[:, layer_slot] if (has_reads or has_writes) else None
    target = (output if output is not None else torch.empty(
        (q.shape[0], spec.gdn_value_heads, spec.gdn_value_dim),
        dtype=q.dtype, device=q.device))
    low = 0
    for first in range(0, count, per_launch):
        last = min(count, first + per_launch)
        high = (q.shape[0] if last == count
                else low + sum(phase.lengths[first:last]))
        bounds = phase.cu_seqlens[first:last + 1] - low
        if direct:
            indices = (phase.write_slots_i32[first:last]
                       if has_writes else None)
            _flashinfer_delta_rule(
                q[low:high], k[low:high], v[low:high], g[low:high],
                beta[low:high], None, bounds, has_writes, g_is_exp=True,
                output=target[low:high],
                output_state=pool if has_writes else None,
                state_indices=indices)
        else:
            initial = pool[phase.read_slots[first:last]] if has_reads else None
            _, final = _flashinfer_delta_rule(
                q[low:high], k[low:high], v[low:high], g[low:high],
                beta[low:high], initial, bounds, has_writes, g_is_exp=True,
                output=target[low:high])
            if has_writes:
                copy_slots(final, torch.arange(last - first, device=q.device),
                           phase.write_slots[first:last], dst_pool=pool)
        low = high
    return target


def _decode_conv_kernel(x, weight, bias, arena, phase, layer_slot):
    """Advance each one-token sequence's convolution state in place.

    The caller has already moved each read slot's state to its write slot.
    """
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    out = x.new_empty(x.shape)
    causal_conv1d_update(
        x, arena.conv_state[:, layer_slot].transpose(1, 2), weight, bias,
        activation="silu", conv_state_indices=phase.write_slots_i32,
        null_block_id=None, out=out)
    return out


_GDN_KERNELS = {}


def _gdn_kernels() -> dict:
    """Compile the Gated DeltaNet Triton kernels on first use."""
    if _GDN_KERNELS:
        return _GDN_KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gather_gdn_inputs(
        qkv_ptr, a_ptr, b_ptr, rows_ptr, qkv_out_ptr, a_out_ptr, b_out_ptr,
        stride_qkv, stride_a, stride_b, stride_qo, stride_ao, stride_bo,
        CONV_DIM: tl.constexpr, HV: tl.constexpr, QKV_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr, HV_BLOCK: tl.constexpr,
    ):
        dst_row = tl.program_id(0).to(tl.int64)
        col_block = tl.program_id(1)
        src_row = tl.load(rows_ptr + dst_row).to(tl.int64)
        if col_block < QKV_BLOCKS:
            offs = col_block * BLOCK + tl.arange(0, BLOCK)
            mask = offs < CONV_DIM
            vals = tl.load(qkv_ptr + src_row * stride_qkv + offs,
                           mask=mask, other=0.0)
            tl.store(qkv_out_ptr + dst_row * stride_qo + offs, vals, mask=mask)
        else:
            hv_offs = tl.arange(0, HV_BLOCK)
            hv_mask = hv_offs < HV
            tl.store(a_out_ptr + dst_row * stride_ao + hv_offs,
                     tl.load(a_ptr + src_row * stride_a + hv_offs,
                             mask=hv_mask, other=0.0), mask=hv_mask)
            tl.store(b_out_ptr + dst_row * stride_bo + hv_offs,
                     tl.load(b_ptr + src_row * stride_b + hv_offs,
                             mask=hv_mask, other=0.0), mask=hv_mask)

    @triton.jit
    def scatter_gdn_rows(
        src_ptr, rows_ptr, dst_ptr, stride_src, stride_dst,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        src_row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < WIDTH
        dst_row = tl.load(rows_ptr + src_row).to(tl.int64)
        tl.store(dst_ptr + dst_row * stride_dst + offs,
                 tl.load(src_ptr + src_row * stride_src + offs,
                         mask=mask, other=0.0), mask=mask)

    @triton.jit
    def copy_slots(src_ptr, dst_ptr, src_index_ptr, dst_index_ptr, stride_src,
                   stride_dst, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        # a wave of blocks covers one slot, so it touches few pages
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        pair = tl.program_id(1)
        mask = offs < WIDTH
        src = tl.load(src_index_ptr + pair).to(tl.int64)
        dst = tl.load(dst_index_ptr + pair).to(tl.int64)
        tl.store(dst_ptr + dst * stride_dst + offs,
                 tl.load(src_ptr + src * stride_src + offs, mask=mask),
                 mask=mask)

    # PDL invariant: loads before gdc_wait read only weights and staged indices.
    @triton.jit
    def gdn_decode_step(
        x_ptr, a_ptr, b_ptr, a_log_ptr, dt_bias_ptr, state_ptr, state_rows_ptr,
        out_ptr, out_rows_ptr, stride_x, stride_a, stride_b, stride_state,
        stride_out, scale, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
        V: tl.constexpr, BV: tl.constexpr, HAS_OUT_ROWS: tl.constexpr,
        PDL: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        seq_head = tl.program_id(1)
        seq = (seq_head // HV).to(tl.int64)
        hv = seq_head % HV
        h = hv // (HV // H)
        state_row = tl.load(state_rows_ptr + seq).to(tl.int64)
        dt_bias = tl.load(dt_bias_ptr + hv).to(tl.float32)
        a_log = tl.load(a_log_ptr + hv).to(tl.float32)
        out_row = tl.load(out_rows_ptr + seq).to(tl.int64) if HAS_OUT_ROWS else seq
        offs_k = tl.arange(0, K)
        offs_v = v_block * BV + tl.arange(0, BV)
        p_state = (state_ptr + state_row * stride_state + hv * V * K
                   + offs_v[:, None] * K + offs_k[None, :])
        if PDL:
            tl.extra.cuda.gdc_wait()
        row = x_ptr + seq * stride_x
        q = tl.load(row + h * K + offs_k).to(tl.float32)
        k = tl.load(row + H * K + h * K + offs_k).to(tl.float32)
        v = tl.load(row + 2 * H * K + hv * V + offs_v).to(tl.float32)
        q = (q / tl.sqrt(tl.sum(q * q) + 1e-6)).to(tl.bfloat16).to(tl.float32) * scale
        k = (k / tl.sqrt(tl.sum(k * k) + 1e-6)).to(tl.bfloat16).to(tl.float32)
        x = tl.load(a_ptr + seq * stride_a + hv).to(tl.float32) + dt_bias
        softplus = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)),
                            tl.log(1.0 + tl.exp(x)))
        g = -tl.exp(a_log) * tl.where(x <= 20.0, softplus, x)
        beta = tl.sigmoid(tl.load(b_ptr + seq * stride_b + hv).to(tl.float32))
        state = tl.load(p_state) * tl.exp(g)
        v = (v - tl.sum(state * k[None, :], 1)) * beta
        state += v[:, None] * k[None, :]
        o = tl.sum(state * q[None, :], 1)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(p_state, state)
        tl.store(out_ptr + out_row * stride_out + hv * V + offs_v,
                 o.to(out_ptr.dtype.element_ty))

    _GDN_KERNELS.update(
        gather_gdn_inputs=gather_gdn_inputs,
        scatter_gdn_rows=scatter_gdn_rows,
        copy_slots=copy_slots,
        gdn_decode_step=gdn_decode_step,
    )
    return _GDN_KERNELS


def gather_gdn_inputs(qkv, a, b, rows):
    """Gather qkv, a, and b at selected row indices in one vectorized launch."""
    import torch

    from quail.backends.quail.executor.attention import launch_triton

    count, conv_dim, hv = rows.shape[0], qkv.shape[1], a.shape[1]
    qkv_out = torch.empty((count, conv_dim), dtype=qkv.dtype, device=qkv.device)
    a_out = torch.empty((count, hv), dtype=a.dtype, device=a.device)
    b_out = torch.empty((count, hv), dtype=b.dtype, device=b.device)
    if count == 0:
        return qkv_out, a_out, b_out
    block = 1024
    qkv_blocks = (conv_dim + block - 1) // block
    hv_block = max(16, 1 << (hv - 1).bit_length())
    launch_triton(
        _gdn_kernels()["gather_gdn_inputs"], (count, qkv_blocks + 1),
        (qkv, a, b, rows, qkv_out, a_out, b_out, qkv.stride(0), a.stride(0),
         b.stride(0), qkv_out.stride(0), a_out.stride(0), b_out.stride(0)),
        key=(conv_dim, hv, qkv_blocks, block, hv_block),
        kwargs=dict(CONV_DIM=conv_dim, HV=hv, QKV_BLOCKS=qkv_blocks,
                    BLOCK=block, HV_BLOCK=hv_block, num_warps=4))
    return qkv_out, a_out, b_out


def scatter_gdn_rows(core, rows, out):
    """Scatter 3D core recurrence outputs into out at selected row indices."""
    from quail.backends.quail.executor.attention import launch_triton

    count = rows.shape[0]
    if count == 0:
        return out
    core_2d, out_2d = core.reshape(count, -1), out.view(out.shape[0], -1)
    width, block = core_2d.shape[1], 1024
    launch_triton(
        _gdn_kernels()["scatter_gdn_rows"],
        (count, (width + block - 1) // block),
        (core_2d, rows, out_2d, core_2d.stride(0), out_2d.stride(0)),
        key=(width, block), kwargs=dict(WIDTH=width, BLOCK=block, num_warps=4))
    return out


def copy_slots(pool, src, dst, *, dst_pool=None):
    """Copy pool[src[i]] into dst_pool[dst[i]]; within one pool no dst is a src."""
    from quail.backends.quail.executor.attention import launch_triton

    target = pool if dst_pool is None else dst_pool
    count = src.shape[0]
    if count == 0:
        return target
    assert pool[0].is_contiguous() and target[0].is_contiguous()
    width, block = pool[0].numel(), 4096
    launch_triton(
        _gdn_kernels()["copy_slots"], ((width + block - 1) // block, count),
        (pool, target, src, dst, pool.stride(0), target.stride(0)),
        key=(width, block), kwargs=dict(WIDTH=width, BLOCK=block, num_warps=8))
    return target


def gdn_decode_step(conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out,
                    *, num_key_heads, scale, out_rows=None, block_v=8,
                    num_warps=1):
    """Advance one-token Gated DeltaNet sequences and write their outputs."""
    from quail.backends.quail.executor.attention import launch_triton, pdl_enabled

    count = state_rows.shape[0]
    if count == 0:
        return out
    _, value_heads, value_dim, key_dim = state_pool.shape
    assert state_pool.stride()[1:] == (value_dim * key_dim, key_dim, 1)
    out_2d = out.view(out.shape[0], -1)
    has_out_rows = out_rows is not None
    pdl = pdl_enabled(conv_out.device)
    launch_triton(
        _gdn_kernels()["gdn_decode_step"],
        (value_dim // block_v, count * value_heads),
        (conv_out, a, b, a_log, dt_bias, state_pool, state_rows, out_2d,
         state_rows if not has_out_rows else out_rows,
         conv_out.stride(0), a.stride(0), b.stride(0), state_pool.stride(0),
         out_2d.stride(0), scale),
        key=(num_key_heads, value_heads, key_dim, value_dim, block_v,
             has_out_rows, num_warps, pdl),
        kwargs=dict(H=num_key_heads, HV=value_heads, K=key_dim, V=value_dim,
                    BV=block_v, HAS_OUT_ROWS=has_out_rows, PDL=pdl,
                    num_warps=num_warps, launch_pdl=pdl))
    return out


def _move_states(arena, phase, layer_slot):
    """Copy each moved sequence's read-slot state into its write slot."""
    if phase.copy_reads is None:
        return
    if phase.moves_in_place and arena.gdn_state.is_cuda:
        copy_slots(arena.gdn_state[:, layer_slot], phase.copy_reads,
                   phase.copy_writes)
        copy_slots(arena.conv_state[:, layer_slot], phase.copy_reads,
                   phase.copy_writes)
        return
    arena.gdn_state[phase.copy_writes, layer_slot] = (
        arena.gdn_state[phase.copy_reads, layer_slot])
    arena.conv_state[phase.copy_writes, layer_slot] = (
        arena.conv_state[phase.copy_reads, layer_slot])


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
    active_phases = [phase for phase in phases if phase is not None]
    single_identity = (
        len(active_phases) == 1
        and getattr(active_phases[0], "contiguous_from_zero", False)
        and active_phases[0].output
        and active_phases[0].rows.numel() == rows
    )
    covered_rows = sum(phase.rows.numel() for phase in active_phases if phase.output)
    full_cover = covered_rows == rows
    out = (output if single_identity and output is not None else
           (output if full_cover else output.zero_()) if output is not None else
           (torch.empty if full_cover else torch.zeros)(
               (rows, spec.gdn_value_heads, spec.gdn_value_dim),
               dtype=qkv.dtype, device=qkv.device)
           if not single_identity else None)
    scale = spec.gdn_key_dim ** -0.5
    use_flashinfer = qkv.is_cuda and gdn_prefill_backend() == "flashinfer"
    for phase in active_phases:
        span = phase.contiguous_span
        if single_identity:
            mixed_in = qkv
            a_in, b_in = tensors["a"], tensors["b"]
        elif span is not None:
            mixed_in = qkv[span[0]:span[1]]
            a_in = tensors["a"][span[0]:span[1]]
            b_in = tensors["b"][span[0]:span[1]]
        elif qkv.is_cuda:
            mixed_in, a_in, b_in = gather_gdn_inputs(
                qkv, tensors["a"], tensors["b"], phase.rows)
        else:
            mixed_in = qkv.index_select(0, phase.rows)
            a_in = tensors["a"].index_select(0, phase.rows)
            b_in = tensors["b"].index_select(0, phase.rows)
        decode = phase.state_rows is not None and phase.write_slots_i32 is not None
        # the phase runs on its write slots in the arena pools after each
        # moved state is copied there
        in_place = qkv.is_cuda and phase.write_slots_i32 is not None and (
            (phase.read_slots is None and phase.unique_writes)
            or phase.same_unique_slots or phase.moves_in_place)
        if decode or in_place:
            _move_states(arena, phase, layer_slot)
        if decode:
            mixed = _decode_conv_kernel(
                mixed_in, tensors["conv_weight"], tensors["conv_bias"],
                arena, phase, layer_slot)
            if out is None:
                out = torch.empty(
                    (rows, spec.gdn_value_heads, spec.gdn_value_dim),
                    dtype=qkv.dtype, device=qkv.device)
            if single_identity:
                target, out_rows = out, None
            elif not phase.output:
                target = torch.empty(
                    (len(phase.lengths), spec.gdn_value_heads,
                     spec.gdn_value_dim), dtype=qkv.dtype, device=qkv.device)
                out_rows = None
            elif span is not None:
                target, out_rows = out[span[0]:span[1]], None
            else:
                target, out_rows = out, phase.rows
            gdn_decode_step(
                mixed, a_in, b_in, tensors["A_log"], tensors["dt_bias"],
                arena.gdn_state_rows[:, layer_slot], phase.state_rows, target,
                num_key_heads=spec.gdn_key_heads, scale=scale,
                out_rows=out_rows)
            if single_identity:
                return out
            continue
        if in_place:
            mixed, _ = causal_conv_kernel(
                mixed_in, tensors["conv_weight"], phase, None,
                tensors["conv_bias"], state_pool=arena.conv_state[:, layer_slot])
        else:
            history = _conv_history(arena, phase, layer_slot)
            mixed, tails = causal_conv_kernel(
                mixed_in, tensors["conv_weight"], phase, history,
                tensors["conv_bias"])
            if phase.write_slots is not None:
                arena.conv_state[phase.write_slots, layer_slot] = tails
        g_exp = use_flashinfer
        q, k, v, g, b = fused_post_conv_prep(
            conv_output=mixed, a=a_in, b=b_in,
            A_log=tensors["A_log"], dt_bias=tensors["dt_bias"],
            num_k_heads=spec.gdn_key_heads, head_k_dim=spec.gdn_key_dim,
            head_v_dim=spec.gdn_value_dim, apply_l2norm=True,
            output_g_exp=g_exp)
        if use_flashinfer and spec.gdn_key_dim == 128:
            direct_out = (out if single_identity
                          else out[span[0]:span[1]]
                          if phase.output and span is not None else None)
            core = _pool_delta_rule(
                spec, q, k, v, g, b, arena, phase, layer_slot,
                output=direct_out)
            if single_identity:
                return core
            if phase.output and direct_out is None:
                if out.is_cuda:
                    scatter_gdn_rows(core, phase.rows, out)
                else:
                    out.index_copy_(0, phase.rows, core)
            continue

        def load(first, last, phase=phase):
            if phase.read_slots is None:
                if use_flashinfer:
                    return None
                return torch.zeros(
                    (last - first, spec.gdn_value_heads,
                     spec.gdn_value_dim, spec.gdn_key_dim),
                    dtype=torch.float32, device=qkv.device)
            return arena.gdn_state[phase.read_slots[first:last], layer_slot]

        def store(first, last, state, phase=phase):
            arena.gdn_state[phase.write_slots[first:last], layer_slot] = state

        core = _kernel_in_batches(
            spec, q, k, v, g, b, load,
            store if phase.write_slots is not None else None, phase, scale,
            g_is_exp=g_exp)
        if single_identity:
            if out is not None:
                out.copy_(core)
                return out
            return core
        if phase.output:
            if out.is_cuda:
                scatter_gdn_rows(core, phase.rows, out)
            else:
                out.index_copy_(0, phase.rows, core)
    return out if out is not None else torch.zeros(
        (rows, spec.gdn_value_heads, spec.gdn_value_dim),
        dtype=qkv.dtype, device=qkv.device)
