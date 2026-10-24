"""Move rows between the arena and pinned host memory with one kernel launch.

Pinned host memory is mapped into the device's address space, so a
kernel reads and writes it directly over PCIe. A document's rows sit
in scattered pages of many pools, so copying them with the copy
engine would take one copy per page per layer or a contiguous staging
block on the device. The kernel here takes an index of source or
destination rows instead and moves every row in one launch, with no
device memory beyond the index. On an H100 it moves a 45 MB document
at about 26 GB/s either way.

The pinned tensor must come from torch's pinned allocator
(pin_memory=True); its data pointer is then valid on the device.
"""


def copy_rows(source, destination, *, source_rows=None, destination_rows=None):
    """Copy rows of `source` to `destination`, one of them on the host.

    Both are 2-D (rows, width) with the same width and dtype and
    contiguous rows; the stride between rows may differ. Exactly one
    of source_rows and destination_rows is given, an int64 tensor: the
    rows read from the source in order (a gather), or the rows written
    in the destination in order (a scatter). On CUDA the copy is
    queued on the current stream and the index is a device tensor; on
    the CPU it runs at once.
    """
    if (source_rows is None) == (destination_rows is None):
        raise ValueError("give the source rows or the destination rows")
    gather = source_rows is not None
    index = source_rows if gather else destination_rows
    count = int(index.numel())
    if count == 0:
        return
    width = source.shape[1]
    if destination.shape[1] != width or source.dtype != destination.dtype:
        raise ValueError("source and destination rows differ in width or dtype")
    if source.stride(1) != 1 or destination.stride(1) != 1:
        raise ValueError("rows must be contiguous")
    if not (source.is_cuda or destination.is_cuda):
        if gather:
            destination[:count].copy_(source.index_select(0, index))
        else:
            destination.index_copy_(0, index, source[:count])
        return
    block = 1024
    _kernel()[(count, -(-width // block))](
        source, destination, index, width, source.stride(0),
        destination.stride(0), GATHER=gather, BLOCK=block)


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def move_rows(source_ptr, destination_ptr, index_ptr, width,
                  source_stride, destination_stride,
                  GATHER: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        part = tl.program_id(1).to(tl.int64)
        other = tl.load(index_ptr + row).to(tl.int64)
        offsets = part * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < width
        if GATHER:
            read, write = other, row
        else:
            read, write = row, other
        values = tl.load(source_ptr + read * source_stride + offsets, mask=mask)
        tl.store(destination_ptr + write * destination_stride + offsets, values,
                 mask=mask)

    _KERNEL = move_rows
    return _KERNEL
