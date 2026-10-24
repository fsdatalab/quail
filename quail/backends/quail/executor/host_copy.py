"""Move rows between the arena and pinned host memory with one kernel launch.

Pinned host memory is mapped into the device's address space, so a
kernel reads and writes it directly over PCIe. A document's rows sit
in scattered pages of many pools, so copying them with the copy
engine would take one copy per page per layer or a contiguous staging
block on the device. The kernels here take an index of source or
destination rows instead and move every row in one launch, with no
device memory beyond the index.

The pinned tensor must come from torch's pinned allocator
(pin_memory=True); its data pointer is then valid on the device.
"""


def copy_rows(source, destination, *, source_rows=None, destination_rows=None):
    """Copy rows of `source` to `destination`, one of them on the host.

    Both are 2-D (rows, width) with the same width and dtype. Exactly
    one of source_rows and destination_rows is given, an int64 device
    tensor: the rows read from the source in order (a gather), or the
    rows written in the destination in order (a scatter). The copy is
    queued on the current stream.
    """
    if (source_rows is None) == (destination_rows is None):
        raise ValueError("give the source rows or the destination rows")
    index = source_rows if source_rows is not None else destination_rows
    count = int(index.numel())
    if count == 0:
        return
    width = source.shape[1]
    if destination.shape[1] != width or source.dtype != destination.dtype:
        raise ValueError("source and destination rows differ in width or dtype")
    if not (source.is_contiguous() and destination.is_contiguous()):
        raise ValueError("source and destination must be contiguous")
    block = 1024
    _kernel()[(count, -(-width // block))](
        source, destination, index, width,
        GATHER=source_rows is not None, BLOCK=block)


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def move_rows(source_ptr, destination_ptr, index_ptr, width,
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
        values = tl.load(source_ptr + read * width + offsets, mask=mask)
        tl.store(destination_ptr + write * width + offsets, values, mask=mask)

    _KERNEL = move_rows
    return _KERNEL
