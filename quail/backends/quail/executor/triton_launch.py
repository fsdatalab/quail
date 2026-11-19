"""Cached-handle Triton kernel launcher and Hopper/Blackwell PDL helper.

Bypasses Triton's per-call Python binder and cache-key computation after first
compilation while preserving pointer-alignment and specialization invariants.
"""

_COMPILED_CACHE = {}
_PDL_CAPABLE = {}
_SM_COUNT = {}

# Maximum row count for enabling programmatic dependent launch (PDL) on
# general norm and gate kernels; larger prefill launches gain negligible
# prologue overlap and can suffer register pressure.
PDL_MAX_ROWS = 512

# Module switches toggled by differential GPU evaluators.
USE_FAST_LAUNCH = True
USE_PDL = True


def _device_index(device) -> int:
    """Return the integer CUDA device index for a torch.device."""
    if device.index is not None:
        return int(device.index)
    import torch

    return int(torch.cuda.current_device())


def num_multiprocessors(device) -> int:
    """Return and cache the multiprocessor count of device."""
    index = _device_index(device)
    count = _SM_COUNT.get(index)
    if count is None:
        import torch

        count = int(torch.cuda.get_device_properties(index).multi_processor_count)
        _SM_COUNT[index] = count
    return count


def pdl_enabled(device, rows=None) -> bool:
    """Return whether PDL is enabled, supported on device, and helpful for rows.

    Args:
        device: CUDA torch.device of the tensors being launched.
        rows: Optional row count; when greater than PDL_MAX_ROWS, returns False.

    Returns:
        True when Hopper/Blackwell PDL should be used for the launch.
    """
    if not USE_PDL or (rows is not None and rows > PDL_MAX_ROWS):
        return False
    index = _device_index(device)
    capable = _PDL_CAPABLE.get(index)
    if capable is None:
        import torch

        capable = torch.cuda.get_device_capability(index)[0] >= 9
        _PDL_CAPABLE[index] = capable
    return capable


def _arg_signature(args, *, exact_args: bool) -> tuple:
    """Return dtype, alignment, and integer ABI/divisibility classes for args."""
    signature = []
    for value in args:
        if hasattr(value, "data_ptr"):
            if exact_args:
                signature.append((value.dtype, (value.data_ptr() & 15) == 0))
            else:
                signature.append(value.dtype)
        elif isinstance(value, int) and exact_args:
            wide = value < -0x80000000 or value > 0x7FFFFFFF
            signature.append((value == 1, (value & 15) == 0, wide))
    return tuple(signature)


def _tensor_device_index(args) -> int:
    """Return the CUDA device index of the first tensor in args."""
    for value in args:
        device = getattr(value, "device", None)
        if device is not None and device.type == "cuda":
            return _device_index(device)
    import torch

    return int(torch.cuda.current_device())


def launch_triton(kernel, grid, args, *, key, kwargs, exact_args=True):
    """Launch a Triton kernel via its cached compiled handle when safe.

    Falls back to standard Triton dispatch when USE_FAST_LAUNCH is False or
    when Triton debug mode or launch hooks are active.

    Args:
        kernel: Triton JITFunction.
        grid: 1-, 2-, or 3-tuple launch grid.
        args: Positional runtime arguments (tensors and scalars).
        key: Hashable specialization key covering every constexpr and launch
            option passed in kwargs.
        kwargs: Constexprs and launch options forwarded to Triton on cache miss.
        exact_args: When True, include pointer 16-byte alignment and integer
            divisibility/width in the cache key. Set False only when every
            argument whose alignment or divisibility can vary is excluded from
            Triton specialization (and all remaining tensors keep fixed aligned
            layouts by caller contract).

    Returns:
        The compiled kernel on cache miss or fallback, or None on fast hit.
    """
    from triton import knobs
    from triton.runtime import driver

    if (not USE_FAST_LAUNCH or knobs.runtime.debug
            or knobs.runtime.launch_enter_hook is not None
            or knobs.runtime.launch_exit_hook is not None
            or knobs.runtime.add_stages_inspection_hook is not None):
        return kernel[grid](*args, **kwargs)
    device = _tensor_device_index(args)
    cache_key = (
        device, id(kernel), key,
        _arg_signature(args, exact_args=exact_args),
    )
    entry = _COMPILED_CACHE.get(cache_key)
    if entry is None:
        compiled = kernel[grid](*args, **kwargs)
        _COMPILED_CACHE[cache_key] = (
            compiled.run, compiled.function, compiled.packed_metadata,
            tuple(sorted(kwargs.items())))
        return compiled
    run, function, metadata, _saved_kwargs = entry
    stream = driver.active.get_current_stream(device)
    g0 = grid[0]
    g1 = grid[1] if len(grid) > 1 else 1
    g2 = grid[2] if len(grid) > 2 else 1
    run(g0, g1, g2, stream, function, metadata, None, None, None, *args)
    return None
