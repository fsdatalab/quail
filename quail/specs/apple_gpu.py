"""The Apple GPU of the Mac this process runs on.

Macs differ in memory and speed, and an Apple GPU shares its memory
with the rest of the machine, so this spec is built from the local
machine when a Session asks for the device. MLX reports the machine.

The constants were measured on an Apple M4 with a 10-core GPU and
34.4 GB of memory, on battery with Low Power Mode off, running
Decision-2.0-Kai-0.6B in bf16. A Mac with another chip uses the same
constants, and its spec carries a note saying so.

- mem_bytes is a fraction of the working set Metal recommends for the
  GPU, DEFAULT_MEMORY_FRACTION unless the session's EngineConfig gives
  memory_fraction. It is everything Quail holds: weights, KV pools, and
  the buffers MLX keeps for reuse. Metal recommends 22.9 GB on the M4.
  At 0.5, 0.25, and 0.125 a two-filter chain over 200 documents and a
  join of 200 documents with 6 gave the same answers and evicted no KV.
- peak_flops: the model's bf16 projections ran at 2.33e12 FLOP/s at
  2,048 rows per chunk.
- hbm_bw: scaling and adding arrays of 0.5 to 2 GB moved 94e9 to 98e9
  bytes per second, compared with the 120e9 Apple publishes for the M4.
- chunk_cap_tokens: chunks of 8,192 tokens ran at 2,353 fresh tokens
  per second, compared with 2,245 at 2,048 tokens and 2,180 at 16,384.
- scratch_bytes: beside the weights and the KV pools, the same chain
  held at most 0.44 GB with chunks of 2,048 tokens, 0.57 GB at 4,096,
  0.67 GB at 8,192, and 0.94 GB at 16,384. That is about 0.37 GB and
  35,000 bytes per chunk token. The model's activation reserve is
  65,536 bytes per chunk token and has no fixed part.
"""

from .base import DeviceSpec

APPLE_GPU = "apple-gpu"
DEFAULT_MEMORY_FRACTION = 0.5
MEASURED_CHIP = "Apple M4"
M4_BF16_FLOPS = 2.33e12
M4_MEMORY_BANDWIDTH = 95e9
CHUNK_CAP_TOKENS = 8192
SCRATCH_BYTES = 0.4e9


def local_apple_gpu(memory_fraction: float | None = None) -> DeviceSpec:
    """Build the spec of this Mac's GPU.

    Args:
        memory_fraction: The fraction of the working set Metal
            recommends that Quail may hold; None takes
            DEFAULT_MEMORY_FRACTION.

    Raises:
        RuntimeError: mlx is not installed, as on any other platform.
        ValueError: The fraction is not above 0 and at most 1.
    """
    fraction = DEFAULT_MEMORY_FRACTION if memory_fraction is None else memory_fraction
    if not 0 < fraction <= 1:
        raise ValueError(
            f"memory_fraction must be above 0 and at most 1, not {fraction}")
    try:
        import mlx.core as mx
    except ImportError as error:
        from quail.backends.quail.executor.mlx_device.kernels import INSTALL_TEXT

        raise RuntimeError(INSTALL_TEXT) from error
    info = mx.device_info()
    return DeviceSpec(
        name=APPLE_GPU,
        mem_bytes=float(info["max_recommended_working_set_size"]) * fraction,
        hbm_bw=M4_MEMORY_BANDWIDTH,
        peak_flops=M4_BF16_FLOPS,
        bf16_flops=M4_BF16_FLOPS,
        implementation="mlx",
        chunk_cap_tokens=CHUNK_CAP_TOKENS,
        scratch_bytes=SCRATCH_BYTES,
        notes=chip_notes(info.get("device_name", "")),
    )


def chip_notes(chip: str) -> tuple[str, ...]:
    """Return the note for a chip whose speed was not measured.

    Args:
        chip: The chip name MLX reports, such as "Apple M4 Max".
    """
    if chip == MEASURED_CHIP:
        return ()
    return (f"{APPLE_GPU} speed constants were measured on an {MEASURED_CHIP}; "
            f"this Mac has {chip or 'another chip'}, so estimated times use "
            "the M4's speed",)
