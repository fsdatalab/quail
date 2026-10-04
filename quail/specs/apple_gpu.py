"""The Apple GPU of the Mac this process runs on.

Macs differ in memory and speed, and an Apple GPU shares its memory
with the rest of the machine, so this spec is built from the local
machine when a Session asks for the device. MLX reports the machine.

Sources of the constants:

- mem_bytes is a fraction of the working set Metal recommends for the
  GPU, DEFAULT_MEMORY_FRACTION unless the session's EngineConfig gives
  memory_fraction. It is everything Quail holds: weights, KV pools, and
  the buffers MLX keeps for reuse. On an M4 with 34.4 GB, Metal
  recommends 22.9 GB. At 0.5 a two-filter chain over 200 documents and a
  join of 200 documents with 6 evicted no KV. At 0.9 the same chain took
  the same time, and macOS compressed 8.6 GB of other applications'
  memory to make room.
- peak_flops is measured: the bf16 projections of
  Decision-2.0-Kai-0.6B ran at 2.33e12 FLOP/s on an M4 with a 10-core
  GPU at 2,048 rows per chunk. Other chips use the same number.
- hbm_bw is Apple's published memory bandwidth of the M4, 120 GB/s.
  It was not measured.
- chunk_cap_tokens is measured on the same M4: chunks of 8,192 tokens
  ran at 2,353 fresh tokens per second, compared with 2,245 at 2,048
  tokens and 2,180 at 16,384.
"""

from .base import DeviceSpec

APPLE_GPU = "apple-gpu"
DEFAULT_MEMORY_FRACTION = 0.5
M4_BF16_FLOPS = 2.33e12
M4_MEMORY_BANDWIDTH = 120e9
CHUNK_CAP_TOKENS = 8192


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
    recommended = mx.device_info()["max_recommended_working_set_size"]
    return DeviceSpec(
        name=APPLE_GPU,
        mem_bytes=float(recommended) * fraction,
        hbm_bw=M4_MEMORY_BANDWIDTH,
        peak_flops=M4_BF16_FLOPS,
        bf16_flops=M4_BF16_FLOPS,
        implementation="mlx",
        chunk_cap_tokens=CHUNK_CAP_TOKENS,
    )
