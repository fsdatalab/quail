"""The MLX device implementation, on Apple silicon.

Chunk inputs become MLX arrays, the KV pools are MLX arrays that
vllm-metal's kernels write and read in place, and the readouts are the
Decision 2.0 ones. The forward pass is executor.models.qwen3_mlx.
"""

import time

import numpy as np

from quail.backends.quail.executor.device import DeviceImplementation
from quail.backends.quail.executor.mlx_device.kernels import INSTALL_TEXT, metal_ops
from quail.backends.quail.executor.mlx_device.pools import MlxKVPools
from quail.backends.quail.executor.mlx_device.readout import MlxDecisionChoices


class MlxImplementation(DeviceImplementation):
    """Chunk staging, waiting, pools, and readouts with MLX."""

    name = "mlx"

    def __init__(self):
        import mlx.core as mx

        self.mx = mx
        self.timing = False
        self._limits_before = None

    @staticmethod
    def problem() -> str | None:
        """Return why this process cannot run MLX models, or None when it can."""
        try:
            import mlx.core  # noqa: F401

            metal_ops()
        except (ImportError, RuntimeError):
            return INSTALL_TEXT
        return None

    def kv_pools(self, dtype=None) -> MlxKVPools:
        """Return unbuilt KV pools; bf16 when no dtype is given."""
        return MlxKVPools(dtype or self.mx.bfloat16)

    # ---- chunk inputs

    def stage(self, values, dtype, name=None, staging=None):
        return self.mx.array(np.asarray(values, dtype=dtype))

    def stage_tokens(self, ids, staging=None):
        return self.mx.array(ids)

    def select_rows(self, rows, index):
        mx = self.mx
        if not isinstance(rows, mx.array):
            return super().select_rows(rows, index)
        if not isinstance(index, mx.array):
            index = mx.array(np.asarray(index, dtype=np.int64))
        return rows[index]

    # ---- timing and memory

    def time_chunks(self, enabled: bool) -> None:
        self.timing = enabled

    def record_event(self):
        # MLX computes lazily, so a mark is exact only once the work
        # before it has finished. Waiting stops the host from packing
        # the next chunk while the GPU runs, so it waits only when the
        # query reads chunk times.
        if self.timing:
            self.mx.synchronize()
        return time.perf_counter()

    def synchronize(self) -> None:
        self.mx.synchronize()

    def peak_memory_bytes(self) -> int:
        return int(self.mx.get_peak_memory())

    def hold_within(self, budget_bytes: float) -> None:
        """Keep everything MLX holds inside a budget, and keep it in memory.

        Call it once the weights and the KV pools are loaded. MLX keeps
        the buffers it frees and reuses one only for a request of the
        same size. Chunks differ in size, so without a limit the kept
        buffers grow to many times a chunk's own. The limit is what the
        budget leaves after the memory in use now.

        The whole budget is wired, so macOS neither compresses nor
        swaps it while other applications want memory.

        Args:
            budget_bytes: Everything Quail may hold on the device.
        """
        mx = self.mx
        # MLX must not change the wired limit while it evaluates
        mx.synchronize()
        wired = mx.set_wired_limit(int(budget_bytes))
        kept = mx.set_cache_limit(
            max(0, int(budget_bytes) - mx.get_active_memory()))
        if self._limits_before is None:
            self._limits_before = wired, kept

    def release_limits(self) -> None:
        """Give MLX back the limits it had before hold_within."""
        if self._limits_before is None:
            return
        mx = self.mx
        mx.synchronize()
        wired, kept = self._limits_before
        mx.set_wired_limit(wired)
        mx.set_cache_limit(kept)
        self._limits_before = None

    # ---- readouts

    def decision_choices(self, head, offsets):
        return MlxDecisionChoices(head, offsets)
