"""The MLX device implementation, on Apple silicon.

Chunk inputs become MLX arrays, the KV pools are MLX arrays that
vllm-metal's kernels write and read in place, and the readouts are the
Decision 2.0 ones. The forward pass is executor.models.qwen3_mlx.
"""

import numpy as np

from quail.backends.quail.executor.device import DeviceImplementation
from quail.backends.quail.executor.mlx_device.pools import MlxKVPools
from quail.backends.quail.executor.mlx_device.readout import MlxDecisionChoices


class MlxImplementation(DeviceImplementation):
    """Chunk staging, waiting, pools, and readouts with MLX."""

    name = "mlx"

    def __init__(self):
        import mlx.core as mx

        self.mx = mx

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

    def synchronize(self) -> None:
        self.mx.synchronize()

    def peak_memory_bytes(self) -> int:
        return int(self.mx.get_peak_memory())

    # ---- readouts

    def decision_choices(self, head, offsets):
        return MlxDecisionChoices(head, offsets)
