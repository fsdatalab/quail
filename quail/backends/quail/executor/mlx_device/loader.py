"""Load a Decision 2.0 package's backbone and head as MLX arrays."""

import json
from pathlib import Path

from quail.backends.quail.executor.mlx_device.readout import MlxDecisionHead
from quail.backends.quail.executor.mlx_device.weights import (
    Qwen3Config,
    Qwen3Weights,
)
from quail.backends.quail.executor.model import is_decision2


def load_decision_backbone(path, dtype) -> Qwen3Weights:
    """Load the Qwen3 backbone of a Decision 2.0 package.

    The package keeps its backbone in backbone/. The weights are
    evaluated before this returns, so loading them is part of startup.

    Args:
        path: The local package directory.
        dtype: MLX dtype every weight is cast to.

    Raises:
        ValueError: The directory is not a Decision 2.0 package.
    """
    import mlx.core as mx

    if not is_decision2(path):
        raise ValueError(f"{path} is not a Decision 2.0 package")
    backbone = Path(path) / "backbone"
    config = Qwen3Config.from_dict(
        json.loads((backbone / "config.json").read_text()))
    tensors = {}
    for file in sorted(backbone.glob("*.safetensors")):
        tensors.update(mx.load(str(file)))
    return Qwen3Weights.from_tensors(tensors, config, dtype)


def load_decision_head(path) -> MlxDecisionHead:
    """Load decision_head.safetensors from a Decision 2.0 package."""
    import mlx.core as mx

    return MlxDecisionHead(mx.load(str(Path(path) / "decision_head.safetensors")))
