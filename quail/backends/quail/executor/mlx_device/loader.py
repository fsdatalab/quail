"""Load Qwen3 weights and the Decision 2.0 head as MLX arrays."""

import json
from pathlib import Path

from quail.backends.quail.executor.mlx_device.readout import MlxDecisionHead
from quail.backends.quail.executor.mlx_device.weights import (
    Qwen3Config,
    Qwen3Weights,
)
from quail.backends.quail.executor.model import is_decision2


def load_qwen3_weights(path, dtype) -> Qwen3Weights:
    """Load the Qwen3 weights in a checkpoint directory.

    A Decision 2.0 package keeps its backbone in backbone/, with tensor
    names that lack the `model.` prefix of a plain Qwen3 checkpoint.
    The weights are evaluated before this returns, so loading them is
    part of startup.

    Args:
        path: The local checkpoint directory.
        dtype: MLX dtype every weight is cast to.
    """
    import mlx.core as mx

    directory = Path(path)
    if is_decision2(directory):
        directory = directory / "backbone"
    config = Qwen3Config.from_dict(
        json.loads((directory / "config.json").read_text()))
    tensors = {}
    for file in sorted(directory.glob("*.safetensors")):
        tensors.update(mx.load(str(file)))
    prefix = "model." if "model.embed_tokens.weight" in tensors else ""
    return Qwen3Weights.from_tensors(tensors, config, dtype, prefix=prefix)


def load_decision_head(path) -> MlxDecisionHead:
    """Load decision_head.safetensors from a Decision 2.0 package."""
    import mlx.core as mx

    return MlxDecisionHead(mx.load(str(Path(path) / "decision_head.safetensors")))
