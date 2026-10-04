"""One forward pass per device implementation and model architecture.

ModelSpec.arch names the architecture; build_pipeline picks its forward
pass for a device implementation. Adding an architecture is adding a
file and a registry line.
"""

from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.executor.models.diffusion_gemma import (
    DiffusionGemmaPipeline,
)
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline
from quail.backends.quail.executor.models.qwen3_mlx import MlxQwen3Pipeline

# (device implementation name, architecture) -> forward pass
PIPELINES = {("cuda", "qwen3"): Qwen3Pipeline,
             ("cuda", "diffusion_gemma"): DiffusionGemmaPipeline,
             ("mlx", "qwen3"): MlxQwen3Pipeline}


def supported_archs(implementation: str = "cuda") -> frozenset:
    """Return the architectures a device implementation has a forward pass for."""
    return frozenset(arch for name, arch in PIPELINES if name == implementation)


def build_pipeline(spec, model, arena, *, implementation: str = "cuda",
                   **kwargs) -> ModelPipeline:
    """Build the forward pass for spec.arch over a loaded model and arena.

    Args:
        spec: The model's ModelSpec.
        model: The loaded model, as the device implementation holds it.
        arena: The KVArena the forward pass reads and writes.
        implementation: The device implementation's name.
        **kwargs: Passed to the forward pass's constructor.

    Raises:
        ValueError: The implementation has no forward pass for spec.arch.
    """
    try:
        cls = PIPELINES[implementation, spec.arch]
    except KeyError:
        raise ValueError(
            f"no {implementation} forward pass for architecture "
            f"{spec.arch!r}; known: {sorted(supported_archs(implementation))}"
        ) from None
    return cls(model, arena, spec=spec, **kwargs)


__all__ = ["ModelPipeline", "PIPELINES", "build_pipeline", "supported_archs"]
