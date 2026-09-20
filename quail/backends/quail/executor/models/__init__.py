"""One forward pass per model architecture.

ModelSpec.arch names the file; build_pipeline picks it. Adding an
architecture is adding a file here and a registry line.
"""

from quail.backends.quail.executor.models.base import ModelPipeline
from quail.backends.quail.executor.models.diffusion_gemma import (
    DiffusionGemmaPipeline,
)
from quail.backends.quail.executor.models.qwen3 import Qwen3Pipeline

PIPELINES = {"qwen3": Qwen3Pipeline,
             "diffusion_gemma": DiffusionGemmaPipeline}


def supported_archs() -> frozenset:
    return frozenset(PIPELINES)


def build_pipeline(spec, model, arena, **kwargs) -> ModelPipeline:
    """Build the forward pass for spec.arch over a loaded model and arena."""
    try:
        cls = PIPELINES[spec.arch]
    except KeyError:
        raise ValueError(
            f"no forward pass for architecture {spec.arch!r}; "
            f"known: {sorted(PIPELINES)}") from None
    return cls(model, arena, spec=spec, **kwargs)


__all__ = ["ModelPipeline", "PIPELINES", "build_pipeline", "supported_archs"]
