"""The engine configuration a session runs on."""

from dataclasses import dataclass

from quail.physical.plan import Refusal
from quail.specs import MODELS


def resolve_model(name: str, models=None):
    """Return a ModelSpec by name, or a Refusal if unknown."""
    if models is None:
        models = MODELS
    if name in models:
        return models[name]
    return Refusal(
        reasons=(f"unknown model {name!r}; registered models: "
                 f"{sorted(models)}",),
        constraint="unknown_model", needed=1, available=0, unit="specs")


@dataclass(frozen=True, kw_only=True)
class EngineConfig:
    """Top-level engine configuration.

    `model` and `device` have no defaults: they name the hardware and
    weights the query runs on. `gpus` counts model copies on one host.
    `host_kv_bytes` is the host memory that keeps document KV the arena
    frees, so a later operator or query reads it back instead of
    computing it; 0 keeps none.
    """
    model: str
    device: str
    gpus: int = 1
    backend: str = "quail"
    gpu_timing: bool = False
    canvas_draws: int = 4
    attention: str | None = None
    host_kv_bytes: int = 0
