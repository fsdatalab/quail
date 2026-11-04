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
    """
    model: str
    device: str
    gpus: int = 1
    backend: str = "quail"
    # sum the CUDA event pair each forward chunk records into gpu_s;
    # off by default so a run never pays for a measurement it does not read
    gpu_timing: bool = False
    # the most noise draws a diffusion model averages per AI.CLASSIFY
    # letters read and per one-table AI.SCORE; an answer whose first
    # draw is uncertain takes the rest. An autoregressive model reads
    # one answer whatever this says
    canvas_draws: int = 4
    # one attention path, "tree" or "unified", forced for every filter
    # and join; None lets the planner choose per node
    attention: str | None = None
