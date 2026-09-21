"""Speed of light calculation shared by the planner and reports.

The workload code counts tokens, attention pairs, and KV movement. The
dense decoder cost code turns those counts into named model
components. The generic roofline code prices each component
separately and adds their times in execution order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quail.cost import work as _workload
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.roofline import ComponentLatency, component_latencies
from quail.cost.vision_cost import vision_components
from quail.specs import DeviceSpec, ModelSpec


def _latencies(work: _workload.Work, model: ModelSpec, device: DeviceSpec,
               passes: float) -> tuple[ComponentLatency, ...]:
    """Return the priced components for one work record.

    The vision tower runs before the decoder. A record with no image
    work has no tower components, so a text query's list is unchanged.
    """
    return component_latencies(
        vision_components(work, model)
        + dense_decoder_components(work, model, passes),
        device)


def compute_seconds(work: _workload.Work, model: ModelSpec,
                    device: DeviceSpec) -> float:
    """Return ideal compute time without memory movement."""
    return sum(component.compute_seconds
               for component in _latencies(
                   work, model, device, passes=0.0))


def unrounded_seconds(work: _workload.Work, model: ModelSpec, device: DeviceSpec,
                      chunk_tokens: int) -> float:
    """Return component time with fractional ideal forward passes."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    passes = work.tokens / chunk_tokens
    return sum(component.seconds
               for component in _latencies(work, model, device, passes))


def prefix_recompute_seconds(prefix_tokens: int, model: ModelSpec,
                             device: DeviceSpec) -> float:
    """Return ideal compute time for one document prefix."""
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be nonnegative")
    return compute_seconds(
        _workload.scan(prefix_tokens, 0, window=model.sliding_window),
        model,
        device,
    )


@dataclass(frozen=True)
class SpeedOfLight:
    """The complete bound and its component breakdown."""

    work: _workload.Work
    passes: int
    components: tuple[ComponentLatency, ...]

    def component(self, name: str) -> ComponentLatency:
        """Return one named component result."""
        return next(component for component in self.components
                    if component.name == name)

    @property
    def bytes_moved(self) -> float:
        return sum(component.bytes_moved for component in self.components)

    @property
    def compute(self) -> float:
        return sum(component.compute_seconds for component in self.components)

    @property
    def memory(self) -> float:
        return sum(component.memory_seconds for component in self.components)

    @property
    def seconds(self) -> float:
        return sum(component.seconds for component in self.components)

    @property
    def bound_by(self) -> str:
        limits = {component.bound_by for component in self.components}
        return limits.pop() if len(limits) == 1 else "mixed"

    def explain(self) -> str:
        """Return a plain text component breakdown."""
        work = self.work
        lines = [
            f"tokens         {work.tokens:>18,.0f}",
            f"pairs          {work.pairs:>18,.0f}",
            f"kv written     {work.kv_written:>18,.0f}",
            f"kv read        {work.kv_read:>18,.0f}",
            f"forward passes {self.passes:>18,d}",
        ]
        if work.image_patches:
            lines.extend([
                f"image patches  {work.image_patches:>18,.0f}",
                f"image pairs    {work.image_pairs:>18,.0f}",
                f"soft tokens    {work.image_soft_tokens:>18,.0f}",
            ])
        lines.append(f"bytes moved    {self.bytes_moved:>18,.0f}")
        for latency in self.components:
            name = latency.name
            lines.extend([
                f"{name} compute {latency.compute_seconds:>16.4f} s",
                f"{name} memory  {latency.memory_seconds:>16.4f} s",
                f"{name} time    {latency.seconds:>16.4f} s",
            ])
        lines.extend([
            f"T_compute total{self.compute:>18.4f} s",
            f"T_memory total {self.memory:>18.4f} s",
            f"SoL            {self.seconds:>18.4f} s ({self.bound_by} bound)",
        ])
        return "\n".join(lines)


def speed_of_light(work: _workload.Work, model: ModelSpec, device: DeviceSpec,
                   chunk_tokens: int) -> SpeedOfLight:
    """Price aggregate query work with ideal query-wide packing."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    passes = math.ceil(work.tokens / chunk_tokens) if work.tokens else 0
    return SpeedOfLight(
        work=work,
        passes=passes,
        components=_latencies(work, model, device, passes),
    )
