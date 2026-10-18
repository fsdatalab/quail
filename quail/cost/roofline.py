"""Generic component roofline calculation."""

from __future__ import annotations

from dataclasses import dataclass

from quail.specs import DeviceSpec, Precision


@dataclass(frozen=True)
class CostComponent:
    """One model component's arithmetic and memory work."""

    name: str
    flops: float
    bytes_moved: float
    precision: Precision


@dataclass(frozen=True)
class ComponentLatency:
    """The two limits and final time for one component."""

    component: CostComponent
    compute_seconds: float
    memory_seconds: float
    seconds: float

    @property
    def name(self) -> str:
        return self.component.name

    @property
    def flops(self) -> float:
        return self.component.flops

    @property
    def bytes_moved(self) -> float:
        return self.component.bytes_moved

    @property
    def precision(self) -> Precision:
        return self.component.precision

    @property
    def bound_by(self) -> str:
        if self.seconds > max(self.compute_seconds, self.memory_seconds) * (1 + 1e-12):
            return "mixed"
        return ("compute" if self.compute_seconds >= self.memory_seconds
                else "memory")


def component_latency(component: CostComponent,
                      device: DeviceSpec) -> ComponentLatency:
    """Price one component against the device limits."""
    compute = component.flops / device.arithmetic_bandwidth(component.precision)
    memory = component.bytes_moved / device.hbm_bw
    return ComponentLatency(component, compute, memory, max(compute, memory))


def component_latencies(
        components, device: DeviceSpec) -> tuple[ComponentLatency, ...]:
    """Price model components in execution order."""
    return tuple(component_latency(component, device)
                 for component in components)


def sum_latencies(groups) -> tuple[ComponentLatency, ...]:
    """Sum component times after pricing each dependent batch."""
    totals = {}
    for group in groups:
        for value in group:
            old = totals.get(value.name)
            if old is None:
                totals[value.name] = value
                continue
            component = CostComponent(value.name, old.flops + value.flops,
                                      old.bytes_moved + value.bytes_moved,
                                      value.precision)
            totals[value.name] = ComponentLatency(
                component, old.compute_seconds + value.compute_seconds,
                old.memory_seconds + value.memory_seconds, old.seconds + value.seconds)
    return tuple(totals.values())
