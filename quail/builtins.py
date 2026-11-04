"""The registry of Quail's built in backends, models, and runtimes."""

from quail.backends import QuailBackend
from quail.backends.quail import quail_runtimes
from quail.baselines import register_baselines
from quail.execution.runtimes import built_in_runtimes
from quail.extensions import ExtensionRegistry, LogicalRule, PhysicalRule
from quail.physical import built_in_codecs
from quail.planner.native import LOGICAL_RULES, PHYSICAL_RULES
from quail.specs import DEVICES, MODELS


def built_in_registry() -> ExtensionRegistry:
    """Create one registry with Quail built ins."""
    registry = ExtensionRegistry()
    for model in MODELS.values():
        registry.register_model(model)
    for device in DEVICES.values():
        registry.register_device(device)
    registry.register_backend(QuailBackend())
    for name in LOGICAL_RULES:
        registry.register_logical_rule(LogicalRule(name))
    for name in PHYSICAL_RULES:
        registry.register_physical_rule(PhysicalRule(name))
    for codec in built_in_codecs():
        registry.register_codec(codec)
    for runtimes in (built_in_runtimes(), quail_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime, key=runtime_key)
    # the comparison backends, for the benchmark runner and the server
    return register_baselines(registry)
