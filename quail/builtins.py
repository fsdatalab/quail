"""The registry of Quail's built in backends, models, and runtimes."""

import _quail_native

from quail.backends import QuailBackend
from quail.backends.quail import quail_runtimes
from quail.baselines import register_baselines
from quail.execution.runtimes import built_in_runtimes
from quail.extensions import ExtensionRegistry, LogicalRule, PhysicalRule
from quail.specs import DEVICES, MODELS

# the logical optimizer rules, in order: DataFusion's default rules with
# Quail's before replace_distinct_aggregate; claim_model_regions always
# runs after them
LOGICAL_RULES = tuple(_quail_native.LOGICAL_RULES)
# the built-in physical rules, in the order they apply
PHYSICAL_RULES = tuple(_quail_native.PHYSICAL_RULES)

def built_in_registry(logical_rules=LOGICAL_RULES,
                      physical_rules=None) -> ExtensionRegistry:
    """Create one registry with Quail built ins.

    Args:
        logical_rules: The names of the logical rules to register, in the
            order they apply; leaving a rule out turns it off.
        physical_rules: The names of the physical rules to register, in
            the order they apply; leaving a rule out turns it off. None
            registers PHYSICAL_RULES as the module holds it when called.
    """
    registry = ExtensionRegistry()
    for model in MODELS.values():
        registry.register_model(model)
    for device in DEVICES.values():
        registry.register_device(device)
    registry.register_backend(QuailBackend())
    for name in logical_rules:
        registry.register_logical_rule(LogicalRule(name))
    for name in PHYSICAL_RULES if physical_rules is None else physical_rules:
        registry.register_physical_rule(PhysicalRule(name))
    for runtimes in (built_in_runtimes(), quail_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime, key=runtime_key)
    # the comparison backends, for the benchmark runner and the server
    return register_baselines(registry)
