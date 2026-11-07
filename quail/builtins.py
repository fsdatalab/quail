"""The registry of Quail's built in backends, models, and runtimes."""

from quail.backends import QuailBackend
from quail.backends.quail import quail_runtimes
from quail.baselines import register_baselines
from quail.execution.runtimes import built_in_runtimes
from quail.extensions import ExtensionRegistry, LogicalRule, PhysicalRule
from quail.specs import DEVICES, MODELS

# the optimizer rules each model region runs, in order, as
# quail_datafusion::REGION_RULES names them
LOGICAL_RULES = (
    "eliminate_distinct", "place_ai_nodes", "absorb_into_ai_nodes",
    "push_down_distinct", "push_down_filter", "optimize_projections",
)
# the built-in physical rules, in the order they apply
PHYSICAL_RULES = (
    "should_push_down_limit", "choose_classify_implementation", "pack_set_passes",
    "join_document_ends", "should_retain_kv", "should_share_prefixes",
    "choose_attention_path", "should_deduplicate",
)


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
    for runtimes in (built_in_runtimes(), quail_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime, key=runtime_key)
    # the comparison backends, for the benchmark runner and the server
    return register_baselines(registry)
