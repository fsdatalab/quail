"""The registry of Quail's built in backends, models, and runtimes."""

import _quail_native

from quail.backends import QuailBackend
from quail.backends.quail import quail_runtimes
from quail.execution.runtimes import built_in_runtimes
from quail.extensions import ExtensionRegistry, LogicalRule, PhysicalRule
from quail.specs import DEVICES, MODELS

# the logical optimizer rules, in order: DataFusion's default rules with
# Quail's before replace_distinct_aggregate
LOGICAL_RULES = tuple(_quail_native.LOGICAL_RULES)
# the built-in physical rules, in the order they apply
PHYSICAL_RULES = tuple(_quail_native.PHYSICAL_RULES)
# the rules that each derive one option of an AI operator; listed in
# place of "derive", they leave the others out
DERIVE_OPTIONS = ("derive.filters", "derive.fetch", "derive.projection",
    "derive.guard")


def logical_rules_without(name: str) -> tuple[str, ...]:
    """Return LOGICAL_RULES with one rule, or one option of derive, off.

    Args:
        name: A rule in LOGICAL_RULES, or one of DERIVE_OPTIONS.
    """
    if name not in DERIVE_OPTIONS:
        return tuple(rule for rule in LOGICAL_RULES if rule != name)
    kept = tuple(option for option in DERIVE_OPTIONS if option != name)
    rules = []
    for rule in LOGICAL_RULES:
        rules.extend(kept if rule == "derive" else (rule,))
    return tuple(rules)


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
    from quail.baselines.request import RequestBackend
    from quail.baselines.vllm import (
        DefaultVLLMEngine,
        VLLMEngine,
        ngram_vllm_backend,
        stock_vllm_backend,
    )
    from quail.execution.runtimes import ModelNodeRuntime
    from quail.physical import RequestExecution

    for backend in (
        stock_vllm_backend(),
        RequestBackend(
            name="pipelined_vllm", engine=VLLMEngine(),
            filter_submission="pipelined", answer_submission="pipelined"),
        RequestBackend(
            name="shuffled_vllm", engine=VLLMEngine(),
            filter_submission="operator-at-a-time",
            answer_submission="shuffled"),
        ngram_vllm_backend(),
        RequestBackend(
            name="dumb_vllm", engine=DefaultVLLMEngine(),
            filter_submission="operator-at-a-time"),
    ):
        registry.register_backend(backend)
    registry.register_runtime(
        ModelNodeRuntime(), key=RequestExecution.runtime_key)
    return registry
