"""The registry of Quail's built in backends, models, and runtimes."""

from quail.backends import QuailBackend
from quail.backends.quail import quail_runtimes
from quail.baselines import register_baselines
from quail.execution.runtimes import built_in_runtimes
from quail.extensions import ExtensionRegistry, LogicalRule, PhysicalRule
from quail.specs import DEVICES, MODELS

# the logical optimizer rules, in order, as quail_datafusion::LOGICAL_RULES
# names them: DataFusion's default rules with Quail's eleven before
# replace_distinct_aggregate; claim_model_regions always runs after them
LOGICAL_RULES = (
    "rewrite_set_comparison", "optimize_unions", "unions_to_filter",
    "simplify_expressions",
    "eliminate_distinct", "place_ai_nodes", "absorb_into_ai_nodes",
    "merge_repeated_ai_calls", "push_down_distinct", "push_down_limit_into_ai_nodes",
    "push_down_projection_into_ai_nodes", "push_down_case_guards",
    "push_down_substring_prechecks", "push_down_semi_join_fetch",
    "push_down_count_threshold",
    "replace_distinct_aggregate", "eliminate_join",
    "decorrelate_predicate_subquery", "scalar_subquery_to_join",
    "decorrelate_lateral_join", "extract_equijoin_predicate",
    "eliminate_duplicated_expr", "eliminate_filter", "eliminate_cross_join",
    "eliminate_limit", "propagate_empty_relation", "filter_null_join_keys",
    "eliminate_outer_join", "push_down_limit", "push_down_filter",
    "single_distinct_aggregation_to_group_by", "eliminate_group_by_constant",
    "common_sub_expression_eliminate", "extract_leaf_expressions",
    "push_down_leaf_projections", "optimize_projections",
)
# the built-in physical rules, in the order they apply
PHYSICAL_RULES = (
    "should_push_down_limit", "choose_classify_implementation", "pack_set_passes",
    "join_document_ends", "should_retain_kv", "should_share_prefixes",
    "choose_attention_path", "should_deduplicate",
)


def built_in_registry(logical_rules=LOGICAL_RULES) -> ExtensionRegistry:
    """Create one registry with Quail built ins.

    Args:
        logical_rules: The names of the logical rules to register, in the
            order they apply; leaving a rule out turns it off.
    """
    registry = ExtensionRegistry()
    for model in MODELS.values():
        registry.register_model(model)
    for device in DEVICES.values():
        registry.register_device(device)
    registry.register_backend(QuailBackend())
    for name in logical_rules:
        registry.register_logical_rule(LogicalRule(name))
    for name in PHYSICAL_RULES:
        registry.register_physical_rule(PhysicalRule(name))
    for runtimes in (built_in_runtimes(), quail_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime, key=runtime_key)
    # the comparison backends, for the benchmark runner and the server
    return register_baselines(registry)
