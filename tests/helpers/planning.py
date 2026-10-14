"""Plan as a session would: the logical rules, then the physical planner."""

from quail.builtins import built_in_registry
from quail.planner.api import plan_query
from quail.planner.optimizer.logical import (
    LogicalPlanningContext,
    apply_logical_rules,
)
from quail.planner.rules.logical import built_in_logical_rules
from quail.specs import H100_SXM, QWEN3_4B_FP8


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def tok(text):
    return text.split()


def _optimize(logical, doc_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Run the built in logical rules as a session would.

    Returns:
        The optimized plan and the names of the rules that changed it.
    """
    context = LogicalPlanningContext(
        None, None, model=model, device=H100_SXM,
        gpu_count=kwargs.get("gpus", 1), document_tokens=doc_tokens,
        order=kwargs.get("order"), tokenizer=tok,
        pair_fractions=kwargs.get("pair_fractions") or {})
    return apply_logical_rules(logical, built_in_logical_rules(), context)


def _plan(logical, doc_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Plan as a session would: the logical rules, then the physical planner."""
    optimized, _ = _optimize(logical, doc_tokens, model, **kwargs)
    return plan_query(optimized, model=model, device=H100_SXM,
                      registry=built_in_registry(),
                      doc_tokens=doc_tokens, tokenizer=tok, **kwargs)
