"""Plan as a session would: the tree rewrites, then the physical planner."""

from quail.builtins import built_in_registry
from quail.planner.optimize import plan_query
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.tree_rewrites import logical_alternatives
from quail.planner.rules import built_in_rules
from quail.specs import H100_SXM, QWEN3_4B_FP8


def node_kinds(plan):
    return [type(node).__name__ for node in plan.nodes]


def tokenizer(text):
    return text.split()


def _optimize(logical, document_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Run the built in tree rewrites as a session would.

    Returns:
        The optimized plan and the names of the rules that changed it.
    """
    context = PlanningContext(
        model=model, device=H100_SXM,
        gpus=kwargs.get("gpus", 1), document_tokens=document_tokens,
        order=kwargs.get("order"), tokenizer=tokenizer,
        pair_fractions=kwargs.get("pair_fractions") or {})
    candidate, = logical_alternatives(logical, built_in_rules(), context)
    return candidate.plan, candidate.rules


def _plan(logical, document_tokens, model=QWEN3_4B_FP8, **kwargs):
    """Plan as a session would: the tree rewrites, then the physical planner."""
    return plan_query(logical, model=model, device=H100_SXM,
                      registry=built_in_registry(),
                      document_tokens=document_tokens, tokenizer=tokenizer, **kwargs)
