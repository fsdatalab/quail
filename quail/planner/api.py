"""Select backend plans and apply the registered physical rules."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.physical_optimizer import (
    ModelRegion,
    PlanningContext,
    apply_physical_rules,
)
from quail.planner.plan import Refusal, score_seconds
from quail.planner.validation import ClassifyRefusedError
from quail.specs import DeviceSpec, ModelSpec


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry=None, tokenizer=None, pair_fractions=None,
               scan_fractions=None,
               canvas_draws: int = 4,
               attention: str | None = None, memo=None):
    """Plan one query with the selected model backend.

    Args:
        plan: The logical plan to plan, as the logical rules left it.
        model: Model spec.
        device: Device spec.
        doc_tokens: Per document token counts for each table alias.
        gpus: GPU count handed to the backend as gpu_count.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        backend: Registered model backend name.
        canvas_draws: Maximum diffusion draws per individual-document
            classification or score. One disables repeated draws.
        attention: An attention path, "tree" or "unified", forced for
            every filter and join; None lets the planner choose.
        registry: Optional session extension registry.
        tokenizer: Optional callable (text -> token list) handed to the
            planning context.
        scan_fractions: alias -> the fraction of its documents the
            column tests are expected to keep.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        memo: Results the logical rules computed for this plan, such
            as its statistics, for the physical planner to reuse.

    Returns:
        A PhysicalPlan, or a Refusal explaining why the query cannot run.
    """
    if canvas_draws < 1:
        return Refusal(
            reasons=(f"canvas_draws must be at least 1, got {canvas_draws}",),
            constraint="canvas_draws", needed=1, available=canvas_draws)
    if registry is None:
        # the built in registry imports every backend, and backends
        # import this planner; build it only when no session gave one
        from quail.builtins import built_in_registry
        registry = built_in_registry()
    try:
        selected = registry.backend(backend)
    except ValueError as error:
        return Refusal(
            reasons=(str(error),),
            constraint="unknown_backend",
            needed=1,
            available=0,
            unit="backends",
        )
    support = selected.supports(model, device, gpus)
    if not support.supported:
        return Refusal(
            reasons=(support.reason or "unsupported backend configuration",),
            constraint="unsupported_backend_configuration",
            needed=1,
            available=0,
            unit="configurations",
        )

    context = PlanningContext(
        model=model,
        device=device,
        gpu_count=gpus,
        document_tokens=doc_tokens,
        backend=backend,
        order=order,
        canvas_draws=canvas_draws,
        attention=attention,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
        scan_fractions=dict(scan_fractions or {}),
        logical_plan=plan,
        memo={} if memo is None else memo,
    )
    region = ModelRegion(plan)
    candidates = tuple(selected.plan(region, context))
    for physical_planner in registry.physical_planners.values():
        candidates += tuple(physical_planner.plan(region, context))
    if not candidates:
        return Refusal(
            reasons=(f"backend {backend!r} could not produce a plan",),
            constraint="no_physical_plan",
            needed=1,
            available=0,
            unit="plans",
        )
    # min keeps the first of equal candidates, so a backend lists the
    # plan as written first
    selected_candidate = min(
        candidates,
        key=lambda candidate: candidate.estimated_seconds,
    )
    selected_plan = selected_candidate.plan
    if isinstance(selected_plan, Refusal):
        return selected_plan
    if selected_plan.backend != backend:
        raise ValueError(
            f"physical planner returned backend {selected_plan.backend!r} "
            f"for selected backend {backend!r}")
    if selected_candidate.logical_plan is not None:
        context = replace(context, logical_plan=selected_candidate.logical_plan)
    return apply_plan_rules(
        selected_plan, tuple(registry.physical_rules.values()), context)


def refine_plan(plan, *, model: ModelSpec, device: DeviceSpec,
                doc_tokens: dict, gpus: int = 1, backend: str = "quail",
                registry=None, order: str | None = None,
                tokenizer=None, pair_fractions=None,
                canvas_draws: int = 4,
                attention: str | None = None):
    """Run the physical rules again over a plan once its inputs are exact.

    A plan made on estimated document lengths never saw the token
    stores, which some rules read (prefix_sharing measures the shared
    prefixes of a corpus). Takes the same inputs as plan_query and
    returns the plan with any rule's rewrite applied.
    """
    if isinstance(plan, Refusal):
        return plan
    if registry is None:
        from quail.builtins import built_in_registry
        registry = built_in_registry()
    context = PlanningContext(
        model=model,
        device=device,
        gpu_count=gpus,
        document_tokens=doc_tokens,
        backend=backend,
        order=order,
        canvas_draws=canvas_draws,
        attention=attention,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
    )
    return apply_plan_rules(
        plan, tuple(registry.physical_rules.values()), context)


def apply_plan_rules(plan, rules, context):
    """Apply physical rules in order to a plan and return the result.

    The rules see the plan's settings on the context and may add to
    them. A rule that re-estimates a classification or score moves the
    plan's estimated seconds by the same amount.

    Args:
        plan: A PhysicalPlan.
        rules: The physical rules, in the order to apply them.
        context: The PlanningContext.

    Returns:
        The rewritten PhysicalPlan, or a Refusal when a classification
        no rule can score refuses the plan.
    """
    context = replace(context, settings=dict(plan.settings))
    try:
        graph, changed = apply_physical_rules(plan.graph, rules, context)
    except ClassifyRefusedError as refused:
        return refused.refusal()
    if not changed and context.settings == dict(plan.settings):
        return plan
    # a rule that re-estimates a classification or score moves the
    # plan's total by the same amount
    seconds = plan.estimated_seconds + (
        score_seconds(graph.nodes) - score_seconds(plan.nodes))
    return replace(
        plan, nodes=graph.nodes, root=graph.root, estimated_seconds=seconds,
        settings=context.settings)
