"""Search logical alternatives and select a refined physical plan."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.optimizer.alternatives import logical_alternatives
from quail.planner.optimizer.physical import (
    ModelRegion,
    PhysicalCandidate,
    PlanningContext,
    apply_physical_rules,
)
from quail.planner.plan import Refusal, inference_seconds
from quail.planner.validation import ClassifyRefusedError
from quail.specs import DeviceSpec, ModelSpec


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry, tokenizer=None, pair_fractions=None,
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
        registry: The session extension registry: its backends, rules,
            and codecs.
        tokenizer: Optional callable (text -> token list) handed to the
            planning context.
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep.
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
    return search_plan(plan, context, registry)


def search_candidates(logical, context, registry) -> tuple[PhysicalCandidate, ...]:
    """Lower and refine every logical alternative before comparing costs."""
    backend = registry.backend(context.backend)
    planners = (backend, *registry.physical_planners.values())
    rules = tuple(registry.physical_rules.values())
    alternatives = logical_alternatives(
        logical, tuple(registry.logical_alternatives.values()), context)
    candidates = []
    for alternative in alternatives:
        candidate_context = replace(
            context, logical_plan=alternative.plan, settings={})
        region = ModelRegion(alternative.plan)
        for planner in planners:
            for candidate in planner.plan(region, candidate_context):
                candidate = replace(
                    candidate,
                    logical_plan=candidate.logical_plan or alternative.plan,
                    logical_rules=alternative.rules)
                plan = candidate.plan
                if not isinstance(plan, Refusal):
                    if plan.backend != context.backend:
                        raise ValueError(
                            f"physical planner returned backend {plan.backend!r} "
                            f"for selected backend {context.backend!r}")
                    plan = apply_plan_rules(
                        replace(plan, estimated_seconds=candidate.estimated_seconds),
                        rules, replace(candidate_context,
                                       logical_plan=candidate.logical_plan))
                refused = isinstance(plan, Refusal)
                candidates.append(replace(
                    candidate, plan=plan,
                    graph=None if refused else plan.graph,
                    estimated_seconds=float("inf") if refused
                    else plan.estimated_seconds))
    return tuple(candidates)


def search_plan(logical, context, registry):
    """Select the cheapest refined candidate and record the comparison."""
    candidates = search_candidates(logical, context, registry)
    if not candidates:
        return Refusal(
            reasons=(f"backend {context.backend!r} could not produce a plan",),
            constraint="no_physical_plan",
            needed=1, available=0, unit="plans")
    # Original alternatives come first, so min preserves them on a cost tie.
    selected = min(range(len(candidates)),
                   key=lambda index: candidates[index].estimated_seconds)
    plan = candidates[selected].plan
    if isinstance(plan, Refusal):
        return plan
    comparison = tuple({
        "logical_rules": candidate.logical_rules,
        "estimated_seconds": (None if isinstance(candidate.plan, Refusal)
                              else candidate.estimated_seconds),
        "refusal": (candidate.plan.constraint
                    if isinstance(candidate.plan, Refusal) else None),
        "selected": index == selected,
    } for index, candidate in enumerate(candidates))
    return replace(plan, settings={**plan.settings, "candidates": comparison})


def refine_plan(plan, *, model: ModelSpec, device: DeviceSpec,
                doc_tokens: dict, gpus: int = 1, backend: str = "quail",
                registry, order: str | None = None,
                tokenizer=None, pair_fractions=None,
                canvas_draws: int = 4,
                attention: str | None = None):
    """Refine an explicit physical plan without exploring other placements.

    Used for physical plans edited by the caller. Normal query plans use
    plan_query again when exact token stores arrive, so all alternatives
    can benefit from prefix sharing before selection.
    """
    if isinstance(plan, Refusal):
        return plan
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
    them. Changes to inference estimates update both the plan total and
    its search cost, and the node estimates are refreshed for explain.

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
    difference = inference_seconds(graph.nodes) - inference_seconds(plan.nodes)
    seconds = plan.estimated_seconds + difference
    if "search_seconds" in plan.settings:
        context.settings["search_seconds"] = (
            plan.settings["search_seconds"] + difference)
    return replace(
        plan, nodes=graph.nodes, root=graph.root, estimated_seconds=seconds,
        settings=context.settings,
        estimates=plan.estimator(graph) if plan.estimator else plan.estimates)
