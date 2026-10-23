"""Search logical alternatives and select a refined physical plan."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner.optimizer.candidates import ModelRegion, PhysicalCandidate
from quail.planner.optimizer.context import PlanningContext
from quail.planner.optimizer.graph_refinement import apply_plan_rules
from quail.planner.optimizer.tree_rewrites import logical_alternatives
from quail.planner.plan import Refusal
from quail.specs import DeviceSpec, ModelSpec


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, document_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry, tokenizer=None, pair_fractions=None,
               scan_fractions=None, scan_maxima=None,
               canvas_draws: int = 4,
               attention: str | None = None, memo=None,
               catalog=None, engine_config=None, logical_candidates=None):
    """Plan one query with the selected model backend.

    Args:
        plan: The input logical plan.
        model: Model spec.
        device: Device spec.
        document_tokens: Per document token counts for each table alias.
        gpus: GPU count handed to the backend.
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
        scan_maxima: alias -> the longest document its regular
            predicates keep, in tokens.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        memo: Results the tree rewrites computed for this plan, such
            as its statistics, for the physical planner to reuse.
        catalog: Catalog available to tree rewrites during search.
        engine_config: Session configuration available to tree rewrites.
        logical_candidates: Already rewritten trees, or None to run logical rules.

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
        gpus=gpus,
        document_tokens=document_tokens,
        backend=backend,
        order=order,
        canvas_draws=canvas_draws,
        attention=attention,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
        scan_fractions=dict(scan_fractions or {}),
        scan_maxima=dict(scan_maxima or {}),
        logical_plan=plan,
        memo={} if memo is None else memo,
        catalog=catalog, engine_config=engine_config,
    )
    return search_plan(
        plan, context, registry, logical_candidates=logical_candidates)


def search_candidates(
    logical, context, registry, *, logical_candidates=None,
) -> tuple[PhysicalCandidate, ...]:
    """Lower and refine every logical alternative before comparing costs."""
    backend = registry.backend(context.backend)
    planners = (backend, *registry.physical_planners.values())
    rules = tuple(registry.physical_rules.values())
    alternatives = (logical_alternatives(
        logical, tuple(registry.logical_rules.values()), context)
        if logical_candidates is None else logical_candidates)
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
                    rewrites=alternative.rules)
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


def search_plan(logical, context, registry, *, logical_candidates=None):
    """Select the cheapest refined candidate and record the comparison."""
    candidates = search_candidates(
        logical, context, registry, logical_candidates=logical_candidates)
    if not candidates:
        return Refusal(
            reasons=(f"backend {context.backend!r} could not produce a plan",),
            constraint="no_physical_plan",
            needed=1, available=0, unit="plans")
    # Candidate order breaks cost ties; built-in rules return the original first.
    selected = min(range(len(candidates)),
                   key=lambda index: candidates[index].estimated_seconds)
    plan = candidates[selected].plan
    if isinstance(plan, Refusal):
        return plan
    comparison = tuple({
        "rewrites": candidate.rewrites,
        "estimated_seconds": (None if isinstance(candidate.plan, Refusal)
                              else candidate.estimated_seconds),
        "refusal": (candidate.plan.constraint
                    if isinstance(candidate.plan, Refusal) else None),
        "selected": index == selected,
    } for index, candidate in enumerate(candidates))
    return replace(
        plan, logical_plan=candidates[selected].logical_plan,
        settings={**plan.settings, "candidates": comparison})


def refine_plan(plan, *, model: ModelSpec, device: DeviceSpec,
                document_tokens: dict, gpus: int = 1, backend: str = "quail",
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
        gpus=gpus,
        document_tokens=document_tokens,
        backend=backend,
        order=order,
        canvas_draws=canvas_draws,
        attention=attention,
        tokenizer=tokenizer,
        pair_fractions=dict(pair_fractions or {}),
    )
    return apply_plan_rules(
        plan, tuple(registry.physical_rules.values()), context)
