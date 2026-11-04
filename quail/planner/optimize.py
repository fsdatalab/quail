"""Search logical alternatives and select a refined physical plan."""

from dataclasses import replace

from quail.logical import LogicalPlan
from quail.planner import native
from quail.planner.optimizer.candidates import ModelRegion
from quail.planner.optimizer.context import PlanningContext
from quail.planner.plan import Refusal
from quail.specs import DeviceSpec, ModelSpec


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, document_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry, tokenizer=None, pair_fractions=None,
               scan_fractions=None,
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
        logical_plan=plan,
        memo={} if memo is None else memo,
        catalog=catalog, engine_config=engine_config,
    )
    return search_plan(
        plan, context, registry, logical_candidates=logical_candidates)


def search_plan(logical, context, registry, *, logical_candidates=None):
    """Select the cheapest plan of the logical alternatives.

    The Rust planner plans the Quail backend. Another backend lowers
    each alternative with its own ``plan`` and runs no physical rules.

    Raises:
        ValueError: The context has no tokenizer.
    """
    if context.tokenizer is None:
        raise ValueError("planning needs the model's tokenizer")
    alternatives = (native.logical_alternatives(logical, registry, context)
                    if logical_candidates is None else logical_candidates)
    if context.backend == "quail":
        return native.search_plan(alternatives, context, registry)
    return _search_backend(alternatives, context, registry)


def _search_backend(alternatives, context, registry):
    """Select the cheapest of a request backend's candidates."""
    backend = registry.backend(context.backend)
    candidates = []
    for alternative in alternatives:
        candidate_context = replace(
            context, logical_plan=alternative.plan, settings={})
        for candidate in backend.plan(ModelRegion(alternative.plan), candidate_context):
            plan = candidate.plan
            if not isinstance(plan, Refusal) and plan.backend != context.backend:
                raise ValueError(
                    f"backend {context.backend!r} returned a plan for "
                    f"backend {plan.backend!r}")
            candidates.append(replace(
                candidate,
                logical_plan=candidate.logical_plan or alternative.plan,
                rewrites=alternative.rules))
    if not candidates:
        return Refusal(
            reasons=(f"backend {context.backend!r} could not produce a plan",),
            constraint="no_physical_plan",
            needed=1, available=0, unit="plans")
    # candidate order breaks cost ties; the first alternative is the input
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
    """Apply the physical rules again to a plan the caller edited.

    Normal query plans use plan_query again when exact token stores
    arrive, so all alternatives can benefit from prefix sharing before
    selection. A plan another backend planned is returned unchanged.
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
    return native.refine_plan(plan, context, registry)
