"""Physical planning for numeric AI.SCORE expressions."""

import math

from quail.cost import budgets
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, triangle
from quail.logical import (
    SHARED_PRE,
    Apply,
    ScoreExpression,
    effective_selectivity,
)
from quail.physical import (
    AiScore,
    Limit,
    PortRef,
    Project,
    Scan,
    ScoreFilter,
    ScoreSpec,
)
from quail.physical.base import input_ports
from quail.planner import collect_operators, hash_join_nodes
from quail.planner.physical_optimizer import PhysicalCandidate
from quail.planner.plan import PhysicalPlan, Refusal
from quail.reranker import render_qwen3_reranker_input


def _refusal(reason: str, constraint: str = "unsupported_score_query"):
    return Refusal(
        reasons=(reason,),
        constraint=constraint,
        needed=1,
        available=0,
        unit="queries",
    )


def _prompt_aliases(prompt) -> tuple[str, ...]:
    return tuple(dict.fromkeys(argument.alias for argument in prompt.args))


def _query_template(prompt) -> str:
    aliases = _prompt_aliases(prompt)
    if len(aliases) == 1:
        marker = f"{SHARED_PRE}{{0}}"
        return prompt.template.replace(marker, "", 1).strip()
    if len(aliases) == 2 and len(prompt.args) == 2:
        return prompt.template.replace("{1}", "this document")
    raise ValueError("AI.SCORE supports one document or one document pair")


def _score_work(count, mean_tokens, fixed_tokens, *, variance=0.0,
                prefix_tokens=0.0, prefix_variance=0.0, groups=0.0,
                shared_tokens=0, copies=1) -> Work:
    if count <= 0:
        return Work()
    length = mean_tokens + fixed_tokens
    work = Work(
        tokens=length,
        pairs=triangle(length) + variance / 2,
        kv_written=length,
    ) * count
    reused = max(0.0, count - groups)
    work += Work(
        tokens=-prefix_tokens,
        pairs=-triangle(prefix_tokens) - prefix_variance / 2,
        kv_written=-prefix_tokens,
        kv_read=prefix_tokens,
    ) * reused
    shared_reuses = max(0.0, groups - copies)
    return work + Work(
        tokens=-shared_tokens,
        pairs=-triangle(shared_tokens),
        kv_written=-shared_tokens,
        kv_read=shared_tokens,
    ) * shared_reuses


def _variance(lengths) -> float:
    mean = sum(lengths) / max(1, len(lengths))
    return sum((length - mean) ** 2 for length in lengths) / max(1, len(lengths))


def _score_spec(
    prompt,
    *,
    name: str,
    expected_inputs: float,
    mean_tokens: float,
    context,
    chunk_tokens: int,
    pair_fraction: float = 1.0,
    prefix_groups: float | None = None,
):
    query_template = _query_template(prompt)
    rendered = render_qwen3_reranker_input(query_template, "{document}")
    before, after = rendered.split("{document}")
    parts = (before, after)
    if len(_prompt_aliases(prompt)) == 2:
        first, middle = before.split("{0}")
        parts = (first, middle, after)
    token_parts = tuple(tuple(context.tokenizer(part)) for part in parts)
    aliases = _prompt_aliases(prompt)
    lengths = [context.document_tokens[alias] for alias in aliases]
    fixed_tokens = sum(map(len, token_parts))
    shared = len(token_parts[0])
    prefix = float(shared)
    prefix_variance = 0.0
    groups = min(float(context.gpu_count), expected_inputs)
    capacity = budgets.arena_tokens(context.model, context.device, chunk_tokens)
    if len(aliases) == 2:
        prefix += sum(lengths[0]) / max(1, len(lengths[0])) + len(token_parts[1])
        prefix_variance = _variance(lengths[0])
        groups = min(
            expected_inputs,
            len(lengths[0]) if prefix_groups is None else prefix_groups,
        )
        longest = sum(max(values, default=0) for values in lengths) + fixed_tokens
        if longest > capacity:
            prefix = float(shared)
            prefix_variance = 0.0
            groups = min(float(context.gpu_count), expected_inputs)
    work = _score_work(
        expected_inputs, mean_tokens, fixed_tokens,
        variance=sum(_variance(values) for values in lengths),
        prefix_tokens=prefix, prefix_variance=prefix_variance,
        groups=groups, shared_tokens=shared, copies=context.gpu_count,
    )
    estimate = speed_of_light(
        work * (1 / max(1, min(context.gpu_count, expected_inputs))),
        context.model, context.device, chunk_tokens,
    ).seconds
    return ScoreSpec(
        name=name,
        aliases=_prompt_aliases(prompt),
        query_template=query_template,
        arguments=tuple(
            (argument.alias, argument.column) for argument in prompt.args
        ),
        expected_inputs=expected_inputs,
        estimated_seconds=estimate,
        pair_fraction=pair_fraction,
        prompt_token_parts=token_parts,
    ), work


def _projection_scores(logical) -> tuple[ScoreExpression, ...]:
    return tuple(
        expression
        for expression in logical.root.columns
        if isinstance(expression, ScoreExpression)
    )


def _score_name(prompt, projected, fallback: str) -> str:
    matches = [item.name for item in projected if item.prompt == prompt]
    if len(matches) > 1:
        raise ValueError("one AI.SCORE expression needs one output name")
    return matches[0] if matches else fallback


def _ordered_filters(predicates, count, mean_tokens, context, chunk):
    remaining = list(enumerate(predicates))
    ordered = []
    live = float(count)
    while remaining:
        choices = []
        for written_pos, predicate in remaining:
            spec, work = _score_spec(
                predicate.prompt,
                name=f"__score_filter_{written_pos}",
                expected_inputs=live,
                mean_tokens=mean_tokens,
                context=context,
                chunk_tokens=chunk,
            )
            rejected = max(
                1e-9, 1.0 - effective_selectivity(predicate.selectivity)
            )
            choices.append(
                (spec.estimated_seconds / rejected, written_pos, work)
            )
        _, selected, work = min(choices)
        ordered.append((selected, predicates[selected], live, work))
        live *= effective_selectivity(predicates[selected].selectivity)
        remaining = [item for item in remaining if item[0] != selected]
    return ordered, live


def plan_reranker(region, context, *, backend_name: str):
    """Build one Quail plan for AI.SCORE expressions."""
    logical = region.logical_plan
    if any(isinstance(node, Apply) for node in logical.walk()):
        refusal = _refusal("AI.SCORE cannot be mixed with apply()")
        return (PhysicalCandidate(None, refusal, float("inf")),)

    scans, filters, joins = collect_operators(logical)
    projected = _projection_scores(logical)
    predicates = [
        predicate
        for values in filters.values()
        for predicate in values
    ] + list(joins)
    if not predicates and not projected:
        refusal = _refusal("a reranker model needs an AI.SCORE expression")
        return (PhysicalCandidate(None, refusal, float("inf")),)
    if any(predicate.comparison is None for predicate in predicates):
        refusal = _refusal(
            "AI.SCORE cannot be mixed with generative AI predicates"
        )
        return (PhysicalCandidate(None, refusal, float("inf")),)
    if any(join.semantics != "full" for join in joins):
        refusal = _refusal("AI.SCORE supports full joins only")
        return (PhysicalCandidate(None, refusal, float("inf")),)
    pair_prompts = [
        *(join.predicate for join in joins),
        *(score.prompt for score in projected
          if len(_prompt_aliases(score.prompt)) == 2),
    ]
    if len(set(pair_prompts)) > 1:
        refusal = _refusal(
            "AI.SCORE currently supports one document pair expression"
        )
        return (PhysicalCandidate(None, refusal, float("inf")),)

    nodes = []
    scan_refs = {}
    counts = {}
    means = {}
    for scan in scans:
        lengths = context.document_tokens[scan.alias]
        count = len(lengths)
        total = sum(int(length) for length in lengths)
        counts[scan.alias] = count
        means[scan.alias] = total / max(1, count)
        node = Scan(
            node_id=f"scan:{scan.alias}",
            alias=scan.alias,
            input_id=scan.alias,
            n_docs=count,
            total_tokens=total,
            shard_ranges=((0, count),),
            shard_token_loads=(total,),
        )
        nodes.append(node)
        scan_refs[scan.alias] = PortRef(node.node_id, f"ids:{scan.alias}")

    hash_nodes = hash_join_nodes(
        joins, context.pair_fractions, tuple(scan_refs.values())
    )
    nodes.extend(hash_nodes)
    pair_refs = [
        PortRef(node.node_id, f"pairs:{node.written_pos}")
        for node in hash_nodes
    ]

    chunk = budgets.chunk_budget(context.model, context.device)
    total_work = Work()
    total_seconds = 0.0
    live = {alias: float(count) for alias, count in counts.items()}
    current = dict(scan_refs)
    scored_prompts = set()
    score_index = 0

    def add_score(
        prompt,
        name,
        inputs,
        expected,
        mean,
        pair_fraction=1.0,
        prefix_groups=None,
    ):
        nonlocal score_index, total_work, total_seconds
        spec, work = _score_spec(
            prompt,
            name=name,
            expected_inputs=expected,
            mean_tokens=mean,
            context=context,
            chunk_tokens=chunk,
            pair_fraction=pair_fraction,
            prefix_groups=prefix_groups,
        )
        node = AiScore(
            node_id=f"ai-score:{score_index}",
            inputs=input_ports(tuple(inputs)),
            backend_name=backend_name,
            model=context.model.name,
            spec=spec,
        )
        score_index += 1
        nodes.append(node)
        total_work += work
        total_seconds += spec.estimated_seconds
        scored_prompts.add(prompt)
        return PortRef(node.node_id, "scores"), spec

    for alias in counts:
        ordered, live_count = _ordered_filters(
            filters.get(alias, ()),
            counts[alias],
            means[alias],
            context,
            chunk,
        )
        for written_pos, predicate, expected, _work in ordered:
            name = _score_name(
                predicate.prompt,
                projected,
                f"__score_{alias}_{written_pos}",
            )
            score_ref, spec = add_score(
                predicate.prompt,
                name,
                (current[alias],),
                expected,
                means[alias],
            )
            filtered = ScoreFilter(
                node_id=f"score-filter:{alias}:{written_pos}",
                inputs=input_ports((score_ref,)),
                score_name=spec.name,
                aliases=(alias,),
                comparison=predicate.comparison,
                threshold=predicate.threshold,
                selectivity=predicate.selectivity,
                written_pos=written_pos,
            )
            nodes.append(filtered)
            current[alias] = PortRef(filtered.node_id, "scores")
        live[alias] = live_count

        for expression in projected:
            if expression.prompt in scored_prompts:
                continue
            if _prompt_aliases(expression.prompt) != (alias,):
                continue
            current[alias], _ = add_score(
                expression.prompt,
                expression.name,
                (current[alias],),
                live[alias],
                means[alias],
            )

    pair_prompt = pair_prompts[0] if pair_prompts else None
    sink = current[scans[0].alias]
    if pair_prompt is not None:
        left, right = _prompt_aliases(pair_prompt)
        pair_fraction = context.pair_fractions.get(0, 1.0)
        expected = live[left] * live[right] * pair_fraction
        touched = (
            1.0 if pair_fraction >= 1 else
            -math.expm1(live[right] * math.log1p(-pair_fraction))
        )
        prefix_groups = live[left] * touched
        projected_name = next(
            (score.name for score in projected if score.prompt == pair_prompt),
            "__score_join_0",
        )
        score_ref, spec = add_score(
            pair_prompt,
            projected_name,
            [current[left], current[right], *pair_refs],
            expected,
            means[left] + means[right],
            pair_fraction,
            prefix_groups=prefix_groups,
        )
        matching_join = next(
            (join for join in joins if join.predicate == pair_prompt), None
        )
        if matching_join is not None:
            written_pos = joins.index(matching_join)
            filtered = ScoreFilter(
                node_id=f"score-filter:join:{written_pos}",
                inputs=input_ports((score_ref,)),
                score_name=spec.name,
                aliases=(left, right),
                comparison=matching_join.comparison,
                threshold=matching_join.threshold,
                selectivity=matching_join.selectivity,
                written_pos=written_pos,
            )
            nodes.append(filtered)
            sink = PortRef(filtered.node_id, "scores")
        else:
            sink = score_ref

    columns = tuple(
        expression.name
        if isinstance(expression, ScoreExpression)
        else f"{expression.alias}.{expression.column}"
        for expression in logical.root.columns
    )
    nodes.append(Project(
        node_id="project",
        inputs=input_ports((sink,)),
        columns=columns,
    ))
    if logical.root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=logical.root.limit,
        ))

    estimate = total_seconds
    plan = PhysicalPlan(
        model=context.model.name,
        device=context.device.name,
        workers=context.gpu_count,
        backend=backend_name,
        estimated_seconds=estimate,
        nodes=tuple(nodes),
        settings={
            "chunk_tokens": chunk,
            "true_ids": list(context.tokenizer("yes")),
            "false_ids": list(context.tokenizer("no")),
            "retained_kv_tokens": budgets.arena_tokens(
                context.model, context.device, chunk
            ),
            "prefix_reuse": "fixed prompt and first document within each score",
            "survivor_assumption": "uniform independent selection",
            "estimated_fresh_tokens": total_work.tokens,
            "estimated_attention_pairs": total_work.pairs,
            "batching": "token_based_admission",
            "data_parallel_copies": context.gpu_count,
            "score_normalization": "yes_no_softmax",
            "order_rule": "cost_per_expected_rejection",
        },
    )
    return (PhysicalCandidate(plan.graph, plan, estimate),)
