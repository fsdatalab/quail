"""Choose filter order, joins, anchors, and KV retention before execution."""

from dataclasses import replace

from quail.executor.retention import retention_pages
from quail.logical import (
    LogicalPlan,
    Project,
    Scan,
    SemanticFilter,
    SemanticJoin,
)
from quail.physical import (
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    FilterStage,
    JoinStage,
    Limit,
    PortRef,
    Recombine,
)
from quail.physical import (
    Project as PhysicalProject,
)
from quail.physical import (
    Scan as PhysicalScan,
)
from quail.physical.base import input_ports
from quail.planner import budgets, retention
from quail.planner import joins as joinsearch
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
from quail.planner.sol import speed_of_light, unrounded_seconds
from quail.planner.work import Work, ask, scan
from quail.planning import ModelRegion, PlanningContext, apply_physical_rules
from quail.specs import DeviceSpec, ModelSpec

# ---------------------------------------------------------- tree walk

def collect_operators(plan: LogicalPlan):
    """Return (scans, filters_by_alias, joins_in_written_order)."""
    scans, filters, joins = [], {}, []

    def walk(node):
        if isinstance(node, Project):
            walk(node.input)
        elif isinstance(node, SemanticJoin):
            for child in node.inputs:
                walk(child)
            joins.append(node)
        elif isinstance(node, SemanticFilter):
            walk(node.input)
            filters[node.input.alias] = list(node.predicates)
        elif isinstance(node, Scan):
            scans.append(node)
        else:
            for child in node.children():
                walk(child)

    walk(plan.root)
    return scans, filters, joins


def _question_tokens(prompt) -> int:
    """Token count of the prompt's per-evaluation tail."""
    if prompt.tail_tokens is None or prompt.preamble_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return prompt.tail_tokens


def preamble_tokens(filters, joins) -> int:
    """Return the engine preamble's token count from any bound prompt."""
    for fs in filters.values():
        for p in fs:
            if p.prompt.preamble_tokens is not None:
                return p.prompt.preamble_tokens
    for j in joins:
        if j.predicate.preamble_tokens is not None:
            return j.predicate.preamble_tokens
    return 0


# --------------------------------------------------------- filter order

def default_order_rule(filters, joins) -> tuple[str, str]:
    """Return (rule, source).

    'by_cost' when every predicate has a selectivity, 'as_written'
    otherwise.
    """
    preds = [p for fs in filters.values() for p in fs]
    sels = [p.selectivity for p in preds] + [j.selectivity for j in joins]
    if sels and all(s is not None for s in sels):
        return "by_cost", "default: every gated predicate has a selectivity"
    return "as_written", "default: at least one predicate has no selectivity"


def filter_cost(predicate, prefix_tokens: float, model: ModelSpec,
                device: DeviceSpec, chunk_tokens: int, *, first: bool) -> float:
    """Return ideal time for one filter evaluation."""
    operation = scan if first else ask
    work = operation(prefix_tokens, _question_tokens(predicate.prompt))
    return unrounded_seconds(work, model, device, chunk_tokens)


def order_filters_indexed(predicates, rule: str, *, prefix_tokens: float,
                          model: ModelSpec, device: DeviceSpec,
                          chunk_tokens: int):
    """Return written positions of predicates in execution order.

    'by_cost' minimizes ideal expected time. It sorts the asks once,
    then prices each predicate as the first scan. Written order breaks
    ties.
    """
    idx = list(range(len(predicates)))
    if rule == "as_written":
        return idx

    def selectivity(i):
        return (predicates[i].selectivity
                if predicates[i].selectivity is not None else 1.0)

    ask_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=False)
        for p in predicates
    ]
    scan_costs = [
        filter_cost(p, prefix_tokens, model, device, chunk_tokens,
                    first=True)
        for p in predicates
    ]

    def score(i):
        killed = 1.0 - selectivity(i)
        if killed <= 0:
            return float("inf")
        return ask_costs[i] / killed

    ask_order = sorted(idx, key=score)
    prefix_live = [1.0]
    prefix_cost = [0.0]
    for i in ask_order:
        prefix_cost.append(
            prefix_cost[-1] + prefix_live[-1] * ask_costs[i])
        prefix_live.append(prefix_live[-1] * selectivity(i))

    total_ask_cost = prefix_cost[-1]
    candidates = []
    for position, first in enumerate(ask_order):
        expected = (
            scan_costs[first]
            + selectivity(first) * prefix_cost[position]
            + total_ask_cost - prefix_cost[position + 1]
        )
        candidates.append((expected, first))

    first = min(candidates)[1]
    return [first, *(i for i in ask_order if i != first)]


def order_filters(predicates, rule: str, *, prefix_tokens: float,
                  model: ModelSpec, device: DeviceSpec,
                  chunk_tokens: int):
    return [predicates[i] for i in order_filters_indexed(
        predicates, rule, prefix_tokens=prefix_tokens,
        model=model, device=device, chunk_tokens=chunk_tokens)]


def _join_aliases(join) -> list:
    """Return the join's table aliases in placeholder order."""
    return [r.alias for r in join.predicate.args]


def _label_counts(join) -> dict:
    """Return alias -> (block_label_tokens, anchor_frame_tokens)."""
    out = {a: (lt, nt) for a, lt, nt in join.predicate.labels}
    if any(lt is None for lt, _ in out.values()):
        raise ValueError(
            "join prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return out


def join_specs(joins) -> list:
    """The joins as the search's spec dicts, in written order."""
    out = []
    for i, j in enumerate(joins):
        labels = _label_counts(j)
        out.append(dict(
            written_pos=i, aliases=_join_aliases(j), anchor=j.anchor,
            anchor_free=(j.anchor is None and j.semantics == "full"),
            semantics=j.semantics, selectivity=j.selectivity,
            frame_tokens={a: nt for a, (lt, nt) in labels.items()},
            label_tokens={a: lt for a, (lt, nt) in labels.items()},
            tail_tokens=_question_tokens(j.predicate)))
    return out


def _filter_work(filters, stats, filter_orders: dict, pre: int) -> Work:
    """Expected Work of every filter chain.

    The first stage scans each document, later stages ask over
    resident KV.
    """
    total = Work()
    for alias, preds in filters.items():
        mean = stats[alias].mean_doc_tokens
        n = float(stats[alias].n_docs)
        for si, predicate_index in enumerate(filter_orders[alias]):
            p = preds[predicate_index]
            q = _question_tokens(p.prompt)
            op = scan if si == 0 else ask
            total = total + op(pre + mean, q) * n
            n *= p.selectivity if p.selectivity is not None else 1.0
    return total


# ----------------------------------------------- KV keep (residency)

def _length_stats(doc_tokens) -> joinsearch.AliasStats:
    if isinstance(doc_tokens, joinsearch.AliasStats):
        return doc_tokens
    return joinsearch.summarize_alias(doc_tokens)


def possible_anchor_aliases(specs) -> set:
    """Return every legal anchor considered during planning."""
    out = set()
    for spec in specs:
        out.update(joinsearch.anchor_candidates(spec))
    return out


def balanced_shards(doc_tokens, workers: int):
    """Greedily partition documents into shards balanced by token count."""
    if workers == 1:
        return (tuple(range(len(doc_tokens))),), [sum(doc_tokens)]
    order = sorted(range(len(doc_tokens)), key=lambda i: -doc_tokens[i])
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(i)
        loads[w] += doc_tokens[i]
    return tuple(tuple(sorted(s)) for s in shards), loads


def contiguous_shards(doc_tokens, workers: int):
    """Split ordered documents into compact token balanced ranges."""
    lengths = doc_tokens
    n_docs = len(lengths)
    total = sum(lengths)
    ranges = []
    loads = []
    start = 0
    consumed = 0
    for worker in range(workers):
        if worker == workers - 1:
            stop = n_docs
            load = total - consumed
        else:
            remaining_workers = workers - worker
            target = (total - consumed) / remaining_workers
            stop = start
            load = 0
            while stop < n_docs:
                next_length = int(lengths[stop])
                with_next = load + next_length
                if load and abs(target - load) <= abs(target - with_next):
                    break
                load = with_next
                stop += 1
                if load >= target:
                    break
        ranges.append((start, stop))
        loads.append(load)
        start = stop
        consumed += load
    return tuple(ranges), loads


# ---------------------------------------------------------- the planner

def plan_quail(plan: LogicalPlan, *, model: ModelSpec,
                device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                order: str | None = None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    Args:
        plan: The logical plan to compile.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> list of per-document token counts.
        gpus: GPU count; one model copy runs per GPU.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
    """
    scans, filters, joins = collect_operators(plan)
    length_stats = {a: joinsearch.summarize_alias(t)
                    for a, t in doc_tokens.items()}
    stats = {
        a: CorpusStats(n_docs=s.count, total_tokens=s.total,
                       max_doc_tokens=s.maximum)
        for a, s in length_stats.items()
    }
    for s in scans:
        if s.alias not in stats:
            raise ValueError(f"no doc_tokens for alias {s.alias!r}")
    remarks = []

    # ---- refusals first
    weight_gpus = budgets.minimum_weight_gpus(model, device)
    if weight_gpus > 1:
        return Refusal(
            reasons=(
                f"one model copy needs the memory of {weight_gpus} GPUs, "
                "but Quail does not split weights across GPUs",
            ),
            constraint="weights_need_more_cards",
            needed=weight_gpus, available=1, unit="cards")
    workers = gpus

    chunk = budgets.chunk_budget(model, device)
    admission = budgets.arena_tokens(model, device, chunk)
    pre = preamble_tokens(filters, joins)
    specs = join_specs(joins)

    # ---- the order rule first: the search below needs it
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)
    fixed = rule == "as_written"
    filter_orders = {
        alias: order_filters_indexed(
            predicates, rule,
            prefix_tokens=pre + stats[alias].mean_doc_tokens,
            model=model, device=device, chunk_tokens=chunk)
        for alias, predicates in filters.items()
    }

    # ---- expected live counts after filters, and the fixed filter
    # work every candidate join plan shares
    live0 = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in fs:
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live0[_filter_alias(fs[0])] *= surv
    base_work = _filter_work(filters, stats, filter_orders, pre)

    cap_pages = retention_pages(admission, chunk, budgets.PAGE_TOKENS)
    costs = retention.coefficients(model, device)
    # KV reuse is priced as unlimited: a filtered alias pays no prefix
    # at its first anchor use, and no alias pays one at a later use
    filtered = set(filters)

    def run_search(honor_forced=True):
        found = joinsearch.search_joins(
            specs, live0, length_stats, filtered, pre,
            chunk, model, device, base_work=base_work,
            fixed_order=fixed, honor_forced=honor_forced)
        if found is None:
            found = joinsearch.search_joins(
                specs, live0, length_stats, filtered, pre, chunk, model,
                device, base_work=base_work, fixed_order=True,
                honor_forced=honor_forced)
        return found

    found = run_search()
    seq = [(specs[position], anchor) for position, anchor in found["seq"]]
    retention_plan = retention.schedule(seq, live0)
    # a filtered alias whose first use is as an anchor has its chain run
    # right before that group and stream into it with KV pinned; its
    # survivors never enter the retention pool. An alias that was a
    # partner first must finish its chain before that earlier group, so
    # its survivors wait in the pool.
    streamed = {}
    partner_before = set()
    sequence_groups = retention.group_sequence(seq)
    for index, group in enumerate(sequence_groups):
        anchor = group[0][1]
        if anchor in filters and anchor not in streamed \
                and anchor not in partner_before:
            streamed[anchor] = index
        for spec, _ in group:
            for alias in spec["aliases"]:
                if alias != anchor and alias not in streamed:
                    partner_before.add(alias)
    retention_plan["initial"] = {
        alias: use for alias, use in retention_plan["initial"].items()
        if alias not in streamed
    }
    retention_plan.update(**costs, cap_pages=cap_pages)
    forced = sorted({s["anchor"] for s in specs
                     if s["semantics"] == "full"
                     and not s["anchor_free"]})
    if forced:
        free = run_search(honor_forced=False)
        honored_s = speed_of_light(
            base_work + found["work"], model, device, chunk).seconds
        free_s = speed_of_light(
            base_work + free["work"], model, device, chunk).seconds
        if free_s < honored_s:
            remarks.append(
                f"anchors {forced} were forced; a free choice prices "
                f"lower ({free_s:.3f} vs {honored_s:.3f} predicted "
                f"seconds)")
    stage_records = found["records"]

    # ---- refusal checks on the predicted plan
    anchors = {wp: a for wp, a in found["seq"]}
    for s in scans:
        fq = max((_question_tokens(p.prompt)
                  for p in filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_doc_tokens + fq
        if need > chunk:
            return Refusal(
                reasons=(f"a document of {s.alias!r} plus the engine "
                         f"preamble and its question tail needs {need} "
                         f"tokens; the chunk budget is {chunk} and "
                         f"suffixes are atomic - no chunk can ever "
                         f"hold it",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")
    for spec in specs:
        anchor = anchors[spec["written_pos"]]
        need = (pre + stats[anchor].max_doc_tokens
                + spec["frame_tokens"][anchor]
                + sum(spec["label_tokens"][p] + stats[p].max_doc_tokens
                      for p in spec["aliases"] if p != anchor)
                + spec["tail_tokens"])
        if need > chunk:
            return Refusal(
                reasons=(f"the join anchored on {anchor!r} needs "
                         f"{need} tokens per tuple (anchor document, "
                         f"each partner document with its label, and "
                         f"the question, in one prompt); that is over "
                         f"the chunk budget of {chunk}, and suffixes "
                         f"are atomic",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    # ---- build the dataflow graph; ids_src tracks each table's
    # current producer node
    nodes = []
    ids_src = {}
    for s in scans:
        shard_ranges, loads = contiguous_shards(
            doc_tokens[s.alias], workers
        )
        sid = f"input:{s.alias}"
        nodes.append(PhysicalScan(
            node_id=sid,
            alias=s.alias, input_id=s.alias,
            n_docs=stats[s.alias].n_docs,
            total_tokens=stats[s.alias].total_tokens,
            shard_ranges=shard_ranges,
            shard_token_loads=tuple(loads)))
        ids_src[s.alias] = PortRef(sid, f"ids:{s.alias}")

    def emit_filter(alias):
        order_idx = filter_orders[alias]
        n = stats[alias].n_docs
        stages, surv = [], 1.0
        for i in order_idx:
            p = filters[alias][i]
            stages.append(FilterStage(
                written_pos=i,
                question_tokens=_question_tokens(p.prompt),
                preamble_tokens=p.prompt.preamble_tokens,
                selectivity=p.selectivity,
                expected_docs=round(n * surv, 1)))
            surv *= p.selectivity if p.selectivity is not None else 1.0
        keep = alias in retention_plan["initial"]
        pinned = alias in streamed
        writes = len(stages) > 1 or keep or pinned
        # a pinned survivor's pages also cover the consuming join's
        # largest frame, so the join never claims a page for it
        hold = max((spec["frame_tokens"][alias]
                    for spec, _ in sequence_groups[streamed[alias]])
                   if pinned else (0,))
        fid = f"filter:{alias}"
        nodes.append(AiFilter(
            node_id=fid,
            inputs=input_ports((ids_src[alias],)),
            alias=alias, arena_writes=writes,
            keep_kv=keep, pin_survivors=pinned, hold_tokens=hold,
            stages=tuple(stages)))
        ids_src[alias] = PortRef(fid, f"ids:{alias}")
        if alias in streamed:
            remarks.append(
                f"filter on {alias!r} streams its survivors into "
                f"the join anchored on it; each one's KV stays "
                f"pinned until its tuples are answered")
        elif not writes:
            remarks.append(
                f"filter on {alias!r}: arena writes off (one "
                f"stage - nothing reads the KV again)")

    for s in scans:
        if s.alias in filters and s.alias not in streamed:
            emit_filter(s.alias)

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    groups = []
    for (spec, anchor), record in zip(seq, stage_records):
        merge = (groups and spec["semantics"] == "full"
                 and groups[-1]["full"]
                 and groups[-1]["anchor"] == anchor)
        if merge:
            groups[-1]["members"].append((spec, record))
        else:
            groups.append(dict(anchor=anchor,
                               full=(spec["semantics"] == "full"),
                               members=[(spec, record)]))
    exec_idx = 0
    barrier_n = 0
    pairs_edges = []     # every full stage's passing-pairs edge
    out_aliases = []     # recombination's output order
    for g, group in enumerate(groups):
        anchor = group["anchor"]
        if g > 0:
            ahead = [scan.alias for scan in scans]
            bid = f"barrier:{barrier_n}"
            barrier_n += 1
            exchange_inputs = tuple(pairs_edges) + tuple(
                ids_src[a] for a in ahead
            )
            nodes.append(Barrier(
                node_id=bid,
                inputs=input_ports(exchange_inputs),
                next_anchor=anchor,
                aliases=tuple(ahead)))
            for a in ahead:
                ids_src[a] = PortRef(bid, f"ids:{a}")
        if workers > 1:
            # anchors go to the GPU that holds their KV, or balance
            # across GPUs when none does; one GPU passes them through
            xid = f"exchange:{g}"
            nodes.append(Exchange(
                node_id=xid,
                inputs=input_ports((ids_src[anchor],)),
                anchor=anchor))
            ids_src[anchor] = PortRef(xid, f"ids:{anchor}")
        if streamed.get(anchor) == g:
            emit_filter(anchor)
        gid = f"group:{g}"
        stage_dicts = []
        in_aliases = [anchor]
        for spec, record in group["members"]:
            partners = [a for a in spec["aliases"] if a != anchor]
            stage_dicts.append(JoinStage(
                written_pos=spec["written_pos"], exec_idx=exec_idx,
                anchor=anchor, partners=tuple(partners),
                semantics=spec["semantics"],
                selectivity=spec["selectivity"],
                expected_tuples=round(record["tuples"], 1),
                anchor_frame_tokens=spec["frame_tokens"][anchor],
                pair_tail_tokens=spec["tail_tokens"],
                anchor_resident=record["resident"],
                tuple_tokens=round(record["tokens"], 1)))
            exec_idx += 1
            for a in partners:
                if a not in in_aliases:
                    in_aliases.append(a)
            if spec["semantics"] == "full":
                pairs_edges.append(PortRef(
                    gid, f"join_answers:{spec['written_pos']}"
                ))
                for a in [anchor] + partners:
                    if a not in out_aliases:
                        out_aliases.append(a)
        nodes.append(AiJoin(
            node_id=gid,
            inputs=input_ports(tuple(ids_src[a] for a in in_aliases)),
            anchor=anchor,
            anchor_resident=group["members"][0][1]["resident"],
            keep_anchor_kv=anchor in retention_plan["after"][gid],
            stages=tuple(stage_dicts)))
        ids_src[anchor] = PortRef(gid, f"ids:{anchor}")

    if len(pairs_edges) == 1 and len(seq) == 1:
        # one full join and nothing after it: its true pairs are the
        # result rows, so no recombination is needed
        sink_inputs = (pairs_edges[0],)
    elif pairs_edges:
        nodes.append(Recombine(
            node_id="recombine",
            inputs=input_ports(
                tuple(pairs_edges)
                + tuple(ids_src[a] for a in out_aliases)
            ),
            alias_order=tuple(out_aliases)))
        sink_inputs = (PortRef("recombine", "tuples"),)
    else:
        sink_inputs = (ids_src[scans[0].alias],)
    nodes.append(PhysicalProject(
        node_id="sink",
        inputs=input_ports(tuple(sink_inputs)),
        columns=tuple(f"{c.alias}.{c.column}" for c in plan.root.columns)))
    if plan.root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("sink", "rows"),)),
            count=plan.root.limit,
        ))

    estimate = speed_of_light(
        base_work + found["work"], model, device, chunk
    ).seconds
    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        backend="quail", estimated_seconds=estimate,
        nodes=tuple(nodes), remarks=tuple(remarks),
        settings={
            "chunk_tokens": chunk,
            "admission_tokens": admission,
            "retention": retention_plan,
            "order_rule": rule,
            "order_source": source,
        })


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry=None, tokenizer=None):
    """Plan one query with the selected model backend.

    Args:
        plan: The logical plan to plan.
        model: Model spec.
        device: Device spec.
        doc_tokens: Per document token counts for each table alias.
        gpus: GPU count handed to the backend as gpu_count.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule.
        backend: Registered model backend name.
        registry: Optional session extension registry.
        tokenizer: Optional callable (text -> token list) handed to the
            planning context.
    """
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
        tokenizer=tokenizer,
    )
    region = ModelRegion(plan)
    candidates = tuple(selected.plan(region, context))
    for physical_planner in registry.physical_planners.values():
        candidates += tuple(physical_planner.plan(region, context))
    if not candidates:
        return Refusal(
            reasons=(f"backend {backend!r} produced no physical plan",),
            constraint="no_physical_plan",
            needed=1,
            available=0,
            unit="plans",
        )
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
    graph, changed = apply_physical_rules(
        selected_plan.graph,
        tuple(registry.physical_rules.values()),
        context,
    )
    if changed:
        selected_plan = replace(
            selected_plan,
            nodes=graph.nodes,
            root=graph.root,
            remarks=selected_plan.remarks + tuple(
                f"physical rule {name} changed the plan"
                for name in changed
            ),
        )
    return selected_plan


def _filter_alias(pred_or_list):
    p = pred_or_list[0] if isinstance(pred_or_list, list) else pred_or_list
    return p.prompt.args[0].alias


# ------------------------------------------------------------- explain

def explain(logical: LogicalPlan, physical, *, verbose: bool = False) -> str:
    """Format the logical and physical operator trees.

    Args:
        logical: The optimized logical plan.
        physical: The physical plan or planning refusal.
        verbose: Include runtime settings and internal node fields.
    """
    from quail.explain import _fields, logical_tree, physical_tree

    lines = ["logical:"]
    lines.extend("  " + line for line in logical_tree(logical).splitlines())
    if isinstance(physical, Refusal):
        lines.append(f"refusal: {physical.constraint}: needed "
                     f"{physical.needed} {physical.unit}, available "
                     f"{physical.available}")
        lines.extend(f"  {reason}" for reason in physical.reasons)
        return "\n".join(lines)
    lines.append("")
    lines.append(f"physical: (backend={physical.backend}, "
                 f"model={physical.model}, workers={physical.workers})")
    if physical.backend == "quail":
        chunk = physical.settings.get("chunk_tokens")
        admission = physical.settings.get("admission_tokens")
        budgets = ["KV=bf16"]
        if chunk is not None:
            budgets.append(f"chunk budget={chunk:,} tokens")
        if admission is not None:
            budgets.append(f"admission budget={admission:,} tokens")
        lines.append("  " + ", ".join(budgets))
    lines.extend("  " + line for line in physical_tree(
        physical.graph, logical=logical, verbose=verbose).splitlines())
    if verbose:
        lines.append("")
        lines.append("settings:")
        lines.extend(_fields({"model": physical.model,
                              "device": physical.device,
                              **physical.settings}, 1))
        if physical.backend == "quail":
            lines.append("  KV dtype=bf16")
        lines.extend(f"  remark: {remark}" for remark in physical.remarks)
    return "\n".join(lines)
