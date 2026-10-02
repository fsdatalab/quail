"""Build the physical plan of a decided logical plan."""

from dataclasses import replace

from quail.cost import budgets
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, scan
from quail.logical import (
    PROBABILITIES_SUFFIX,
    Alias,
    CompileError,
    LabelIn,
    LabelWork,
    LogicalPlan,
    classified_above_joins,
    effective_selectivity,
    oriented_join_conditions,
)
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    Barrier,
    Exchange,
    Filter,
    FilterStage,
    Foreign,
    HashJoin,
    InList,
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
from quail.planner import joins as joinsearch
from quail.planner import retention
from quail.planner.classify import (
    ClassifyRefusedError,
    classification_refusal,
    classify_table,
)
from quail.planner.filters import default_order_rule
from quail.planner.physical_optimizer import (
    ModelRegion,
    PlanningContext,
    apply_physical_rules,
)
from quail.planner.plan import PhysicalPlan, Refusal
from quail.planner.statistics import (
    cached_statistics,
    filter_orders,
    filter_works,
    live_after_filters,
    question_tokens,
    sequence_specs,
)
from quail.specs import DeviceSpec, ModelSpec


def hash_join_nodes(joins, pair_fractions, scan_ports) -> list:
    """One HashJoin per join with equality conditions, over the scans.

    Args:
        joins: The logical joins in written order.
        pair_fractions: written position -> pairs kept over the cross
            product, as the session measured them.
        scan_ports: The scans' id ports, one per alias, in any order.

    Returns:
        HashJoin nodes with ids ``hash_join:<left>-<right>``.
    """
    pair_fractions = pair_fractions or {}
    by_alias = {port.port.split(":", 1)[1]: port for port in scan_ports}
    nodes = []
    for position, join in enumerate(joins):
        oriented = oriented_join_conditions(join)
        if oriented is None:
            continue
        left, right, conditions = oriented
        on = tuple(
            (left_ref.column, right_ref.column)
            for left_ref, right_ref in conditions
        )
        nodes.append(HashJoin(
            node_id=f"hash_join:{left}-{right}",
            inputs=input_ports((by_alias[left], by_alias[right])),
            left=left, right=right, on=on, written_pos=position,
            pair_fraction=pair_fractions.get(position, 1.0)))
    return nodes



def node_estimates(graph, *, filter_works, stage_works, live, stats, pre,
                   cap_pages, model, device, chunk) -> dict:
    """Price each node's own work, and each chain's recompute if released.

    Returns node id -> {"seconds", and for a filter chain a later join
    anchors on, "release_recompute_tokens" and
    "release_recompute_seconds"}. A node's seconds price its work
    alone; nodes do not add up to the plan's estimate because chunk
    packing shares forward passes across them. The recompute figure
    is the expected survivors past the retention pool cap times their
    prefix cost: what the join pays if the chain's KV is not pinned.
    """
    anchored = {node.anchor for node in graph.nodes if isinstance(node, AiJoin)}
    out = {}
    for node in graph.nodes:
        entry = {}
        if isinstance(node, AiFilter) and node.alias in filter_works:
            entry["seconds"] = speed_of_light(
                filter_works[node.alias], model, device, chunk).seconds
            if node.alias in anchored:
                mean = stats[node.alias].mean_doc_tokens
                prefix = pre + mean
                pages = -(-prefix // budgets.PAGE_TOKENS)
                fits = cap_pages // max(1, pages)
                excess = max(0.0, live.get(node.alias, 0.0) - fits)
                entry["release_recompute_tokens"] = round(excess * prefix)
                entry["release_recompute_seconds"] = speed_of_light(
                    scan(prefix, 0, window=model.sliding_window) * excess,
                    model, device, chunk).seconds
        elif isinstance(node, AiJoin):
            work = Work()
            for stage in node.stages:
                work = work + stage_works.get(stage.written_pos, Work())
            entry["seconds"] = speed_of_light(work, model, device, chunk).seconds
        # other nodes do no model work and get no seconds
        out[node.node_id] = entry
    return out


# ----------------------------------------------- KV keep (residency)

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

def joined_calls(labels: LabelWork) -> list:
    """Return classifications of document pairs and their table aliases."""
    return [(call, call.aliases()) for call, _ in labels.calls
            if len(call.aliases()) == 2]


def plan_quail(plan: LogicalPlan, *, model: ModelSpec,
                device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                order: str | None = None, pair_fractions=None,
                context: PlanningContext | None = None):
    """Compile a decided LogicalPlan into a PhysicalPlan or Refusal.

    The logical rules have made every choice the plan records: each
    table's filter order, the joins' stage order and anchors, and
    where each classification sits. A one-table classification above
    a SemanticJoin runs after the joins, over the documents the joins
    matched; one on its table runs after the table's AI.IF filters. A
    plan without recorded orders and anchors runs as written. Each
    classification's scoring rule is left for the label_scoring
    physical rule and which KV stays resident for kv_retention, so the
    estimate and ``search_seconds`` count the filters, the joins, and
    the classifications of joined rows only.

    Args:
        plan: The logical plan to compile.
        model: Model spec.
        device: Device spec.
        doc_tokens: alias -> list of per-document token counts.
        gpus: GPU count; one model copy runs per GPU.
        order: Stage order rule, 'by_cost' or 'as_written'; None picks
            the default rule. Reported in the plan's settings.
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        context: The planning context, needed when the plan classifies
            documents.

    Returns:
        A PhysicalPlan, or a Refusal explaining why the query cannot run.
    """
    operators = plan.operators()
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    applies = operators.applies
    labels = operators.labels
    classified = {alias for _, alias in labels.calls}
    after_joins = classified_above_joins(plan.root)
    if labels.calls:
        if context is None:
            return Refusal(
                reasons=("AI.CLASSIFY planning needs a planning context",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        if any(len(call.aliases()) not in (1, 2) for call, _ in labels.calls):
            return Refusal(
                reasons=("AI.CLASSIFY reads one document, or one from each "
                         "side of a join",),
                constraint="unsupported_classify_query",
                needed=1, available=0, unit="queries")
        refusal = classification_refusal(context)
        if refusal is not None:
            return refusal
    alias_applies = {}
    join_applies = {}
    for apply in applies:
        if apply.ids == "pairs":
            join_applies.setdefault(apply.written_pos, []).append(apply)
        else:
            alias_applies.setdefault(apply.aliases[0], []).append(apply)
    names = [apply.function for apply in applies]
    if len(set(names)) != len(names):
        raise CompileError(
            f"each apply() needs its own name; {names} repeat one")
    if any(len(group) > 1 for group in join_applies.values()):
        raise CompileError("a join takes one apply() returning pairs")
    if gpus > 1 and any(apply.kind == "per_batch" for apply in applies):
        return Refusal(
            reasons=("per-batch apply() requires one GPU; "
                     "use apply_table() or set gpus=1",),
            constraint="per_batch_apply_needs_one_gpu",
            needed=1, available=gpus, unit="gpus")

    # ---- refusals first
    weight_gpus = budgets.minimum_weight_gpus(model, device)
    if weight_gpus > 1:
        return Refusal(
            reasons=(
                f"the model needs {weight_gpus} GPUs of memory, "
                f"but Quail loads one complete copy per GPU",
            ),
            constraint="weights_need_more_cards",
            needed=weight_gpus, available=1, unit="cards")
    workers = gpus

    statistics = cached_statistics(
        plan, {} if context is None else context.memo, model=model,
        device=device, doc_tokens=doc_tokens, pair_fractions=pair_fractions)
    stats, chunk, pre = statistics.stats, statistics.chunk, statistics.pre
    specs = statistics.specs
    asks = statistics.asks
    ask_filters = {alias: [filters[alias][position] for position in positions]
                   for alias, positions in asks.items() if positions}
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)
    orders = filter_orders(plan)

    # ---- expected live counts after filters, and the filter work
    # every stage's price sits on
    live0 = live_after_filters(plan, statistics)
    works = filter_works(plan, statistics, model)
    base_work = sum(works.values(), Work())

    # ---- the stages as decided: their work and expected tuples
    seq = sequence_specs(plan, statistics)
    stage_work, stage_records = joinsearch.walk(
        seq, live0, statistics.lengths, set(filters), pre, model, device)
    # a second group on the same anchor gets :2, a third :3
    sequence_groups = retention.group_sequence(seq)
    group_ids = []
    seen_ids = {}
    for group in sequence_groups:
        anchor = group[0][1]
        seen_ids[anchor] = seen_ids.get(anchor, 0) + 1
        group_ids.append(f"ai_join:{anchor}" if seen_ids[anchor] == 1
                         else f"ai_join:{anchor}:{seen_ids[anchor]}")

    def unique_id(prefix, name):
        key = f"{prefix}:{name}"
        seen_ids[key] = seen_ids.get(key, 0) + 1
        return key if seen_ids[key] == 1 else f"{key}:{seen_ids[key]}"

    # a chain streams only into its alias's first use, and never through
    # a barrier apply
    barrier_aliases = {alias for alias, group in alias_applies.items()
                       if any(apply.kind == "barrier" for apply in group)}
    for group in sequence_groups:
        anchor = group[0][1]
        for spec, _ in group:
            if any(apply.kind == "barrier"
                   for apply in join_applies.get(spec["written_pos"], ())):
                barrier_aliases.add(anchor)
    chained = retention.chained_aliases(
        sequence_groups, set(ask_filters), barrier_aliases, workers)

    # ---- refusal checks on the predicted plan
    anchors = {spec["written_pos"]: anchor for spec, anchor in seq}
    for s in scans:
        fq = max((question_tokens(p.prompt, model.canvas_tokens)
                  for p in ask_filters.get(s.alias, ())), default=None)
        if fq is None:
            continue
        need = pre + stats[s.alias].max_doc_tokens + fq
        if need > chunk:
            return Refusal(
                reasons=(f"a document in {s.alias!r} needs {need} tokens "
                         f"with its prompt, but the forward pass budget "
                         f"is {chunk} tokens",),
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
                reasons=(f"one join pair anchored on {anchor!r} needs "
                         f"{need} tokens, but the forward pass budget "
                         f"is {chunk} tokens",),
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
        sid = f"scan:{s.alias}"
        nodes.append(PhysicalScan(
            node_id=sid,
            alias=s.alias, input_id=s.alias,
            n_docs=stats[s.alias].n_docs,
            total_tokens=stats[s.alias].total_tokens,
            shard_ranges=shard_ranges,
            shard_token_loads=tuple(loads)))
        ids_src[s.alias] = PortRef(sid, f"ids:{s.alias}")
    # the hash join reads the scans; survivors thin its pairs at the AI join
    pairs_src = {}
    for node in hash_join_nodes(joins, pair_fractions, ids_src.values()):
        nodes.append(node)
        pairs_src[node.written_pos] = node

    # aliases whose filter chain runs in one pipeline with the
    # classification or join after it
    between = classified - after_joins
    pipelined = chained | {
        alias for alias in ask_filters if workers == 1 and alias in between}

    def emit_filter(alias):
        n = stats[alias].n_docs
        stages, surv = [], 1.0
        for i in orders[alias]:
            p = filters[alias][i]
            stages.append(FilterStage(
                written_pos=i,
                question_tokens=question_tokens(p.prompt, model.canvas_tokens),
                preamble_tokens=p.prompt.preamble_tokens,
                selectivity=p.selectivity,
                expected_docs=round(n * surv, 1)))
            surv *= effective_selectivity(p.selectivity)
        # a later stage, or the classification or join of its alias in
        # the same pipeline, reads the KV the chain wrote
        writes = len(stages) > 1 or alias in pipelined
        fid = f"ai_filter:{alias}"
        nodes.append(AiFilter(
            node_id=fid,
            inputs=input_ports((ids_src[alias],)),
            alias=alias, arena_writes=writes,
            stages=tuple(stages)))
        ids_src[alias] = PortRef(fid, f"ids:{alias}")
        emit_applies(alias)

    def emit_applies(alias):
        for apply in alias_applies.get(alias, ()):
            aid = f"apply:{apply.function}"
            nodes.append(Foreign(
                node_id=aid,
                inputs=input_ports((ids_src[alias],)),
                function=apply.function, kind=apply.kind, ids=apply.ids,
                columns=tuple((ref.alias, ref.column)
                              for ref in apply.columns),
                aliases=(alias,)))
            ids_src[alias] = PortRef(aid, f"ids:{alias}")

    # documents expected after a table's AI.IF filters, before its
    # filters on labels
    live_asked = {}
    for alias, predicates in filters.items():
        survival = 1.0
        for position in asks[alias]:
            survival *= effective_selectivity(predicates[position].selectivity)
        live_asked[alias] = float(stats[alias].n_docs) * survival
    label_ports = []

    def emit_classify(alias, live=None):
        """Classify the alias's documents, then keep the accepted labels."""
        calls = [call for call, owner in labels.calls
                 if owner == alias and len(call.aliases()) == 1]
        if not calls or (alias in after_joins and live is None):
            return
        table = classify_table(context, alias, "quail")
        if live is None:
            live = live_asked.get(alias, float(stats[alias].n_docs))
        steps = [(predicate.expression.call, position)
                 for position, predicate in enumerate(filters.get(alias, ()))
                 if isinstance(predicate.expression, LabelIn)]
        steps.extend((call, None) for call in calls if call not in labels.tests)
        classified_calls = set()
        scores = None
        for call, position in steps:
            if call not in classified_calls:
                spec = table.prepare(call, labels.names[call], live)
                node = table.node(spec, scores or ids_src[alias],
                                  sum(isinstance(n, AiClassify) for n in nodes))
                nodes.append(node)
                scores = PortRef(node.node_id, "scores")
                ids_src[alias] = PortRef(node.node_id, f"ids:{alias}")
                if call in labels.projected:
                    label_ports.append(scores)
                classified_calls.add(call)
            if position is not None:
                predicate = filters[alias][position]
                lid = f"filter:{alias}:{position}"
                nodes.append(Filter(
                    node_id=lid, inputs=input_ports((scores,)),
                    predicate=InList(labels.names[call],
                                     tuple(predicate.expression.accepted)),
                    aliases=(alias,), selectivity=predicate.selectivity,
                    written_pos=position))
                scores = PortRef(lid, "scores")
                ids_src[alias] = PortRef(lid, f"ids:{alias}")
                live *= effective_selectivity(predicate.selectivity)

    try:
        for s in scans:
            if s.alias in ask_filters:
                emit_filter(s.alias)
            else:
                emit_applies(s.alias)
            emit_classify(s.alias)
    except ClassifyRefusedError as refused:
        return refused.refusal()

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    records = iter(stage_records)
    groups = [[(spec, next(records)) for spec, _ in group]
              for group in sequence_groups]
    exec_idx = 0
    pairs_edges = []     # every full stage's passing-pairs edge
    out_aliases = []     # recombination's output order
    for g, group in enumerate(groups):
        anchor = sequence_groups[g][0][1]
        if g > 0:
            ahead = [scan.alias for scan in scans]
            bid = unique_id("barrier", anchor)
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
            # anchors go to the GPU holding their KV, else balanced
            xid = unique_id("exchange", anchor)
            nodes.append(Exchange(
                node_id=xid,
                inputs=input_ports((ids_src[anchor],)),
                anchor=anchor))
            ids_src[anchor] = PortRef(xid, f"ids:{anchor}")
        gid = group_ids[g]
        stage_dicts = []
        in_aliases = [anchor]
        pair_inputs = []
        for spec, record in group:
            partners = [a for a in spec["aliases"] if a != anchor]
            pairs_from = ""
            if spec["written_pos"] in pairs_src:
                producer = pairs_src[spec["written_pos"]]
                pair_inputs.append(
                    PortRef(producer.node_id, f"pairs:{spec['written_pos']}"))
                pairs_from = producer.node_id
            for apply in join_applies.get(spec["written_pos"], ()):
                # the function's pairs reach the join on their own port
                aid = f"apply:{apply.function}"
                nodes.append(Foreign(
                    node_id=aid,
                    inputs=input_ports(tuple(
                        ids_src[a] for a in apply.aliases)),
                    function=apply.function, kind=apply.kind, ids="pairs",
                    columns=tuple((ref.alias, ref.column)
                                  for ref in apply.columns),
                    aliases=tuple(apply.aliases),
                    written_pos=spec["written_pos"]))
                pair_inputs.append(PortRef(aid, f"pairs:{spec['written_pos']}"))
                pairs_from = aid
            stage_dicts.append(JoinStage(
                written_pos=spec["written_pos"], exec_idx=exec_idx,
                anchor=anchor, partners=tuple(partners),
                semantics=spec["semantics"],
                selectivity=spec["selectivity"],
                expected_tuples=round(record["tuples"], 1),
                anchor_frame_tokens=spec["frame_tokens"][anchor],
                pair_tail_tokens=spec["tail_tokens"],
                anchor_resident=record["resident"],
                tuple_tokens=round(record["tokens"], 1),
                pairs_from=pairs_from))
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
            inputs=input_ports(tuple(ids_src[a] for a in in_aliases)
                               + tuple(pair_inputs)),
            anchor=anchor,
            anchor_resident=group[0][1]["resident"],
            stages=tuple(stage_dicts)))
        ids_src[anchor] = PortRef(gid, f"ids:{anchor}")

    # a classification of joined rows follows the join that keeps them,
    # in the join's pipeline: each kept row's partner block, question,
    # and label paths run over the anchor's resident KV
    try:
        for call, aliases in joined_calls(labels):
            for g, group in enumerate(sequence_groups):
                anchor = group[0][1]
                stage = next((spec for spec, _ in group
                              if set(spec["aliases"]) == set(aliases)), None)
                if stage is not None:
                    break
            else:
                raise AssertionError(
                    "a classification of joined rows without its join")
            partner = next(alias for alias in aliases if alias != anchor)
            table = classify_table(context, anchor, "quail")
            join = joins[stage["written_pos"]]
            pairs = (effective_selectivity(join.selectivity)
                     * live0[anchor] * live0[partner])
            spec, _ = table.classify_joined(
                call, labels.names[call], partner, pairs,
                stats[partner].mean_doc_tokens)
            node = table.node(
                spec, PortRef(group_ids[g], f"join_answers:{stage['written_pos']}"),
                sum(isinstance(n, AiClassify) for n in nodes))
            nodes.append(node)
            if call in labels.projected:
                label_ports.append(PortRef(node.node_id, "scores"))
    except ClassifyRefusedError as refused:
        return refused.refusal()

    # classifications after the joins run over the documents the joins
    # matched: a document with at least one true pair, expected as the
    # join selectivity times the partners it met, at most every document
    try:
        for alias in sorted(after_joins):
            met = sum(
                effective_selectivity(join.selectivity)
                * sum(live0[argument.alias] for argument in join.prompt.args
                      if argument.alias != alias)
                for join in joins
                if alias in {argument.alias for argument in join.prompt.args})
            emit_classify(alias, live=live0[alias] * min(1.0, met))
    except ClassifyRefusedError as refused:
        return refused.refusal()

    if len(pairs_edges) == 1 and len(seq) == 1 and not after_joins:
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
    columns = []
    for c in plan.root.columns:
        columns.append(c.name if isinstance(c, Alias)
                       else f"{c.alias}.{c.column}")
        if isinstance(c, Alias) and getattr(c.expression, "probabilities", False):
            columns.append(c.name + PROBABILITIES_SUFFIX)
    nodes.append(PhysicalProject(
        node_id="project",
        inputs=input_ports(tuple(sink_inputs) + tuple(label_ports)),
        columns=tuple(columns)))
    if plan.root.limit is not None:
        nodes.append(Limit(
            node_id="limit",
            inputs=input_ports((PortRef("project", "rows"),)),
            count=plan.root.limit,
        ))

    estimate = (speed_of_light(base_work + stage_work, model, device,
                               chunk).seconds + _classify_seconds(nodes))
    stage_works = {record["written_pos"]: record["work"]
                   for record in stage_records}

    def estimator(graph):
        return node_estimates(
            graph, filter_works=works, stage_works=stage_works,
            live=live0, stats=stats, pre=pre, cap_pages=statistics.cap_pages,
            model=model, device=device, chunk=chunk)

    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        backend="quail", estimated_seconds=estimate,
        nodes=tuple(nodes),
        settings={
            "chunk_tokens": chunk,
            "arena_pages": list(statistics.arena_pages),
            "admission_tokens": statistics.admission,
            "order_rule": rule,
            "order_source": source,
            "search_seconds": estimate,
            **({"classify_placement": ("after joins" if after_joins
                                       else "before joins")}
               if labels.calls else {}),
            **({"canvas_draws": context.canvas_draws}
               if labels.calls and model.answer_canvas else {}),
        },
        estimator=estimator)


def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None, backend: str = "quail",
               registry=None, tokenizer=None, pair_fractions=None,
               canvas_draws: int = 4,
               attention: str | None = None, remarks=(), memo=None):
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
        pair_fractions: join written position -> the fraction of the
            cross product its equality conditions keep.
        remarks: The logical rules' remarks, placed before the
            physical planner's.
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
    if remarks:
        selected_plan = replace(
            selected_plan, remarks=tuple(remarks) + selected_plan.remarks)
    return _apply_rules(selected_plan, tuple(registry.physical_rules.values()),
                        context, "")


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
    return _apply_rules(plan, tuple(registry.physical_rules.values()), context,
                        " once the documents were tokenized")


def _apply_rules(plan, rules, context, when: str):
    """Apply physical rules in order; a remark names each that fired.

    The rules see the plan's settings on the context and may add to
    them and leave remarks; a classification no rule can score refuses
    the plan.
    """
    context = replace(context, settings=dict(plan.settings), remarks=[])
    try:
        graph, changed = apply_physical_rules(plan.graph, rules, context)
    except ClassifyRefusedError as refused:
        return refused.refusal()
    if not changed and not context.remarks \
            and context.settings == dict(plan.settings):
        return plan
    # a rule that re-estimates a classification moves the plan's total
    # by the same amount
    seconds = plan.estimated_seconds + (
        _classify_seconds(graph.nodes) - _classify_seconds(plan.nodes))
    return replace(
        plan, nodes=graph.nodes, root=graph.root, estimated_seconds=seconds,
        settings=context.settings,
        remarks=plan.remarks + tuple(context.remarks) + tuple(
            f"physical rule {name} changed the plan{when}" for name in changed))


def _classify_seconds(nodes) -> float:
    return sum(node.spec.estimated_seconds for node in nodes
               if isinstance(node, AiClassify) and node.spec is not None)


# ------------------------------------------------------------- explain

def explain(logical: LogicalPlan, physical, *, verbose: bool = False,
            result=None, usd_per_hour: float | None = None) -> str:
    """Format the logical and physical operator trees.

    Args:
        logical: The optimized logical plan.
        physical: The physical plan or planning refusal.
        verbose: Include runtime settings and internal node fields.
        result: The QueryResult of running the plan. When given, each
            node shows its measured rows, time, and tokens next to the
            estimates, and the measured totals follow the tree.
        usd_per_hour: Price of one GPU, for the measured cost per query.
    """
    from quail.explain import (
        _fields,
        logical_tree,
        measured_stages,
        physical_tree,
        run_summary,
    )

    lines = ["logical:"]
    lines.extend("  " + line for line in logical_tree(logical).splitlines())
    if isinstance(physical, Refusal):
        lines.append(f"refusal: {physical.constraint}: needed "
                     f"{physical.needed} {physical.unit}, available "
                     f"{physical.available}")
        lines.extend(f"  {reason}" for reason in physical.reasons)
        return "\n".join(lines)
    lines.append("")
    lines.append(f"physical: backend={physical.backend}, "
                 f"model={physical.model}, workers={physical.workers}")
    if physical.backend == "quail":
        chunk = physical.settings.get("chunk_tokens")
        admission = physical.settings.get("admission_tokens")
        budgets = ["KV=bf16"]
        if chunk is not None:
            budgets.append(f"chunk budget={chunk:,} tokens")
        if admission is not None:
            budgets.append(f"admission budget={admission:,} tokens")
        lines.append("  " + ", ".join(budgets))
    lines.append("")
    lines.extend("  " + line for line in physical_tree(
        physical.graph, logical=logical, verbose=verbose,
        estimates=getattr(physical, "estimates", None),
        metrics=None if result is None else result.node_metrics,
        stages=None if result is None else measured_stages(result.report),
    ).splitlines())
    if getattr(physical, "estimates", None):
        lines.append("  est. time is each node's work alone; node times do "
                     "not add up to the plan estimate")
        lines.append("  because chunk packing shares forward passes across "
                     "nodes")
    if result is not None:
        lines.append("")
        lines.append("run:")
        lines.extend("  " + line for line in run_summary(
            result.report, physical.graph, physical.workers, usd_per_hour))
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
