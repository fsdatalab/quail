"""Build the physical plan of a decided logical plan."""

from dataclasses import dataclass

from quail.cost import budgets
from quail.cost import classify as classify_cost
from quail.cost.sol import speed_of_light
from quail.cost.work import Work, scan
from quail.labels import DECISION_SCORING, LETTERS_SCORING
from quail.logical import (
    PROBABILITIES_SUFFIX,
    Alias,
    CompileError,
    LabelWork,
    LogicalPlan,
    classified_above_joins,
    effective_selectivity,
    has_score,
    is_score,
    oriented_join_conditions,
)
from quail.logical.prompts import (
    choice_token_parts,
    classify_prompt_tokens,
    prompt_aliases,
    score_query_template,
)
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    AiScore,
    Barrier,
    ClassifySpec,
    Comparison,
    Exchange,
    Filter,
    FilterStage,
    Foreign,
    HashJoin,
    InList,
    JoinStage,
    PortRef,
    Recombine,
    ScoreSpec,
)
from quail.physical import (
    Scan as PhysicalScan,
)
from quail.physical.base import input_ports
from quail.planner import join_order as joinsearch
from quail.planner import retention
from quail.planner.filter_order import default_order_rule
from quail.planner.ordering import choose_filter_orders, choose_join_sequence
from quail.planner.physical_optimizer import PlanningContext
from quail.planner.plan import PhysicalPlan, Refusal, score_seconds
from quail.planner.results import result_nodes
from quail.planner.statistics import (
    ClassifyStatistics,
    cached_statistics,
    classify_statistics,
    filter_stop_keys,
    filter_works,
    live_after_filters,
    pair_counts,
    question_tokens,
    score_statistics,
)
from quail.planner.validation import (
    ClassifyRefusedError,
    ScoreRefusedError,
    classification_refusal,
    join_input_tokens,
    score_input_tokens,
    score_projections,
    score_refusal,
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


def regular_predicates(scan) -> tuple:
    """Return a logical scan's regular predicates as (column, comparison, value)."""
    return tuple((predicate.column.column, predicate.comparison,
                  predicate.value) for predicate in scan.predicates)


def build_physical_plan(plan: LogicalPlan, *, model: ModelSpec,
                        device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
                        order: str | None = None, pair_fractions=None,
                        context: PlanningContext | None = None,
                        scan_fractions=None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    The builder chooses each table's filter order and the joins' stage
    order and anchors by cost (quail.planner.ordering), unless order
    is "as_written". A one-table classification above a SemanticJoin
    runs after the joins, over the documents the joins matched. One on
    its table runs after the table's AI.IF filters.

    The label_scoring physical rule picks each classification's scoring
    rule, and the kv_retention rule picks which KV stays resident. The
    estimate and ``search_seconds`` count only the filters, the joins,
    and the classifications of joined rows.

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
        scan_fractions: alias -> the fraction of its documents the
            regular predicates are expected to keep; the context's when None.

    Returns:
        A PhysicalPlan, or a Refusal explaining why the query cannot run.
    """
    operators = plan.operators()
    scans, filters, joins = operators.scans, operators.filters, operators.joins
    applies = operators.applies
    if scan_fractions is None and context is not None:
        scan_fractions = context.scan_fractions
    labels = operators.labels
    classified = {alias for _, alias in labels.calls}
    after_joins = classified_above_joins(plan.root)
    scored = has_score(plan)
    projected_scores = score_projections(plan)
    if scored:
        refusal = score_refusal(plan, context)
        if refusal is not None:
            return refusal
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
        device=device, doc_tokens=doc_tokens, pair_fractions=pair_fractions,
        scan_fractions=scan_fractions, context=context)
    stats, chunk, pre = statistics.stats, statistics.chunk, statistics.pre
    specs = statistics.specs
    asks = statistics.asks
    ask_filters = {alias: [filters[alias][position] for position in positions]
                   for alias, positions in asks.items() if positions}
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(operators.all_filters(), joins)
    orders = choose_filter_orders(
        plan, statistics, model=model, device=device, rule=rule,
        context=context)
    stop_keys = filter_stop_keys(plan)

    # ---- expected live counts after filters, and the filter work
    live0 = live_after_filters(plan, statistics)
    works = filter_works(plan, statistics, model, orders)
    base_work = sum(works.values(), Work())

    # ---- the stages in the chosen order: their work and expected tuples
    seq = [(specs[position], anchor) for position, anchor in
           choose_join_sequence(plan, statistics, orders, model=model,
                                device=device, rule=rule)]
    stage_work, stage_records = joinsearch.walk(
        seq, live0, statistics.lengths, set(operators.all_filters()), pre,
        model, device)
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
        if "cost" in spec:
            continue
        anchor = anchors[spec["written_pos"]]
        need = join_input_tokens(
            spec, anchor, {alias: stats[alias].max_doc_tokens
                           for alias in spec["aliases"]}, pre)
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
    rows_src = {}
    scoring = ScoreLowering(context, nodes, projected_scores, chunk) if scored else None
    for s in scans:
        shard_ranges, loads = contiguous_shards(
            doc_tokens[s.alias], workers
        )
        sid = f"scan:{s.alias}"
        nodes.append(PhysicalScan(
            node_id=sid,
            alias=s.alias, input_id=s.alias,
            n_docs=len(doc_tokens[s.alias]),
            total_tokens=sum(doc_tokens[s.alias]),
            shard_ranges=shard_ranges,
            shard_token_loads=tuple(loads),
            predicates=regular_predicates(s),
            expected_docs=(float(stats[s.alias].n_docs)
                           if s.predicates else None)))
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
            stages=tuple(stages),
            stop_key=stop_keys.get(alias, ())))
        ids_src[alias] = PortRef(fid, f"ids:{alias}")
        emit_applies(alias)

    def emit_scores(alias):
        live = float(stats[alias].n_docs)
        for position in orders.get(alias, ()):
            predicate = filters[alias][position]
            source = rows_src.get(alias, ids_src[alias])
            if predicate.prompt not in scoring.names:
                source = scoring.score(
                    predicate.prompt, f"__score_{alias}_{position}", (source,),
                    live, stats[alias].mean_doc_tokens)
            source = scoring.comparison(predicate, source, (alias,), position)
            rows_src[alias] = source
            ids_src[alias] = PortRef(source.node_id, f"ids:{alias}")
            live *= effective_selectivity(predicate.selectivity)
        for expression in projected_scores:
            call = expression.expression
            if call.aliases() != (alias,) or call.prompt in scoring.names:
                continue
            rows_src[alias] = scoring.score(
                call.prompt, expression.name,
                (rows_src.get(alias, ids_src[alias]),), live,
                stats[alias].mean_doc_tokens)

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

    # documents expected after a table's model predicates, before its
    # filters on labels
    live_asked = {}
    for alias, predicates in filters.items():
        survival = 1.0
        for predicate in predicates:
            survival *= effective_selectivity(predicate.selectivity)
        live_asked[alias] = float(stats[alias].n_docs) * survival
    label_ports = []

    def emit_classify(alias, live=None):
        """Classify the alias's documents, then keep the accepted labels."""
        calls = [call for call, owner in labels.calls
                 if owner == alias and len(call.aliases()) == 1]
        if not calls or (alias in after_joins and live is None):
            return
        table = classify_builder(context, alias, "quail")
        if live is None:
            live = live_asked.get(alias, float(stats[alias].n_docs))
        steps = [(test.call, test)
                 for test in operators.label_filters.get(alias, ())]
        steps.extend((call, None) for call in calls if call not in labels.tests)
        classified_calls = set()
        scores = None
        for call, test in steps:
            if call not in classified_calls:
                spec, work = table.prepare(call, labels.names[call], live)
                if scoring is not None:
                    scoring.work += work
                node = table.node(spec, scores or rows_src.get(alias, ids_src[alias]),
                                  sum(isinstance(n, AiClassify) for n in nodes))
                nodes.append(node)
                scores = PortRef(node.node_id, "scores")
                ids_src[alias] = PortRef(node.node_id, f"ids:{alias}")
                if call in labels.projected:
                    label_ports.append(scores)
                classified_calls.add(call)
            if test is not None:
                lid = f"filter:{alias}:{test.position}"
                nodes.append(Filter(
                    node_id=lid, inputs=input_ports((scores,)),
                    predicate=InList(labels.names[call], test.values),
                    aliases=(alias,), selectivity=test.selectivity,
                    written_pos=test.position))
                scores = PortRef(lid, "scores")
                ids_src[alias] = PortRef(lid, f"ids:{alias}")
                live *= effective_selectivity(test.selectivity)
        if scores is not None:
            rows_src[alias] = scores

    try:
        for s in scans:
            if scoring is not None:
                emit_scores(s.alias)
            elif s.alias in ask_filters:
                emit_filter(s.alias)
            else:
                emit_applies(s.alias)
            emit_classify(s.alias)
    except ClassifyRefusedError as refused:
        return refused.refusal()
    except ScoreRefusedError as refused:
        return refused.refusal

    def emit_score_pair(prompt, position, fraction, pair_inputs):
        aliases = tuple(dict.fromkeys(ref.alias for ref in prompt.args))
        expected, prefixes = pair_counts(live0, aliases, fraction)
        return scoring.score(
            prompt, f"__score_join_{position}",
            tuple(rows_src.get(a, ids_src[a]) for a in aliases) + pair_inputs,
            expected, sum(stats[a].mean_doc_tokens for a in aliases),
            pair_fraction=fraction, prefix_groups=prefixes)

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    records = iter(stage_records)
    groups = [[(spec, next(records)) for spec, _ in group]
              for group in sequence_groups]
    exec_idx = 0
    pairs_edges = []     # every full stage's passing-pairs edge
    out_aliases = []     # recombination's output order
    score_pairs = None
    for g, group in enumerate(groups):
        anchor = sequence_groups[g][0][1]
        if "cost" in group[0][0]:
            try:
                for spec, _ in group:
                    position = spec["written_pos"]
                    join = joins[position]
                    pair_inputs = ()
                    if position in pairs_src:
                        producer = pairs_src[position]
                        pair_inputs = (PortRef(producer.node_id, f"pairs:{position}"),)
                    if score_pairs is None:
                        score_pairs = emit_score_pair(
                            join.prompt, position, spec["pair_fraction"], pair_inputs)
                    score_pairs = scoring.comparison(
                        join, score_pairs, spec["aliases"], position)
            except ScoreRefusedError as refused:
                return refused.refusal
            continue
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
            table = classify_builder(context, anchor, "quail")
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

    try:
        for expression in projected_scores:
            call = expression.expression
            if len(call.aliases()) == 2 and call.prompt not in scoring.names:
                score_pairs = emit_score_pair(call.prompt, 0, 1.0, ())
    except ScoreRefusedError as refused:
        return refused.refusal

    if score_pairs is not None:
        sink_inputs = (score_pairs,)
    elif len(pairs_edges) == 1 and len(seq) == 1 and not after_joins:
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
        alias = scans[0].alias
        sink_inputs = (rows_src.get(alias, ids_src[alias]),)
    columns = []
    for c in plan.projection.columns:
        columns.append(c.name if isinstance(c, Alias)
                       else f"{c.alias}.{c.column}")
        if isinstance(c, Alias) and getattr(c.expression, "probabilities", False):
            columns.append(c.name + PROBABILITIES_SUFFIX)
    nodes.extend(result_nodes(
        plan.result, tuple(sink_inputs) + tuple(label_ports), tuple(columns)))

    stage_work = sum((record["work"] for record in stage_records
                      if not is_score(joins[record["written_pos"]].predicate)), Work())
    estimate = (speed_of_light(base_work + stage_work, model, device,
                               chunk).seconds + score_seconds(nodes))
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
            **({
                "retained_kv_tokens": scoring.capacity,
                "prefix_reuse": "fixed prompt and first document within each score",
                "survivor_assumption": "uniform independent selection",
                "estimated_fresh_tokens": scoring.work.tokens,
                "estimated_attention_pairs": scoring.work.pairs,
                "batching": "token_based_admission",
                "data_parallel_copies": workers,
                "score_normalization": {
                    "reranker": "yes_no_softmax",
                    "decision": "decision_head_softmax",
                }.get(model.role, "true_false_softmax"),
            } if scoring is not None else {}),
            **({"classify_placement": ("after joins" if after_joins
                                       else "before joins")}
               if labels.calls else {}),
            **({"canvas_draws": context.canvas_draws}
               if labels.calls and model.answer_canvas else {}),
        },
        estimator=estimator)


def _pages(tokens: int) -> int:
    return -(-tokens // budgets.PAGE_TOKENS)


def score_name(prompt, projected, fallback: str) -> str:
    # the front end rejects one prompt projected under two names
    return next(
        (item.name for item in projected if item.expression.prompt == prompt),
        fallback,
    )


def score_spec(prompt, *, name, expected_inputs, mean_tokens, context,
               chunk_tokens, pair_fraction=1.0, prefix_groups=None):
    """Build a score specification from prepared prompt tokens and costs."""
    parts, cost = score_statistics(
        prompt, context, chunk_tokens, mean_tokens=mean_tokens)
    work, seconds = cost.estimate(expected_inputs, prefix_groups=prefix_groups)
    return ScoreSpec(
        name=name, aliases=prompt_aliases(prompt),
        query_template=score_query_template(prompt),
        arguments=tuple((ref.alias, ref.column) for ref in prompt.args),
        expected_inputs=expected_inputs, estimated_seconds=seconds,
        pair_fraction=pair_fraction, prompt_token_parts=parts, draws=cost.draws), work


class ScoreLowering:
    """Append score and comparison operators to a shared physical graph."""

    def __init__(self, context, nodes, projected, chunk):
        self.context = context
        self.nodes = nodes
        self.projected = projected
        self.chunk = chunk
        self.capacity = budgets.arena_tokens(context.model, context.device, chunk)
        self.names = {}
        self.work = Work()
        self.index = 0

    def score(self, prompt, fallback, inputs, expected, mean,
              *, pair_fraction=1.0, prefix_groups=None) -> PortRef:
        name = score_name(prompt, self.projected, fallback)
        spec, work = score_spec(
            prompt, name=name, expected_inputs=expected, mean_tokens=mean,
            context=self.context, chunk_tokens=self.chunk,
            pair_fraction=pair_fraction, prefix_groups=prefix_groups)
        prefix, suffix = score_input_tokens(spec, self.context)
        what = (f"a document in {spec.aliases[0]!r}" if len(spec.aliases) == 1
                else f"one pair of {spec.aliases[0]!r} and {spec.aliases[1]!r}")
        needed, available, unit = prefix + suffix, self.chunk, "tokens"
        if needed <= available:
            needed = _pages(prefix) + _pages(suffix)
            available, unit = self.capacity // budgets.PAGE_TOKENS, "pages"
        if needed > available:
            raise ScoreRefusedError(Refusal(
                reasons=(f"{what} needs {needed} {unit} with its prompt, "
                         f"but the execution budget is {available} {unit}",),
                constraint="suffix_over_chunk", needed=needed,
                available=available, unit=unit))
        node = AiScore(
            node_id=f"ai-score:{self.index}", inputs=input_ports(tuple(inputs)),
            backend_name=self.context.backend, model=self.context.model.name,
            spec=spec)
        self.index += 1
        self.nodes.append(node)
        self.names[prompt] = name
        self.work += work
        return PortRef(node.node_id, "scores")

    def comparison(self, predicate, source, aliases, position) -> PortRef:
        alias = aliases[0] if len(aliases) == 1 else "join"
        expression = (predicate.expression if len(aliases) == 1
                      else predicate.predicate)
        node = Filter(
            node_id=f"filter:{alias}:{position}",
            inputs=input_ports((source,)),
            predicate=Comparison(self.names[predicate.prompt],
                                 expression.comparison, expression.threshold),
            aliases=tuple(aliases), selectivity=predicate.selectivity,
            written_pos=position)
        self.nodes.append(node)
        return PortRef(node.node_id, "scores")


def classify_builder(context, alias, backend_name):
    """Prepare the inputs for constructing classification operators."""
    stats = classify_statistics(context, alias)
    return ClassifyBuilder(**vars(stats), tokenizer=context.tokenizer,
                           backend_name=backend_name)


@dataclass(frozen=True, kw_only=True)
class ClassifyBuilder(ClassifyStatistics):
    """Construct classification specifications and operators."""

    tokenizer: object
    backend_name: str

    def prepare(self, call, name, live) -> tuple[ClassifySpec, Work]:
        """Build a classification specification and work for a required method.

        The specification carries the prompt that names the labels; the
        label_scoring rule picks the scoring rule, swaps in the
        lettered prompt when it picks letters, and fills the estimate.

        Args:
            call: Logical AI.CLASSIFY call.
            name: Result column name.
            live: Expected number of documents to classify.

        Returns:
            The specification and its work when the model requires a scoring
            method. Otherwise, the physical label_scoring rule chooses the
            method and fills in its work later.
        """
        if self.model.role == "decision":
            return self._choice_spec(call, name, live, (self.alias,),
                                     tuple(self.tokenizer(call.prompt.preamble)))
        head, tail, labels = classify_prompt_tokens(
            call.prompt, call.labels, self.tokenizer)
        return ClassifySpec(
            name=name, aliases=(self.alias,),
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=live, estimated_seconds=0.0,
            prompt_token_parts=(head, tail),
            labels=tuple(call.labels),
            label_token_ids=labels,
            scoring="", probabilities=call.probabilities,
        ), Work()

    def node(self, spec, input_port, index, *ports) -> AiClassify:
        """Build the physical node for one classification."""
        return AiClassify(node_id=f"ai-classify:{index}",
                          inputs=input_ports((input_port, *ports)),
                          backend_name=self.backend_name,
                          model=self.model.name, spec=spec)

    def classify_joined(self, call, name, partner, pairs, partner_tokens):
        """Build a classification specification for joined document pairs.

        Pair classification uses letters and reuses the anchor document's KV.

        Args:
            call: Logical AI.CLASSIFY call referring to both documents.
            name: Result column name.
            partner: Partner table alias.
            pairs: Expected number of joined pairs to classify.
            partner_tokens: Mean partner document length in tokens.

        Returns:
            A tuple containing the ClassifySpec and estimated Work.

        Raises:
            ClassifyRefusedError: Probabilities are requested, the prompt lacks
                a lettered form, or its tokens exceed an execution budget.
        """
        if call.probabilities:
            raise ClassifyRefusedError(
                f"a classification of joined rows returns its label only; "
                f"{name!r} asks for the labels' probabilities", 1, 0)
        if self.model.role == "decision":
            return self._joined_choice_spec(call, name, partner, pairs,
                                            partner_tokens)
        prompt = call.prompt.lettered
        if prompt is None:
            raise ClassifyRefusedError(
                "the letters rule needs a one-token letter for every "
                "label, which the tokenizer does not have", 1, 0)
        _, tail, labels = classify_prompt_tokens(
            prompt, prompt.letters, self.tokenizer)
        head = tuple(prompt.preamble_token_ids)
        parts = {alias: (label, frame)
                 for alias, label, frame in prompt.label_token_ids}
        note, partner_label = parts[self.alias][1], parts[partner][0]
        block = int(round(partner_tokens)) + len(tail) - 1
        chains = [
            block + length
            for length in classify_cost.suffix_lengths(LETTERS_SCORING, labels)
        ]
        estimated = classify_cost.estimate_chains(
            len(head), len(note) + len(partner_label), chains, live=pairs,
            lengths=self.lengths,
            shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device, resident=True,
            canvas_rows=self.model.canvas_tokens)
        need = len(head) + self.longest + len(note) + max(chains)
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a row of {self.alias!r} x {partner!r} needs {need} tokens "
                f"with its classification prompt, but the forward pass "
                f"budget is {self.budget} tokens", need, self.budget)
        return ClassifySpec(
            name=name, aliases=(self.alias, partner),
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=pairs, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=labels, scoring=LETTERS_SCORING,
            join_layout=(tuple(note), tuple(partner_label)),
        ), estimated.work

    def _choice_spec(self, call, name, live, aliases, head):
        """Build a decision_choice specification for one document per row.

        Each document takes the question as its frame, then one request
        of every option block and the closing line.
        """
        tail, frame, blocks = choice_token_parts(call.prompt, self.tokenizer)
        request = len(tail) - frame
        need = len(head) + self.longest + len(tail)
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a document in {self.alias!r} needs {need} tokens with its "
                f"classification prompt, but the forward pass budget is "
                f"{self.budget} tokens", need, self.budget)
        estimated = classify_cost.estimate_chains(
            len(head), frame, [request], live=live, lengths=self.lengths,
            shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device)
        return ClassifySpec(
            name=name, aliases=aliases,
            query_template=call.prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in call.prompt.args),
            expected_inputs=live, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=blocks, scoring=DECISION_SCORING,
            probabilities=call.probabilities, frame_tokens=frame), estimated.work

    def _joined_choice_spec(self, call, name, partner, pairs, partner_tokens):
        """Build a decision_choice specification for joined document pairs.

        The anchor's KV ends in its note and the partner's label; each
        pair's request is the partner document, the question, every
        option block, and the closing line.
        """
        prompt = call.prompt
        head = tuple(prompt.preamble_token_ids)
        tail, frame, blocks = choice_token_parts(call.prompt, self.tokenizer)
        parts = {alias: (label, note)
                 for alias, label, note in prompt.label_token_ids}
        note, partner_label = parts[self.alias][1], parts[partner][0]
        request = int(round(partner_tokens)) + len(tail)
        estimated = classify_cost.estimate_chains(
            len(head), len(note) + len(partner_label), [request], live=pairs,
            lengths=self.lengths, shared=self.shared, chunk=self.chunk,
            capacity=self.capacity or self.budget, model=self.model,
            device=self.device, resident=True)
        need = len(head) + self.longest + len(note) + request
        if need > self.budget:
            raise ClassifyRefusedError(
                f"a row of {self.alias!r} x {partner!r} needs {need} tokens "
                f"with its classification prompt, but the forward pass "
                f"budget is {self.budget} tokens", need, self.budget)
        return ClassifySpec(
            name=name, aliases=(self.alias, partner),
            query_template=prompt.template,
            arguments=tuple((ref.alias, ref.column) for ref in prompt.args),
            expected_inputs=pairs, estimated_seconds=estimated.seconds,
            prompt_token_parts=(head, tail), labels=tuple(call.labels),
            label_token_ids=blocks, scoring=DECISION_SCORING,
            join_layout=(tuple(note), tuple(partner_label)),
            frame_tokens=frame), estimated.work
