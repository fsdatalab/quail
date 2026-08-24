"""The planner's decisions, exactly the section-4 list and exactly
their inputs: token counts, provided selectivities, the two spec
structs, and the calibration constants - nothing else.

Grouped by input:

- Decisions from token arithmetic alone (the serving rate cancels out
  of every comparison, so these survive any miscalibration): filter
  order, join stage order, anchor per stage, sharding.
- Settings from the spec structs alone: the admission budget and the
  chunk budget (quail.planner.budgets).
- Decisions that compare compute against bytes (the only consumers of
  measured constants): access per scan. KV is always bf16.

Pushdown is not a decision at all: filters attach above their scans in
the logical plan, so a filter always runs before the joins its
provider feeds. There is no selectivity estimation and no runtime
re-ordering.
"""

import itertools

from quail.logical import (LogicalPlan, Project, Scan, SemanticFilter,
                           SemanticJoin)
from quail.planner import budgets
from quail.planner.calibration import Calibration, load_calibration
from quail.planner.plan import (CorpusStats, PhysicalPlan, Refusal,
                                StoreSpec)
from quail.specs import DeviceSpec, ModelSpec


# ---------------------------------------------------------- tree walk

def _collect(plan: LogicalPlan):
    """(scans, filters_by_alias, join_specs_in_written_order).

    Each join's first input is the accumulated tree (assemble_plan
    folds specs in written order), so an in-order walk recovers
    written order."""
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

    walk(plan.root)
    return scans, filters, joins


def _question_tokens(prompt) -> int:
    """The token count of a stage's question suffix per evaluation:
    the template text after the document placeholder. The engine
    preamble (before the placeholder) is paid once per KV-owning
    document, not per evaluation - see _preamble_tokens."""
    if prompt.tail_tokens is None or prompt.preamble_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return prompt.tail_tokens


def _preamble_tokens(filters, joins) -> int:
    """The engine preamble's token count. The canonical layout makes
    it identical on every prompt, so any bound prompt supplies it."""
    for fs in filters.values():
        for p in fs:
            if p.prompt.preamble_tokens is not None:
                return p.prompt.preamble_tokens
    for j in joins:
        if j.predicate.preamble_tokens is not None:
            return j.predicate.preamble_tokens
    return 0


# ------------------------------------------------ order (token arithmetic)

def default_order_rule(filters, joins) -> tuple[str, str]:
    """(rule, source). by_cost iff every gated predicate carries a
    selectivity, as_written otherwise."""
    preds = [p for fs in filters.values() for p in fs]
    sels = [p.selectivity for p in preds] + [j.selectivity for j in joins]
    if sels and all(s is not None for s in sels):
        return "by_cost", "default: every gated predicate has a selectivity"
    return "as_written", "default: at least one predicate has no selectivity"


def order_filters_indexed(predicates, rule: str):
    """Written positions of the predicates in execution order. by_cost
    sorts by cost per killed document: question tokens over
    (1 - selectivity). A selectivity of 1 kills nothing and goes last.
    Stable, so ties keep written order. Order changes cost, never
    results."""
    idx = list(range(len(predicates)))
    if rule == "as_written":
        return idx

    def cost(i):
        p = predicates[i]
        killed = 1.0 - (p.selectivity if p.selectivity is not None else 1.0)
        if killed <= 0:
            return float("inf")
        return _question_tokens(p.prompt) / killed

    return sorted(idx, key=cost)


def order_filters(predicates, rule: str):
    return [predicates[i] for i in order_filters_indexed(predicates,
                                                         rule)]


def _surviving_docs(n_docs: float, n_partners: float,
                    tuple_selectivity: float) -> float:
    """Expected distinct documents left with at least one matching
    tuple: n * (1 - (1-s)^partner_tuples)."""
    if tuple_selectivity is None:
        return n_docs
    return n_docs * (1.0 - (1.0 - tuple_selectivity)
                     ** max(1.0, n_partners))


def _join_aliases(join) -> list:
    """The join's table aliases, placeholder order (each placeholder
    names a distinct table, checked at bind time)."""
    return [r.alias for r in join.predicate.args]


def _label_counts(join) -> dict:
    """alias -> (block label tokens, anchor naming-line tokens), from
    the bind-time counts the prompt carries."""
    out = {a: (lt, nt) for a, lt, nt in join.predicate.labels}
    if any(lt is None for lt, _ in out.values()):
        raise ValueError(
            "join prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return out


def _cross_tuples(join, live: dict, anchor: str) -> float:
    tuples = live[anchor]
    for a in _join_aliases(join):
        if a != anchor:
            tuples *= live[a]
    return tuples


def _stage_tokens(join, live: dict, stats: dict, anchor: str,
                  pre_tokens: int = 0) -> float:
    """Token total of one join at the current live counts. The join
    is the cross product under one prompt: the anchor prefix (engine
    preamble, document, naming line - kept KV) is paid once per
    anchor document; every partner document with its block label,
    plus the rendered question, is paid once per tuple."""
    labels = _label_counts(join)
    partners = [a for a in _join_aliases(join) if a != anchor]
    tuples = _cross_tuples(join, live, anchor)
    per_tuple = _question_tokens(join.predicate)
    for p in partners:
        per_tuple += stats[p].mean_doc_tokens + labels[p][0]
    return (live[anchor] * (stats[anchor].mean_doc_tokens + pre_tokens
                            + labels[anchor][1])
            + tuples * per_tuple)


def _thin(live: dict, join, anchor: str) -> None:
    """Update live counts past one join, for cost enumeration only.
    full thins every table to the documents expected in some passing
    tuple; exists keeps matched anchors; anti keeps unmatched ones."""
    sel = join.selectivity
    aliases = _join_aliases(join)

    def others(x):
        out = 1.0
        for a in aliases:
            if a != x:
                out *= live[a]
        return out

    if join.semantics == "full":
        new = {a: _surviving_docs(live[a], others(a), sel)
               for a in aliases}
        live.update(new)
    elif sel is not None:
        matched = _surviving_docs(live[anchor], others(anchor), sel)
        live[anchor] = (matched if join.semantics == "exists"
                        else live[anchor] - matched)


def order_joins(joins, rule: str, stats: dict, anchors: dict,
                pre_tokens: int = 0):
    """Join order over the specs (the gates and the one full join):
    enumerate the permutations (2-4 specs, trivial) and compare
    survivor-thinned token totals. Every spec is self-contained - the
    cross product over its own tables - so every order runs; order
    changes cost, never results. All candidates run at the same rate,
    so the comparison is pure token counting."""
    if rule == "as_written" or len(joins) <= 1:
        return list(joins)

    def total(order):
        live = {a: float(s.n_docs) for a, s in stats.items()}
        tokens = 0.0
        for j in order:
            anchor = anchors[id(j)]
            tokens += _stage_tokens(j, live, stats, anchor, pre_tokens)
            _thin(live, j, anchor)
        return tokens

    best, best_tokens = list(joins), total(list(joins))
    for perm in itertools.permutations(joins):
        t = total(list(perm))
        if t < best_tokens:
            best, best_tokens = list(perm), t
    return best


# ---------------------------------------------- anchor (token arithmetic)

def choose_anchor(join, stats: dict,
                  pre_tokens: int = 0) -> tuple[str, list]:
    """The table whose token total is smallest when the others stream
    - in practice the side with the most document tokens anchors
    (anchor tokens are paid once per document, partner tokens once
    per tuple). A compile-time anchor (a user override, or the outer
    table of an exists/anti gate) wins, with a remark when it prices
    worse."""
    aliases = _join_aliases(join)
    live = {a: float(stats[a].n_docs) for a in aliases}
    cost = {a: _stage_tokens(join, live, stats, a, pre_tokens)
            for a in aliases}
    planner_pick = min(aliases, key=lambda a: cost[a])
    remarks = []
    if join.anchor is not None:
        if join.anchor != planner_pick \
                and cost[join.anchor] > cost[planner_pick]:
            remarks.append(
                f"anchor {join.anchor!r} was forced; {planner_pick!r} "
                f"prices lower ({cost[planner_pick]:,.0f} vs "
                f"{cost[join.anchor]:,.0f} tuple tokens)")
        return join.anchor, remarks
    return planner_pick, remarks


def choose_shared_anchor(joins, stats: dict,
                         pre_tokens: int = 0) -> tuple[str, list]:
    """One anchor for every full join: the table they all share.
    Anchoring every stage there keeps that table's KV resident across
    stages and needs no re-shard between them. When more than one
    table is shared, the one with the smallest summed stage tokens
    anchors. A forced anchor (a user override on any join) wins when
    it is a shared table; anything that would put two stages on
    different anchors needs the between-group re-shard, which is not
    built yet, and raises plainly."""
    shared = set(_join_aliases(joins[0]))
    for j in joins[1:]:
        shared &= set(_join_aliases(j))
    if not shared:
        raise NotImplementedError(
            "multiple full joins that share no table need the "
            "between-group re-shard, which is not built yet")
    forced = {j.anchor for j in joins if j.anchor is not None}
    if len(forced) > 1 or (forced and not forced <= shared):
        raise NotImplementedError(
            f"forced anchors {sorted(forced)} would put the full "
            f"join stages on different anchors; the between-group "
            f"re-shard is not built yet (shared tables: "
            f"{sorted(shared)})")
    if forced:
        (anchor,) = forced
        return anchor, []
    live = {a: float(s.n_docs) for a, s in stats.items()}

    def total(a):
        return sum(_stage_tokens(j, live, stats, a, pre_tokens)
                   for j in joins)

    return min(sorted(shared), key=total), []


# --------------------------------------------- sharding (token arithmetic)

def balanced_shards(doc_tokens, workers: int):
    """Greedy balance by token count: makespan is the slowest shard.
    Filters split documents; joins split anchor documents (every pair
    belongs to exactly one anchor, so gating, dedup, and the next
    stage's pair list stay local to the GPU holding the anchor)."""
    order = sorted(range(len(doc_tokens)), key=lambda i: -doc_tokens[i])
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(i)
        loads[w] += doc_tokens[i]
    return tuple(tuple(sorted(s)) for s in shards), loads


# ------------------------------------- access (the one break-even)

def restore_crossover_tokens(model: ModelSpec, cal: Calibration,
                             read_bw: float) -> float:
    """The document length past which loading KV beats recomputing it:
    solve kappa/bw = a + a2*h for h. 0 means the channel beats
    recompute at every length (bandwidth above kappa/a - the store
    break-even); a2 moves the crossover down for long documents."""
    per_token_load = model.kappa / read_bw
    if per_token_load <= cal.a_s_per_token:
        return 0.0
    return (per_token_load - cal.a_s_per_token) / cal.a2_s_per_token2


def store_length_threshold(doc_tokens, capacity_bytes, kappa) -> int:
    """The store-or-not length cutoff under a capacity: keep the
    LONGEST documents whose KV fits. Recompute cost per byte rises
    with document length, so the top of the length list is worth the
    most per stored byte - and a length threshold cannot be thrashed
    by a scanning query the way LRU can.

    Returns 1 when capacity holds everything, 0 when nothing fits.
    At a boundary tie the threshold moves up so the stored set never
    exceeds the capacity."""
    if capacity_bytes is None:
        return 1
    budget_tok = int(capacity_bytes / kappa)
    lengths = sorted((int(t) for t in doc_tokens), reverse=True)
    if sum(lengths) <= budget_tok:
        return 1
    taken, threshold = 0, 0
    for h in lengths:
        if taken + h > budget_tok:
            break
        taken += h
        threshold = h
    while threshold and sum(h for h in lengths
                            if h >= threshold) > budget_tok:
        threshold += 1
    if threshold and not any(h >= threshold for h in lengths):
        return 0
    return threshold


def access_for_scan(stats: CorpusStats, model: ModelSpec,
                    cal: Calibration, store) -> str:
    """read | restore. Restore wins when the warm store's bandwidth
    beats kappa x the serving rate, or when the corpus's documents sit
    past the a2-refined crossover."""
    if store is None or not store.warm:
        return "read"
    crossover = restore_crossover_tokens(model, cal, store.read_bw)
    return "restore" if stats.mean_doc_tokens >= crossover else "read"


# ---------------------------------------------------------- the planner

def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               store: StoreSpec | None = None,
               order: str | None = None,
               calibration: Calibration | None = None):
    """LogicalPlan + corpus token counts -> PhysicalPlan | Refusal.

    doc_tokens: alias -> per-document token counts (from the cached
    tokenization of each scan's column)."""
    cal = calibration or load_calibration(model, device)
    scans, filters, joins = _collect(plan)
    stats = {a: CorpusStats.from_doc_tokens(t)
             for a, t in doc_tokens.items()}
    for s in scans:
        if s.alias not in stats:
            raise ValueError(f"no doc_tokens for alias {s.alias!r}")
    remarks = []

    # ---- refusals first: infeasible configurations are named, not
    # planned around
    tp = budgets.tensor_parallel(model, device)
    if tp > gpus:
        return Refusal(
            reasons=(f"weights need {tp} cards, {gpus} available",),
            constraint="weights_need_more_cards",
            needed=tp, available=gpus, unit="cards")
    workers = max(1, gpus // tp)

    chunk = budgets.chunk_budget(model, device)
    pre = _preamble_tokens(filters, joins)

    # anchors come before the chunk refusal: a join's atomic suffix
    # (every partner document plus the question) depends on which
    # table anchors
    anchors = {}
    full_joins = [j for j in joins if j.semantics == "full"]
    if len(full_joins) > 1:
        # multiple full joins run as one same-anchor group; anchoring
        # them elsewhere needs the unbuilt between-group re-shard
        anchor, notes = choose_shared_anchor(full_joins, stats, pre)
        for j in full_joins:
            anchors[id(j)] = anchor
        remarks.extend(notes)
    for j in joins:
        if id(j) in anchors:
            continue
        anchor, notes = choose_anchor(j, stats, pre)
        anchors[id(j)] = anchor
        remarks.extend(notes)

    needs = []      # the largest single admission each operator makes
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
        needs.append(need)
    for j in joins:
        anchor = anchors[id(j)]
        labels = _label_counts(j)
        partners = [a for a in _join_aliases(j) if a != anchor]
        need = (pre + stats[anchor].max_doc_tokens + labels[anchor][1]
                + sum(labels[p][0] + stats[p].max_doc_tokens
                      for p in partners)
                + _question_tokens(j.predicate))
        if need > chunk:
            return Refusal(
                reasons=(f"one tuple of the join anchored on "
                         f"{anchor!r} needs {need} tokens (the anchor "
                         f"document, every partner document with its "
                         f"label, and the question, all in one "
                         f"prompt); the chunk budget is {chunk} and "
                         f"suffixes are atomic - no chunk can ever "
                         f"hold it",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")
        needs.append(need)

    admission = budgets.arena_tokens(model, device, chunk)
    working_set = max(needs)
    if admission < working_set and store is None:
        return Refusal(
            reasons=(f"the arena holds {admission} tokens against "
                     f"a {working_set}-token working set and there is "
                     f"no store to spill to (cpu_memory_gb is 0)",),
            constraint="store_needed_but_disabled",
            needed=working_set, available=admission, unit="tokens")

    # ---- order
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)
    ordered_joins = order_joins(joins, rule, stats, anchors, pre)

    accesses = {s.alias: access_for_scan(stats[s.alias], model, cal,
                                         store) for s in scans}
    live = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in order_filters(fs, rule):
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live[_filter_alias(fs[0])] *= surv
    join_token_counts = []
    for j in ordered_joins:
        anchor = anchors[id(j)]
        tuples = _cross_tuples(j, live, anchor)
        tokens = _stage_tokens(j, live, stats, anchor, pre)
        join_token_counts.append((tuples, tokens))
        _thin(live, j, anchor)

    remarks.append("kv_dtype=bf16 (always)")

    store_min = 0
    if store is not None:
        # stored extents are [engine preamble + document] rows, so the
        # capacity arithmetic and the threshold are in those units
        all_lengths = [t + pre
                       for toks in doc_tokens.values() for t in toks]
        store_min = store_length_threshold(
            all_lengths, store.capacity_bytes, model.kappa)
        if store_min > 1:
            stored = [t for t in all_lengths if t >= store_min]
            remarks.append(
                f"store capped: {len(stored)} of {len(all_lengths)} "
                f"documents stored, length threshold {store_min} "
                f"(preamble included)")

    # ---- operators, in execution order
    operators = []
    for s in scans:
        shards, loads = balanced_shards(doc_tokens[s.alias], workers)
        operators.append(dict(
            op="DocScan", alias=s.alias, provider=s.provider,
            column=s.column, access=accesses[s.alias],
            n_docs=stats[s.alias].n_docs,
            total_tokens=stats[s.alias].total_tokens,
            shards=shards, shard_token_loads=loads))
        if s.alias in filters:
            order_idx = order_filters_indexed(filters[s.alias], rule)
            n = stats[s.alias].n_docs
            stages, surv = [], 1.0
            for i in order_idx:
                p = filters[s.alias][i]
                stages.append(dict(
                    written_pos=i,
                    question_tokens=_question_tokens(p.prompt),
                    preamble_tokens=p.prompt.preamble_tokens,
                    selectivity=p.selectivity,
                    expected_docs=round(n * surv, 1)))
                surv *= p.selectivity if p.selectivity is not None else 1.0
            # a lone stage with no store leaves every document's KV
            # with no reader: later stages re-read it, store.save
            # copies it out, and nothing else ever touches it
            writes = len(stages) > 1 or store is not None
            operators.append(dict(op="FilterChain", alias=s.alias,
                                  arena_writes=writes, stages=stages))
            if not writes:
                remarks.append(
                    f"filter on {s.alias!r}: arena writes off (one "
                    f"stage, no store - nothing reads the KV again)")
    written = {id(j): i for i, j in enumerate(joins)}
    for j, (tuples, tokens) in zip(ordered_joins, join_token_counts):
        anchor = anchors[id(j)]
        operators.append(dict(
            op="JoinStage", written_pos=written[id(j)], anchor=anchor,
            partners=[a for a in _join_aliases(j) if a != anchor],
            semantics=j.semantics, selectivity=j.selectivity,
            expected_tuples=round(tuples, 1),
            tuple_tokens=round(tokens, 1)))
    operators.append(dict(
        op="Sink",
        columns=[f"{c.alias}.{c.column}" for c in plan.root.columns]))

    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        tensor_parallel=tp, kv_dtype="bf16", chunk_tokens=chunk,
        admission_tokens=admission, order_rule=rule, order_source=source,
        calibration_source=cal.source,
        store_min_doc_tokens=store_min,
        limit=plan.root.limit,
        operators=tuple(operators), remarks=tuple(remarks))


def _filter_alias(pred_or_list):
    p = pred_or_list[0] if isinstance(pred_or_list, list) else pred_or_list
    return p.prompt.args[0].alias


# ------------------------------------------------------------- explain

def explain(logical: LogicalPlan, physical) -> str:
    """The logical tree, the chosen physical plan, per-operator
    settings and token counts - no run needed."""
    lines = ["logical:"]

    def render(node, depth):
        pad = "  " * (depth + 1)
        if isinstance(node, Project):
            cols = ", ".join(f"{c.alias}.{c.column}" for c in node.columns)
            lim = f" LIMIT {node.limit}" if node.limit is not None else ""
            lines.append(f"{pad}Project [{cols}]{lim}")
            render(node.input, depth + 1)
        elif isinstance(node, SemanticJoin):
            lines.append(f"{pad}SemanticJoin ({node.semantics}, "
                         f"x{len(node.inputs)} inputs, "
                         f"sel={node.selectivity}, anchor={node.anchor})")
            for child in node.inputs:
                render(child, depth + 1)
        elif isinstance(node, SemanticFilter):
            sels = [p.selectivity for p in node.predicates]
            lines.append(f"{pad}SemanticFilter (x{len(node.predicates)}, "
                         f"sels={sels})")
            render(node.input, depth + 1)
        else:
            lines.append(f"{pad}Scan {node.provider} as {node.alias} "
                         f"[{node.column}]")

    render(logical.root, 0)
    if isinstance(physical, Refusal):
        lines.append(f"refusal: {physical.constraint}: needed "
                     f"{physical.needed} {physical.unit}, available "
                     f"{physical.available}")
        for r in physical.reasons:
            lines.append(f"  {r}")
        return "\n".join(lines)
    lines.append("physical:")
    lines.append(f"  workers={physical.workers} "
                 f"tp={physical.tensor_parallel} "
                 f"kv_dtype={physical.kv_dtype}")
    lines.append(f"  chunk_tokens={physical.chunk_tokens} "
                 f"admission_tokens={physical.admission_tokens}")
    if physical.limit is not None:
        lines.append(f"  limit={physical.limit}")
    lines.append("  prompt layout: engine preamble + document + "
                 "suffix (preamble_tokens per stage below count the "
                 "shared preamble)")
    lines.append(f"  order={physical.order_rule} ({physical.order_source})")
    lines.append(f"  calibration: {physical.calibration_source}")
    for op in physical.operators:
        parts = [f"  {op['op']}"]
        for k, v in op.items():
            if k in ("op", "shards", "shard_token_loads", "stages"):
                continue
            parts.append(f"{k}={v}")
        lines.append(" ".join(parts))
        for st in op.get("stages", []):
            lines.append(f"    stage {st}")
    for r in physical.remarks:
        lines.append(f"  remark: {r}")
    return "\n".join(lines)
