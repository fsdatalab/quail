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
  measured constants): access per scan, and the KV dtype argmin.

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

    The tree is left-deep by construction (assemble_plan folds joins
    left to right), so an in-order walk recovers written order."""
    scans, filters, joins = [], {}, []

    def walk(node):
        if isinstance(node, Project):
            walk(node.input)
        elif isinstance(node, SemanticJoin):
            walk(node.left)
            walk(node.right)
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
                    pair_selectivity: float) -> float:
    """Expected distinct documents left with at least one matching
    pair: n * (1 - (1-s)^partners)."""
    if pair_selectivity is None:
        return n_docs
    return n_docs * (1.0 - (1.0 - pair_selectivity) ** max(1, n_partners))


def _stage_tokens(join, live: dict, stats: dict, anchor: str,
                  pre_tokens: int = 0) -> float:
    """Pair-token total of one stage at the current live counts: the
    anchor prefixes (engine preamble, document, and the stage's frame
    written into kept KV) once each, the partner document plus
    question tail once per pair. The frame is per anchor, never per
    pair - that is the point of writing it into kept KV."""
    a1, a2 = _join_sides(join)
    partner = a2 if anchor == a1 else a1
    pairs = live[anchor] * live[partner]
    frame = join.predicate.frame_tokens or 0
    tail = _question_tokens(join.predicate) - frame
    return (live[anchor] * (stats[anchor].mean_doc_tokens + pre_tokens
                            + frame)
            + pairs * (stats[partner].mean_doc_tokens + tail))


def _join_sides(join) -> tuple[str, str]:
    aliases = []
    for r in join.predicate.args:
        if r.alias not in aliases:
            aliases.append(r.alias)
    return aliases[0], aliases[1]


def order_joins(joins, rule: str, stats: dict, first_alias: str,
                anchors: dict, pre_tokens: int = 0):
    """Join stage order over the join graph: enumerate the connected
    permutations (2-4 relations, trivial) and compare survivor-thinned
    pair-token totals. All candidates run at the same rate, so the
    comparison is pure token counting."""
    if rule == "as_written" or len(joins) <= 1:
        return list(joins)

    def total(order):
        live = {a: float(s.n_docs) for a, s in stats.items()}
        scope = {first_alias}
        tokens = 0.0
        for j in order:
            a1, a2 = _join_sides(j)
            if a1 not in scope and a2 not in scope:
                return None    # disconnected: not a runnable order
            scope.update((a1, a2))
            anchor = anchors[id(j)]
            tokens += _stage_tokens(j, live, stats, anchor, pre_tokens)
            partner = a2 if anchor == a1 else a1
            n_a, n_p = live[anchor], live[partner]
            live[anchor] = _surviving_docs(n_a, n_p, j.selectivity)
            live[partner] = _surviving_docs(n_p, n_a, j.selectivity)
        return tokens

    best, best_tokens = list(joins), total(list(joins))
    for perm in itertools.permutations(joins):
        t = total(list(perm))
        if t is not None and (best_tokens is None or t < best_tokens):
            best, best_tokens = list(perm), t
    return best


# ---------------------------------------------- anchor (token arithmetic)

def choose_anchor(join, stats: dict,
                  pre_tokens: int = 0) -> tuple[str, list]:
    """The side whose pair-token total is smaller when the other side
    streams - in practice the longer side anchors (anchor tokens are
    paid once per document, partner tokens once per pair). A user
    override wins, with a remark when it prices worse."""
    a1, a2 = _join_sides(join)
    live = {a1: float(stats[a1].n_docs), a2: float(stats[a2].n_docs)}
    cost = {a: _stage_tokens(join, live, stats, a, pre_tokens)
            for a in (a1, a2)}
    planner_pick = a1 if cost[a1] <= cost[a2] else a2
    remarks = []
    if join.anchor is not None:
        if join.anchor != planner_pick:
            remarks.append(
                f"anchor {join.anchor!r} was forced; {planner_pick!r} "
                f"prices lower ({cost[planner_pick]:,.0f} vs "
                f"{cost[join.anchor]:,.0f} pair tokens)")
        return join.anchor, remarks
    return planner_pick, remarks


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


# ------------------------------------- access and dtype (the break-evens)

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


def choose_kv_dtype(model: ModelSpec, cal: Calibration,
                    fresh_tokens: float, restored_tokens: float,
                    store_bw: float | None,
                    overflow_recompute_saved_s: float = 0.0):
    """The section-6 argmin, one inequality per query:

        fp8 wins  iff  q_kv x fresh_tokens
                       <  (kappa_bf16 - kappa_fp8) / bw x restored_tokens
                          + overflow recompute saved by the 2x store

    A cold query has zero restored tokens, so bf16 wins it by the full
    conversion tax. Returns (dtype, fp8_cost_s, fp8_saving_s)."""
    tax = cal.q_kv_s_per_token * fresh_tokens
    saving = overflow_recompute_saved_s
    if store_bw and restored_tokens:
        elems = model.kv_elements_per_token
        saving += (2.0 - 1.0) * elems * restored_tokens / store_bw
    return ("fp8" if tax < saving else "bf16", tax, saving)


# ---------------------------------------------------------- the planner

def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               store: StoreSpec | None = None,
               kv_dtype: str | None = None,
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
    max_tail = max((_question_tokens(p.prompt)
                    for fs in filters.values() for p in fs),
                   default=0)
    max_tail = max([max_tail] + [_question_tokens(j.predicate)
                                 for j in joins])
    for s in scans:
        need = pre + stats[s.alias].max_doc_tokens + max_tail
        if need > chunk:
            return Refusal(
                reasons=(f"a document of {s.alias!r} plus the engine "
                         f"preamble and its question tail needs {need} "
                         f"tokens; the chunk budget is {chunk} and "
                         f"suffixes are atomic - no chunk can ever "
                         f"hold it",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    admission_bf16 = budgets.arena_tokens(model.with_kv_bytes(2.0),
                                          device, chunk)
    working_set = pre \
        + max(stats[s.alias].max_doc_tokens for s in scans) \
        + max_tail
    if admission_bf16 < working_set and store is None:
        return Refusal(
            reasons=(f"the arena holds {admission_bf16} tokens against "
                     f"a {working_set}-token working set and there is "
                     f"no store to spill to (cpu_memory_gb is 0)",),
            constraint="store_needed_but_disabled",
            needed=working_set, available=admission_bf16, unit="tokens")

    # ---- order
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)

    anchors = {}
    for j in joins:
        anchor, notes = choose_anchor(j, stats, pre)
        anchors[id(j)] = anchor
        remarks.extend(notes)
    first_alias = scans[0].alias
    ordered_joins = order_joins(joins, rule, stats, first_alias,
                                anchors, pre)

    # ---- KV dtype: the argmin over this query's fresh and restored
    # tokens (unless forced). Every KV-owning document pays the
    # engine preamble once, so scans count pre per document.
    accesses = {s.alias: access_for_scan(stats[s.alias], model, cal,
                                         store) for s in scans}
    fresh = sum(stats[s.alias].total_tokens
                + stats[s.alias].n_docs * pre for s in scans
                if accesses[s.alias] == "read")
    restored = sum(stats[s.alias].total_tokens
                   + stats[s.alias].n_docs * pre for s in scans
                   if accesses[s.alias] == "restore")
    live = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in order_filters(fs, rule):
            n = live[_filter_alias(p)]
            fresh += n * surv * _question_tokens(p.prompt)
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live[_filter_alias(fs[0])] *= surv
    join_token_counts = []
    for j in ordered_joins:
        anchor = anchors[id(j)]
        a1, a2 = _join_sides(j)
        partner = a2 if anchor == a1 else a1
        pairs = live[anchor] * live[partner]
        tokens = _stage_tokens(j, live, stats, anchor, pre)
        join_token_counts.append((pairs, tokens))
        # the anchor prefix term is already in the scan totals above
        fresh += tokens - live[anchor] * (stats[anchor].mean_doc_tokens
                                          + pre)
        n_a, n_p = live[anchor], live[partner]
        live[anchor] = _surviving_docs(n_a, n_p, j.selectivity)
        live[partner] = _surviving_docs(n_p, n_a, j.selectivity)

    if kv_dtype is not None:
        dtype, source_dtype = kv_dtype, f"forced kv_dtype={kv_dtype!r}"
    else:
        dtype, tax, saving = choose_kv_dtype(
            model, cal, fresh, restored,
            store.read_bw if store else None)
        source_dtype = (f"argmin: fp8 tax {tax:.2f} s vs byte saving "
                        f"{saving:.2f} s")
        if dtype == "fp8":
            # the fp8 arena's conversion kernels are not built yet;
            # the argmin's preference is recorded, not executed
            source_dtype += ("; fp8 priced lower but its arena is "
                             "not built - running bf16")
            dtype = "bf16"
    remarks.append(f"kv_dtype={dtype} ({source_dtype})")
    kv_bytes = 1.0 if dtype == "fp8" else 2.0
    admission = budgets.arena_tokens(model.with_kv_bytes(kv_bytes),
                                     device, chunk)

    store_min = 0
    if store is not None:
        # stored extents are [engine preamble + document] rows, so the
        # capacity arithmetic and the threshold are in those units
        all_lengths = [t + pre
                       for toks in doc_tokens.values() for t in toks]
        store_min = store_length_threshold(
            all_lengths, store.capacity_bytes,
            model.with_kv_bytes(kv_bytes).kappa)
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
            operators.append(dict(op="FilterChain", alias=s.alias,
                                  stages=stages))
    written = {id(j): i for i, j in enumerate(joins)}
    for j, (pairs, tokens) in zip(ordered_joins, join_token_counts):
        a1, a2 = _join_sides(j)
        anchor = anchors[id(j)]
        operators.append(dict(
            op="JoinStage", written_pos=written[id(j)], anchor=anchor,
            partner=a2 if anchor == a1 else a1,
            semantics=j.semantics, selectivity=j.selectivity,
            expected_pairs=round(pairs, 1),
            pair_tokens=round(tokens, 1)))
    operators.append(dict(
        op="Sink",
        columns=[f"{c.alias}.{c.column}" for c in plan.root.columns]))

    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        tensor_parallel=tp, kv_dtype=dtype, chunk_tokens=chunk,
        admission_tokens=admission, order_rule=rule, order_source=source,
        calibration_source=cal.source,
        store_min_doc_tokens=store_min,
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
            lines.append(f"{pad}Project [{cols}]")
            render(node.input, depth + 1)
        elif isinstance(node, SemanticJoin):
            lines.append(f"{pad}SemanticJoin ({node.semantics}, "
                         f"sel={node.selectivity}, anchor={node.anchor})")
            render(node.left, depth + 1)
            render(node.right, depth + 1)
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
