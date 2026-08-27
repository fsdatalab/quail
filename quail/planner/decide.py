"""Planner decisions: filter order, join order, anchor choice, KV
residency, and sharding from a logical plan and corpus token counts.

The join search itself lives in quail.planner.joins and runs twice
per query: here with expectations (the predicted plan - explain,
refusals, the SoL comparison), and in the worker after the filter
round with the actual survivors and resident KV (the executed plan).
Costs are Work records priced by counted constants; no calibration
constant is read anywhere.
"""

from quail.logical import (LogicalPlan, Project, Scan, SemanticFilter,
                           SemanticJoin)
from quail.planner import budgets, joins as joinsearch, sol
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
from quail.planner.sol import Work
from quail.specs import DeviceSpec, ModelSpec


# ---------------------------------------------------------- tree walk

def _collect(plan: LogicalPlan):
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

    walk(plan.root)
    return scans, filters, joins


def _question_tokens(prompt) -> int:
    """Token count of the prompt's per-evaluation tail."""
    if prompt.tail_tokens is None or prompt.preamble_tokens is None:
        raise ValueError(
            "prompts were bound without a tokenizer; the planner "
            "needs token counts (pass one to compile_sql / docs)")
    return prompt.tail_tokens


def _preamble_tokens(filters, joins) -> int:
    """Return the engine preamble's token count from any bound prompt."""
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
    """Return (rule, source). 'by_cost' when every predicate has a
    selectivity, 'as_written' otherwise."""
    preds = [p for fs in filters.values() for p in fs]
    sels = [p.selectivity for p in preds] + [j.selectivity for j in joins]
    if sels and all(s is not None for s in sels):
        return "by_cost", "default: every gated predicate has a selectivity"
    return "as_written", "default: at least one predicate has no selectivity"


def order_filters_indexed(predicates, rule: str):
    """Return written positions of predicates in execution order.

    'by_cost' sorts by question_tokens / (1 - selectivity). Stable
    sort, so ties keep written order.
    """
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


def _filter_work(filters, stats, rule: str, pre: int) -> Work:
    """Expected Work of every filter chain: the first stage scans
    each document, later stages ask over resident KV."""
    total = Work()
    for alias, preds in filters.items():
        mean = stats[alias].mean_doc_tokens
        n = float(stats[alias].n_docs)
        for si, p in enumerate(order_filters(preds, rule)):
            q = _question_tokens(p.prompt)
            op = sol.scan if si == 0 else sol.ask
            total = total + op(pre + mean, q) * n
            n *= p.selectivity if p.selectivity is not None else 1.0
    return total


# ----------------------------------------------- KV keep (residency)

def _page_round(tokens: float, page_tokens: int) -> float:
    return -(-tokens // page_tokens) * page_tokens


def _split_at(doc_tokens, threshold: int, survivor_frac: float,
              overhead: int, page_tokens: int) -> dict | None:
    """The keep record for one alias at a given length threshold:
    documents at or above it are credited as resident."""
    lengths = [int(t) for t in doc_tokens]
    kept = [t for t in lengths if t >= max(1, threshold)]
    if not kept:
        return None
    unkept = [t for t in lengths if t < max(1, threshold)]
    return dict(
        min_doc_tokens=1 if not unkept else threshold,
        kept_expected_tokens=survivor_frac * sum(
            _page_round(t + overhead, page_tokens) for t in kept),
        survivor_frac=survivor_frac,
        overhead=overhead)


def _raise_threshold(split: dict, doc_tokens, page_tokens: int):
    """Drop the shortest kept length class: the next split up, or
    None when only the longest class was left."""
    boundary = min(t for t in doc_tokens
                   if t >= max(1, split["min_doc_tokens"]))
    higher = sorted({int(t) for t in doc_tokens if t > boundary})
    if not higher:
        return None
    return _split_at(doc_tokens, higher[0], split["survivor_frac"],
                     split["overhead"], page_tokens)


def keep_split(doc_tokens, budget_tokens: float, survivor_frac: float,
               overhead: int, page_tokens: int) -> dict | None:
    """Which survivors of one alias to credit as resident for a join.

    Longest documents first. Resident KV of length L saves L dense
    tokens (2 FLOPs per parameter each, against the fp8 peak) plus
    L(L+1)/2 attention pairs (against the bf16 peak) - both counted,
    no measured constant - while occupying kappa * L bytes. Saved
    work per byte rises with L under any positive weighting of the
    two terms, so length orders the documents; the rise comes from
    the attention term and is small below the dense/attention
    crossover (about 12,300 tokens at 4B), where the linear dense
    term dominates. What makes longest-first exact rather than a
    heuristic is that survival is unknown per document at plan time:
    the expected kept mass is the survivor fraction of the kept
    lengths' mass, a fractional knapsack, where taking by value per
    byte is optimal.

    The runtime is not bound by the threshold: it retains every
    passing survivor and evicts by recompute value under pressure.
    This split is the capacity-planned credit the cost model and the
    SoL comparison use.

    Returns:
        None when not even the longest document fits, else the
        _split_at record; min_doc_tokens is 1 when everything fits.
    """
    lengths = sorted((int(t) for t in doc_tokens), reverse=True)
    if not lengths:
        return None
    taken, threshold = 0.0, 0
    for t in lengths:
        cost = survivor_frac * _page_round(t + overhead, page_tokens)
        if taken + cost > budget_tokens:
            break
        taken += cost
        threshold = t
    if threshold == 0:
        return None
    split = _split_at(doc_tokens, threshold, survivor_frac, overhead,
                      page_tokens)
    while split is not None and \
            split["kept_expected_tokens"] > budget_tokens:
        split = _raise_threshold(split, doc_tokens, page_tokens)
    return split


def possible_anchor_aliases(specs) -> set:
    """Every alias the runtime search could anchor a join on."""
    out = set()
    for spec in specs:
        out.update(joinsearch.anchor_candidates(spec))
    return out


def plan_keeps(specs, filters, doc_tokens: dict, pre: int,
               budget_tokens: float, page_tokens: int) -> dict:
    """Candidate keep credit: every filtered alias the runtime could
    anchor, each split against the whole budget. plan_query trims to
    the aliases the predicted plan anchors and re-checks the joint
    capacity."""
    plan = {}
    anchors = possible_anchor_aliases(specs)
    for alias, preds in filters.items():
        if alias not in anchors:
            continue
        frac = 1.0
        for p in preds:
            frac *= p.selectivity if p.selectivity is not None else 1.0
        split = keep_split(doc_tokens[alias], budget_tokens, frac,
                           pre, page_tokens)
        if split is not None:
            plan[alias] = split
    return plan


def _group_seq(seq):
    """Group consecutive full stages on the same anchor, the same
    rule the node graph uses. seq holds (spec, anchor) pairs."""
    groups = []
    for spec, anchor in seq:
        merge = (groups and spec["semantics"] == "full"
                 and groups[-1][2] and groups[-1][0] == anchor)
        if merge:
            groups[-1][1].append(spec)
        else:
            groups.append([anchor, [spec], spec["semantics"] == "full"])
    return [(a, m) for a, m, _ in groups]


def _keep_timeline(seq, keep_plan, doc_tokens, live0, pre,
                   page_tokens, workers: int):
    """Peak expected resident tokens per worker across the plan.

    A point per group boundary: the end of the filter round holds
    every credited alias's filter mass; after each group, credited
    masses not yet anchored plus the gate-survivor mass of anchors a
    later group re-uses (retention holds every gate survivor, not
    just the credited split). Within a group, anchors run one at a
    time, so the per-anchor working set is the headroom the caller
    adds. Retained prefixes are rewound to preamble + document.
    """
    groups = _group_seq(seq)
    first_use, last_use = {}, {}
    for g, (anchor, members) in enumerate(groups):
        first_use.setdefault(anchor, g)
        last_use[anchor] = g

    def survivor_mass(alias, live_count):
        frac = live_count / max(1.0, float(len(doc_tokens[alias])))
        return frac * sum(_page_round(pre + t, page_tokens)
                          for t in doc_tokens[alias])

    live = dict(live0)
    points = [sum(k["kept_expected_tokens"]
                  for k in keep_plan.values())]
    for g, (anchor, members) in enumerate(groups):
        for spec in members:
            joinsearch.thin(live, spec)
        point = 0.0
        for alias in set(keep_plan) | set(first_use):
            if alias in keep_plan and first_use.get(
                    alias, len(groups)) > g:
                point += keep_plan[alias]["kept_expected_tokens"]
            elif first_use.get(alias, g + 1) <= g \
                    < last_use.get(alias, -1):
                point += survivor_mass(alias, live[alias])
        points.append(point)
    return max(points) / workers


# --------------------------------------------- sharding (token arithmetic)

def balanced_shards(doc_tokens, workers: int):
    """Greedily partition documents into shards balanced by token count."""
    order = sorted(range(len(doc_tokens)), key=lambda i: -doc_tokens[i])
    loads = [0] * workers
    shards = [[] for _ in range(workers)]
    for i in order:
        w = loads.index(min(loads))
        shards[w].append(i)
        loads[w] += doc_tokens[i]
    return tuple(tuple(sorted(s)) for s in shards), loads


# ---------------------------------------------------------- the planner

def plan_query(plan: LogicalPlan, *, model: ModelSpec,
               device: DeviceSpec, doc_tokens: dict, gpus: int = 1,
               order: str | None = None):
    """Compile a LogicalPlan into a PhysicalPlan or Refusal.

    Args:
        doc_tokens: alias -> list of per-document token counts.
    """
    scans, filters, joins = _collect(plan)
    stats = {a: CorpusStats.from_doc_tokens(t)
             for a, t in doc_tokens.items()}
    for s in scans:
        if s.alias not in stats:
            raise ValueError(f"no doc_tokens for alias {s.alias!r}")
    remarks = []

    # ---- refusals first
    tp = budgets.tensor_parallel(model, device)
    if tp > gpus:
        return Refusal(
            reasons=(f"weights need {tp} cards, {gpus} available",),
            constraint="weights_need_more_cards",
            needed=tp, available=gpus, unit="cards")
    workers = max(1, gpus // tp)

    chunk = budgets.chunk_budget(model, device)
    admission = budgets.arena_tokens(model, device, chunk)
    pre = _preamble_tokens(filters, joins)
    specs = join_specs(joins)

    # ---- the order rule first: the search below needs it
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)
    fixed = rule == "as_written"

    # ---- expected live counts after filters, and the fixed filter
    # work every candidate join plan shares
    live0 = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in fs:
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live0[_filter_alias(fs[0])] *= surv
    base_work = _filter_work(filters, stats, rule, pre)

    # ---- the largest single admission any operator makes: the keep
    # arithmetic reserves it as working headroom
    headroom = 0
    for s in scans:
        fq = max((_question_tokens(p.prompt)
                  for p in filters.get(s.alias, ())), default=0)
        if fq:
            headroom = max(headroom,
                           pre + stats[s.alias].max_doc_tokens + fq)
    for spec in specs:
        for a in spec["aliases"]:
            headroom = max(
                headroom,
                pre + stats[a].max_doc_tokens + spec["frame_tokens"][a]
                + sum(spec["label_tokens"][p] + stats[p].max_doc_tokens
                      for p in spec["aliases"] if p != a)
                + spec["tail_tokens"])

    # ---- keep credit candidates, then the search on expectations
    keep_budget = max(0.0, float(admission - headroom)) * workers
    candidates = plan_keeps(specs, filters, doc_tokens, pre,
                            keep_budget, budgets.PAGE_TOKENS)

    def resident_from(plan_keep):
        return {alias: {i for i, t in enumerate(doc_tokens[alias])
                        if t >= max(1, k["min_doc_tokens"])}
                for alias, k in plan_keep.items()}

    def run_search(plan_keep, honor_forced=True):
        found = joinsearch.search_joins(
            specs, live0, doc_tokens, resident_from(plan_keep), pre,
            chunk, model, device, base_work=base_work,
            fixed_order=fixed, honor_forced=honor_forced)
        if found is None:
            # a join predicate with no alias in common with the rest:
            # no connected left deep order exists, so cost the written
            # order directly
            found = joinsearch.search_joins(
                specs, live0, doc_tokens, resident_from(plan_keep),
                pre, chunk, model, device, base_work=base_work,
                fixed_order=True, honor_forced=honor_forced)
        return found

    def consumed_keeps(records, plan_keep):
        """The credits the sequence anchors while their filter KV is
        still resident; the rest have no reader in the prediction."""
        used = {r["anchor"] for r in records
                if r["resident"] == "filter"}
        return {a: k for a, k in plan_keep.items() if a in used}

    def spec_seq(found):
        return [(specs[wp], a) for wp, a in found["seq"]]

    def trim(found, plan_keep):
        """Raise thresholds until the peak expected resident tokens
        per worker fit the arena beside the working headroom. The
        shortest credited documents go first, wherever they are."""
        plan_keep = dict(plan_keep)
        while plan_keep:
            peak = _keep_timeline(spec_seq(found), plan_keep,
                                  doc_tokens, live0, pre,
                                  budgets.PAGE_TOKENS, workers)
            if peak + headroom <= admission:
                break
            alias = min(
                plan_keep,
                key=lambda a: min(t for t in doc_tokens[a]
                                  if t >= max(1, plan_keep[a]
                                              ["min_doc_tokens"])))
            split = _raise_threshold(plan_keep[alias],
                                     doc_tokens[alias],
                                     budgets.PAGE_TOKENS)
            if split is None:
                del plan_keep[alias]
            else:
                plan_keep[alias] = split
        return plan_keep

    found = run_search(candidates)
    kept0 = consumed_keeps(found["records"], candidates)
    keep_plan = trim(found, kept0)
    if keep_plan != kept0:
        # the credited residency shrank: search once more against
        # what the arena can actually hold
        found = run_search(keep_plan)
        keep_plan = trim(found, consumed_keeps(found["records"],
                                               keep_plan))
        found = run_search(keep_plan)
    forced = sorted({s["anchor"] for s in specs
                     if s["semantics"] == "full"
                     and not s["anchor_free"]})
    if forced:
        free = run_search(keep_plan, honor_forced=False)
        honored_s = sol.speed_of_light(
            base_work + found["work"], model, device, chunk).seconds
        free_s = sol.speed_of_light(
            base_work + free["work"], model, device, chunk).seconds
        if free_s < honored_s:
            remarks.append(
                f"anchors {forced} were forced; a free choice prices "
                f"lower ({free_s:.3f} vs {honored_s:.3f} predicted "
                f"seconds)")
    seq = spec_seq(found)
    stage_records = found["records"]

    for alias, k in sorted(keep_plan.items()):
        what = ("all survivors" if k["min_doc_tokens"] <= 1 else
                f"survivors of {k['min_doc_tokens']}+ tokens")
        remarks.append(
            f"keep KV on {alias!r}: {what} priced as resident for "
            f"the join, {k['kept_expected_tokens'] / max(1, workers):,.0f} "
            f"expected tokens per worker of the {admission:,}-token "
            f"arena")

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
                reasons=(f"one tuple of the join anchored on "
                         f"{anchor!r} needs {need} tokens (the anchor "
                         f"document, every partner document with its "
                         f"label, and the question, all in one "
                         f"prompt); the chunk budget is {chunk} and "
                         f"suffixes are atomic - no chunk can ever "
                         f"hold it",),
                constraint="suffix_over_chunk",
                needed=need, available=chunk, unit="tokens")

    remarks.append("kv_dtype=bf16 (always)")

    # ---- build the dataflow graph; ids_src tracks each table's
    # current producer node
    retain_aliases = possible_anchor_aliases(specs) & set(filters)
    nodes = []
    ids_src = {}
    for s in scans:
        shards, loads = balanced_shards(doc_tokens[s.alias], workers)
        sid = f"scan:{s.alias}"
        nodes.append(dict(
            id=sid, op="DocScan", inputs=(),
            alias=s.alias, provider=s.provider,
            column=s.column,
            n_docs=stats[s.alias].n_docs,
            total_tokens=stats[s.alias].total_tokens,
            shards=shards, shard_token_loads=loads))
        ids_src[s.alias] = (sid, f"ids:{s.alias}")
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
            # arena writes when a later stage reads the KV back or
            # the runtime search could anchor a join on this table
            keep = s.alias in retain_aliases
            writes = len(stages) > 1 or keep
            credit = keep_plan.get(s.alias)
            fid = f"filter:{s.alias}"
            nodes.append(dict(
                id=fid, op="FilterChain",
                inputs=(ids_src[s.alias],),
                alias=s.alias, arena_writes=writes,
                keep_kv=keep,
                # the capacity-planned credit; the runtime retains
                # every survivor and evicts by value under pressure
                keep_min_doc_tokens=(credit["min_doc_tokens"]
                                     if credit else 0),
                stages=tuple(stages)))
            ids_src[s.alias] = (fid, f"ids:{s.alias}")
            if not writes:
                remarks.append(
                    f"filter on {s.alias!r}: arena writes off (one "
                    f"stage - nothing reads the KV again)")

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
    group_last_use = {}
    for g, group in enumerate(groups):
        group_last_use[group["anchor"]] = g

    exec_idx = 0
    barrier_n = 0
    prev_anchor = None
    pairs_edges = []     # every full stage's passing-pairs edge
    out_aliases = []     # recombination's output order
    for g, group in enumerate(groups):
        anchor = group["anchor"]
        if prev_anchor is not None and anchor != prev_anchor:
            # barrier: thin tables the remaining stages touch, re-shard
            # the new anchor over the live set
            ahead = []
            for later in groups[g:]:
                for spec, _ in later["members"]:
                    for a in spec["aliases"]:
                        if a not in ahead:
                            ahead.append(a)
            bid = f"barrier:{barrier_n}"
            barrier_n += 1
            nodes.append(dict(
                id=bid, op="Barrier",
                inputs=tuple(pairs_edges)
                + tuple(ids_src[a] for a in ahead),
                next_anchor=anchor, thins=tuple(ahead)))
            for a in ahead:
                ids_src[a] = (bid, f"ids:{a}")
        gid = f"group:{g}"
        stage_dicts = []
        in_aliases = [anchor]
        for spec, record in group["members"]:
            partners = [a for a in spec["aliases"] if a != anchor]
            stage_dicts.append(dict(
                written_pos=spec["written_pos"], exec_idx=exec_idx,
                anchor=anchor, partners=partners,
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
                pairs_edges.append((gid,
                                    f"pairs:{spec['written_pos']}"))
                for a in [anchor] + partners:
                    if a not in out_aliases:
                        out_aliases.append(a)
        nodes.append(dict(
            id=gid, op="JoinGroup",
            inputs=tuple(ids_src[a] for a in in_aliases),
            anchor=anchor,
            anchor_resident=group["members"][0][1]["resident"],
            keep_anchor_kv=group_last_use[anchor] > g,
            stage_idxs=tuple(s["exec_idx"] for s in stage_dicts),
            stages=tuple(stage_dicts)))
        ids_src[anchor] = (gid, f"ids:{anchor}")
        prev_anchor = anchor

    if pairs_edges:
        nodes.append(dict(
            id="recombine", op="Recombine",
            inputs=tuple(pairs_edges)
            + tuple(ids_src[a] for a in out_aliases),
            alias_order=tuple(out_aliases)))
        sink_inputs = (("recombine", "tuples"),)
    else:
        sink_inputs = (ids_src[scans[0].alias],)
    nodes.append(dict(
        id="sink", op="Sink", inputs=sink_inputs,
        columns=[f"{c.alias}.{c.column}" for c in plan.root.columns]))

    return PhysicalPlan(
        model=model.name, device=device.name, workers=workers,
        tensor_parallel=tp, kv_dtype="bf16", chunk_tokens=chunk,
        admission_tokens=admission, order_rule=rule, order_source=source,
        limit=plan.root.limit,
        nodes=tuple(nodes), remarks=tuple(remarks))


def _filter_alias(pred_or_list):
    p = pred_or_list[0] if isinstance(pred_or_list, list) else pred_or_list
    return p.prompt.args[0].alias


# ------------------------------------------------------------- explain

def explain(logical: LogicalPlan, physical) -> str:
    """Format the logical and physical plan as a human-readable string."""
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
    lines.append("  the predicted join order; the worker re-runs the "
                 "same search on the actual filter survivors")
    lines.append(f"  order={physical.order_rule} ({physical.order_source})")
    for n in physical.nodes:
        parts = [f"  {n['op']} {n['id']}"]
        for k, v in n.items():
            if k in ("op", "id", "inputs", "shards",
                     "shard_token_loads", "stages"):
                continue
            parts.append(f"{k}={v}")
        if n.get("inputs"):
            parts.append("<- " + ", ".join(
                f"{src}[{port}]" for src, port in n["inputs"]))
        lines.append(" ".join(parts))
        for st in n.get("stages", []):
            lines.append(f"    stage {st}")
    for r in physical.remarks:
        lines.append(f"  remark: {r}")
    return "\n".join(lines)
