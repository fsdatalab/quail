"""Planner decisions: filter order, join order, anchor choice, KV
residency, and sharding from a logical plan and corpus token counts.

Join costs are expected-value `Work` records (quail.planner.sol):
tokens, attention pairs, KV written, KV read. Candidate plans are
ranked by the speed-of-light seconds those counts price to, from
counted model constants and the device datasheet - no measured or
fitted constant anywhere.
"""

import itertools

from quail.logical import (LogicalPlan, Project, Scan, SemanticFilter,
                           SemanticJoin)
from quail.planner import budgets, sol
from quail.planner.leftdeep import Extension, optimize_left_deep
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


def _surviving_docs(n_docs: float, n_partners: float,
                    tuple_selectivity: float) -> float:
    """Expected distinct documents with at least one matching tuple."""
    if tuple_selectivity is None:
        return n_docs
    return n_docs * (1.0 - (1.0 - tuple_selectivity)
                     ** max(1.0, n_partners))


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


def _cross_tuples(join, live: dict, anchor: str) -> float:
    tuples = live[anchor]
    for a in _join_aliases(join):
        if a != anchor:
            tuples *= live[a]
    return tuples


def _thin(live: dict, join, anchor: str) -> None:
    """Update live document counts after one join's selectivity."""
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


# ------------------------------------------------ expected stage work

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


def _anchor_work(join, live: dict, stats: dict, anchor: str, pre: int,
                 kind, keep_plan: dict) -> Work:
    """Expected Work to put live anchor prefixes and this stage's
    frame in the arena.

    kind "kept": the prefixes are resident (anchored earlier), only
    the frame is computed. kind "filter": the filter chain kept the
    longest survivors; those pay the frame only, the rest scan.
    kind "none": every live anchor scans preamble + document + frame.
    """
    frame = _label_counts(join)[anchor][1]
    n = live[anchor]
    mean = stats[anchor].mean_doc_tokens
    if kind == "kept":
        return sol.ask(pre + mean, frame) * n
    if kind == "filter":
        k = keep_plan[anchor]
        kept_n = n * k["kept_doc_frac"]
        return (sol.ask(pre + k["kept_mean"], frame) * kept_n
                + sol.scan(pre + k["unkept_mean"], frame) * (n - kept_n))
    return sol.scan(pre + mean, frame) * n


def _stream_work(join, live: dict, stats: dict, anchor: str,
                 pre: int) -> Work:
    """Expected Work of the tuple suffixes: partner labels, partner
    documents, and the answer cue, each attending over the resident
    anchor prefix and frame."""
    labels = _label_counts(join)
    partners = [a for a in _join_aliases(join) if a != anchor]
    u = _question_tokens(join.predicate)
    for p in partners:
        u += stats[p].mean_doc_tokens + labels[p][0]
    tuples = _cross_tuples(join, live, anchor)
    ctx = pre + stats[anchor].mean_doc_tokens + labels[anchor][1]
    return Work(tokens=tuples * u,
                pairs=tuples * (u * ctx + sol.triangle(u)),
                kv_written=tuples * u,
                kv_read=live[anchor] * ctx)


def _stage_work(join, live: dict, stats: dict, anchor: str, pre: int,
                kind, keep_plan: dict) -> Work:
    """Expected Work of one join stage at the current live counts."""
    return (_anchor_work(join, live, stats, anchor, pre, kind,
                         keep_plan)
            + _stream_work(join, live, stats, anchor, pre))


def _residency(anchor: str, cached, keep_plan: dict) -> str:
    if anchor in cached:
        return "kept"
    if anchor in keep_plan:
        return "filter"
    return "none"


def _walk(seq, live0: dict, stats: dict, pre: int,
          keep_plan: dict | None = None, persist=None):
    """Cost one (join, anchor) sequence with residency-aware stages.

    persist: aliases whose KV survives an anchor switch. None means
    all of them (the runtime keeps anchors a later group re-uses).

    Returns:
        (work, records): one record per stage with expected_tuples,
        tokens, and the anchor residency its cost assumed.
    """
    keep_plan = keep_plan or {}
    live = dict(live0)
    cached = set()
    total = Work()
    records = []
    for j, anchor in seq:
        kind = _residency(anchor, cached, keep_plan)
        w = _stage_work(j, live, stats, anchor, pre, kind, keep_plan)
        records.append(dict(tuples=_cross_tuples(j, live, anchor),
                            tokens=w.tokens, resident=kind))
        total = total + w
        _thin(live, j, anchor)
        cached.add(anchor)
        if persist is not None:
            cached = {a for a in cached
                      if a == anchor or a in persist}
    return total, records


# ------------------------------ the joint (order x anchor) search

def _anchor_candidates(join, honor_forced: bool = True) -> list:
    """Return candidate anchor aliases for a join stage.

    exists/anti always anchor the outer table. A forced full anchor
    is honored. Otherwise any table of the join is a candidate.
    """
    if join.semantics != "full":
        return [join.anchor]
    if honor_forced and join.anchor is not None:
        return [join.anchor]
    return _join_aliases(join)


def _anchor_fits(join, anchor: str, stats: dict, pre: int,
                 chunk: int) -> bool:
    """Whether the worst-case tuple anchored here fits one chunk."""
    labels = _label_counts(join)
    need = (pre + stats[anchor].max_doc_tokens + labels[anchor][1]
            + sum(labels[p][0] + stats[p].max_doc_tokens
                  for p in _join_aliases(join) if p != anchor)
            + _question_tokens(join.predicate))
    return need <= chunk


def _feasible_anchors(join, honor_forced, stats, pre, chunk) -> list:
    """Anchor candidates whose worst-case tuple fits the chunk. When
    none fits, every candidate is returned so the refusal check can
    name the least-bad need."""
    cands = _anchor_candidates(join, honor_forced)
    fits = [a for a in cands
            if _anchor_fits(join, a, stats, pre, chunk)]
    return fits or cands


def plan_joins(joins, rule: str, stats: dict, live0: dict, pre: int,
               *, chunk: int, keep_plan: dict | None = None,
               rank=None):
    """Search stage order and anchor choice, residency-aware.

    'as_written' keeps the written stage order and searches anchors
    only. 'by_cost' runs the left deep subset DP over (joined
    aliases, cached prefix aliases).

    Args:
        keep_plan: alias -> kept-survivor split from plan_keeps.
        rank: Work -> comparable; the caller prices Work to seconds.

    Returns:
        ([(join, anchor)], remarks).
    """
    if not joins:
        return [], []
    keep_plan = keep_plan or {}
    rank = rank or (lambda w: w.tokens)

    def key(work):
        return (rank(work), work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    def best_fixed_order(order, honor_forced):
        best, best_key = None, None
        for assign in itertools.product(
                *[_feasible_anchors(j, honor_forced, stats, pre, chunk)
                  for j in order]):
            seq = list(zip(order, assign))
            work, _ = _walk(seq, live0, stats, pre, keep_plan)
            k = key(work)
            if best_key is None or k < best_key:
                best, best_key = seq, k
        return best, best_key

    def best_dp(honor_forced):
        aliases = []
        for j in joins:
            for a in _join_aliases(j):
                if a not in aliases:
                    aliases.append(a)
        edge_aliases = [frozenset(_join_aliases(j)) for j in joins]
        live_cache = {}

        def live_for(applied: frozenset):
            if applied not in live_cache:
                live = dict(live0)
                for i in sorted(applied):
                    _thin(live, joins[i], joins[i].anchor
                          or _join_aliases(joins[i])[0])
                live_cache[applied] = live
            return live_cache[applied]

        extension_cache = {}

        def extend(relations, cached, added):
            cache_key = (relations, cached, added)
            if cache_key in extension_cache:
                return extension_cache[cache_key]
            crossing = tuple(
                i for i, ends in enumerate(edge_aliases)
                if added in ends and ends & relations
                and ends <= relations | {added})
            if not crossing:
                extension_cache[cache_key] = ()
                return ()
            applied0 = frozenset(
                i for i, ends in enumerate(edge_aliases)
                if ends <= relations)
            extensions = []

            def visit(order, pos, applied, cached_now, work, steps):
                if pos == len(order):
                    extensions.append(Extension(
                        work=work, cached=frozenset(cached_now),
                        steps=tuple(steps)))
                    return
                i = order[pos]
                j = joins[i]
                live = live_for(applied)
                for anchor in _feasible_anchors(
                        j, honor_forced, stats, pre, chunk):
                    kind = _residency(anchor, cached_now, keep_plan)
                    w = _stage_work(j, live, stats, anchor, pre,
                                    kind, keep_plan)
                    step = dict(written_pos=i, anchor=anchor,
                                tuples=_cross_tuples(j, live, anchor),
                                tokens=w.tokens, resident=kind)
                    visit(order, pos + 1, applied | {i},
                          cached_now | {anchor}, work + w,
                          steps + [step])

            for order in itertools.permutations(crossing):
                visit(order, 0, applied0, set(cached), Work(), [])
            extension_cache[cache_key] = tuple(extensions)
            return extension_cache[cache_key]

        search = optimize_left_deep(aliases, (), Work(), extend)
        if not search.candidates:
            return None, None
        best = min(search.candidates,
                   key=lambda c: key(c.work) + (c.relation_order,))
        seq = [(joins[s["written_pos"]], s["anchor"])
               for s in best.steps]
        return seq, key(best.work)

    def best_seq(honor_forced):
        if rule == "as_written" or len(joins) == 1:
            return best_fixed_order(list(joins), honor_forced)
        seq, k = best_dp(honor_forced)
        if seq is None:
            # a join predicate with no alias in common with the rest:
            # no connected left deep order exists, so cost the written
            # order directly
            return best_fixed_order(list(joins), honor_forced)
        return seq, k

    seq, seq_key = best_seq(True)
    remarks = []
    forced = sorted({j.anchor for j in joins
                     if j.anchor is not None and j.semantics == "full"})
    if forced:
        _, free_key = best_seq(False)
        if free_key < seq_key:
            remarks.append(
                f"anchors {forced} were forced; a free choice prices "
                f"lower ({free_key[0]:.3f} vs {seq_key[0]:.3f} "
                f"predicted seconds)")
    return seq, remarks


# ----------------------------------------------- KV keep (residency)

def _page_round(tokens: float, page_tokens: int) -> float:
    return -(-tokens // page_tokens) * page_tokens


def _split_at(doc_tokens, threshold: int, survivor_frac: float,
              overhead: int, page_tokens: int) -> dict | None:
    """The keep record for one alias at a given length threshold:
    documents at or above it stay resident, the rest recompute."""
    lengths = [int(t) for t in doc_tokens]
    kept = [t for t in lengths if t >= max(1, threshold)]
    if not kept:
        return None
    unkept = [t for t in lengths if t < max(1, threshold)]
    return dict(
        min_doc_tokens=1 if not unkept else threshold,
        kept_doc_frac=len(kept) / len(lengths),
        kept_mean=sum(kept) / len(kept),
        unkept_mean=(sum(unkept) / len(unkept)) if unkept else 0.0,
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
    """Which survivors of one alias stay resident for a later join.

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

    Args:
        overhead: Tokens added to each kept document (engine preamble
            and the frame allowance), page-rounded with it.

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


def _keep_frame_tokens(joins, alias: str) -> int:
    """The widest frame any join could write after this alias's
    documents, the page allowance a kept survivor must reserve."""
    frames = [_label_counts(j)[alias][1] for j in joins
              if alias in _join_aliases(j)]
    return max(frames, default=0)


def plan_keeps(joins, filters, doc_tokens: dict, pre: int,
               budget_tokens: float, page_tokens: int) -> dict:
    """Candidate keep plan: every filtered alias a join could anchor,
    each split against the whole budget. plan_query trims the set to
    the aliases the chosen plan actually anchors and re-checks the
    joint capacity."""
    plan = {}
    for alias, preds in filters.items():
        frame = _keep_frame_tokens(joins, alias)
        if not any(alias in _join_aliases(j) for j in joins):
            continue
        frac = 1.0
        for p in preds:
            frac *= p.selectivity if p.selectivity is not None else 1.0
        split = keep_split(doc_tokens[alias], budget_tokens, frac,
                           pre + frame, page_tokens)
        if split is not None:
            split["frame_tokens"] = frame
            plan[alias] = split
    return plan


def _keep_timeline(seq, keep_plan, stats, doc_tokens,
                   live0, pre, page_tokens, workers: int):
    """Peak expected resident tokens per worker across the plan.

    A point per group boundary: the end of the filter round holds
    every kept alias's filter mass; after each group, kept filter
    masses not yet anchored plus the gate-survivor mass of anchors a
    later group re-uses (keep_anchor_kv holds every gate survivor,
    not just the filter's length-thresholded keep). Within a group,
    anchors run one at a time (run_join gates per anchor), so the
    per-anchor working set is the headroom the caller adds.
    """
    groups = _group_seq(seq)
    first_use, last_use, frames = {}, {}, {}
    for g, (anchor, members) in enumerate(groups):
        first_use.setdefault(anchor, g)
        last_use[anchor] = g
        frame = max(_label_counts(j)[anchor][1] for j in members)
        frames[anchor] = max(frames.get(anchor, 0), frame)

    def survivor_mass(alias, live_count):
        frac = live_count / max(1.0, float(stats[alias].n_docs))
        return frac * sum(
            _page_round(pre + t + frames[alias], page_tokens)
            for t in doc_tokens[alias])

    live = dict(live0)
    points = [sum(k["kept_expected_tokens"]
                  for k in keep_plan.values())]
    for g, (anchor, members) in enumerate(groups):
        for j in members:
            _thin(live, j, anchor)
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


def _group_seq(seq):
    """Group consecutive full stages on the same anchor, the same
    rule the node graph uses."""
    groups = []
    for j, anchor in seq:
        merge = (groups and j.semantics == "full"
                 and groups[-1][2] and groups[-1][0] == anchor)
        if merge:
            groups[-1][1].append(j)
        else:
            groups.append([anchor, [j], j.semantics == "full"])
    return [(a, m) for a, m, _ in groups]


# ------------------------------------------------------ runtime picks

def pick_runtime_anchor(spec: dict, live_doc_tokens: dict,
                        pre_len: int, chunk_budget: int,
                        resident_tokens: dict | None = None) -> str:
    """Choose the cheapest anchor at barrier time using live token counts.

    Args:
        spec: Payload stage spec (aliases, labels, frames, tail as
            token id lists).
        live_doc_tokens: alias -> list of per-document token counts.
        pre_len: Preamble length in tokens.
        chunk_budget: Maximum tokens per chunk.
        resident_tokens: alias -> prefix tokens already resident in
            the arena; those are not recomputed when anchored.

    Returns:
        The alias to anchor on. Only anchors whose worst-case tuple
        fits chunk_budget are candidates.
    """
    tail = len(spec["tail"])
    resident = resident_tokens or {}
    counts = {a: len(t) for a, t in live_doc_tokens.items()}
    means = {a: (sum(t) / len(t)) if t else 0.0
             for a, t in live_doc_tokens.items()}
    maxes = {a: max(t) if t else 0
             for a, t in live_doc_tokens.items()}
    aliases = spec["aliases"]

    def need(a):
        return (pre_len + maxes[a] + len(spec["frames"][a]) + tail
                + sum(len(spec["labels"][p]) + maxes[p]
                      for p in aliases if p != a))

    def total(a):
        tuples = 1.0
        for al in aliases:
            tuples *= counts[al]
        per_tuple = tail + sum(len(spec["labels"][p]) + means[p]
                               for p in aliases if p != a)
        fresh = max(0.0, counts[a] * (means[a] + pre_len)
                    - resident.get(a, 0.0))
        return (fresh + counts[a] * len(spec["frames"][a])
                + tuples * per_tuple)

    candidates = [a for a in aliases
                  if a == spec["anchor"] or need(a) <= chunk_budget]
    return min(candidates, key=total)


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

    # ---- the order rule first: the joint search below needs it
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)

    # ---- expected live counts after filters, and the fixed filter
    # work every candidate join plan shares
    live0 = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in fs:
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live0[_filter_alias(fs[0])] *= surv
    base_work = _filter_work(filters, stats, rule, pre)

    def rank(work):
        return sol.speed_of_light(base_work + work, model, device,
                                  chunk).seconds

    # ---- the largest single admission any operator makes: the keep
    # arithmetic reserves it as working headroom
    headroom = 0
    for s in scans:
        fq = max((_question_tokens(p.prompt)
                  for p in filters.get(s.alias, ())), default=0)
        if fq:
            headroom = max(headroom,
                           pre + stats[s.alias].max_doc_tokens + fq)
    for j in joins:
        labels = _label_counts(j)
        for a in _join_aliases(j):
            headroom = max(
                headroom,
                pre + stats[a].max_doc_tokens + labels[a][1]
                + sum(labels[p][0] + stats[p].max_doc_tokens
                      for p in _join_aliases(j) if p != a)
                + _question_tokens(j.predicate))

    # ---- keep candidates, then the joint (order x anchor) search
    keep_budget = max(0.0, float(admission - headroom)) * workers
    candidates = plan_keeps(joins, filters, doc_tokens, pre,
                            keep_budget, budgets.PAGE_TOKENS)

    def consumed_keeps(seq, records, plan):
        """The keeps the sequence anchors while their filter KV is
        still resident; the rest have no reader and are dropped."""
        used = {a for (j, a), r in zip(seq, records)
                if r["resident"] == "filter"}
        return {a: k for a, k in plan.items() if a in used}

    def trim(seq, plan):
        """Raise thresholds until the peak expected resident tokens
        per worker fit the arena beside the working headroom. The
        shortest kept documents go first, wherever they are."""
        plan = dict(plan)
        while plan:
            peak = _keep_timeline(seq, plan, stats, doc_tokens,
                                  live0, pre, budgets.PAGE_TOKENS,
                                  workers)
            if peak + headroom <= admission:
                break
            alias = min(
                plan,
                key=lambda a: min(t for t in doc_tokens[a]
                                  if t >= max(1, plan[a]
                                              ["min_doc_tokens"])))
            split = _raise_threshold(plan[alias], doc_tokens[alias],
                                     budgets.PAGE_TOKENS)
            if split is None:
                del plan[alias]
            else:
                split["frame_tokens"] = plan[alias]["frame_tokens"]
                plan[alias] = split
        return plan

    seq, search_remarks = plan_joins(
        joins, rule, stats, live0, pre, chunk=chunk,
        keep_plan=candidates, rank=rank)
    _, stage_records = _walk(seq, live0, stats, pre, candidates)
    kept0 = consumed_keeps(seq, stage_records, candidates)
    keep_plan = trim(seq, kept0)
    if keep_plan != kept0:
        # the credited residency shrank: search once more against
        # what the arena can actually hold
        seq, search_remarks = plan_joins(
            joins, rule, stats, live0, pre, chunk=chunk,
            keep_plan=keep_plan, rank=rank)
        _, stage_records = _walk(seq, live0, stats, pre, keep_plan)
        keep_plan = trim(seq, consumed_keeps(seq, stage_records,
                                             keep_plan))
        _, stage_records = _walk(seq, live0, stats, pre, keep_plan)
    remarks.extend(search_remarks)

    for alias, k in sorted(keep_plan.items()):
        what = ("all survivors" if k["min_doc_tokens"] <= 1 else
                f"survivors of {k['min_doc_tokens']}+ tokens")
        remarks.append(
            f"keep KV on {alias!r}: {what} stay resident for the "
            f"join, {k['kept_expected_tokens'] / max(1, workers):,.0f} "
            f"expected tokens per worker of the {admission:,}-token "
            f"arena")

    # ---- refusal checks on the chosen plan
    anchors = {id(j): a for j, a in seq}
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

    remarks.append("kv_dtype=bf16 (always)")

    # ---- build the dataflow graph; ids_src tracks each table's
    # current producer node
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
            keep = keep_plan.get(s.alias)
            # arena writes when a later stage reads the KV back or a
            # join keeps it
            writes = len(stages) > 1 or keep is not None
            fid = f"filter:{s.alias}"
            nodes.append(dict(
                id=fid, op="FilterChain",
                inputs=(ids_src[s.alias],),
                alias=s.alias, arena_writes=writes,
                keep_kv=keep is not None,
                keep_frame_tokens=(keep["frame_tokens"]
                                   if keep else 0),
                # the expected-capacity length threshold: the cost
                # model credited documents at or above it as kept.
                # The runtime keeps every survivor while pages last
                # and evicts under pressure; this records what the
                # plan priced
                keep_min_doc_tokens=(keep["min_doc_tokens"]
                                     if keep else 0),
                stages=tuple(stages)))
            ids_src[s.alias] = (fid, f"ids:{s.alias}")
            if not writes:
                remarks.append(
                    f"filter on {s.alias!r}: arena writes off (one "
                    f"stage - nothing reads the KV again)")

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    groups = []
    for (j, anchor), record in zip(seq, stage_records):
        merge = (groups and j.semantics == "full" and groups[-1]["full"]
                 and groups[-1]["anchor"] == anchor)
        if merge:
            groups[-1]["members"].append((j, record))
        else:
            groups.append(dict(anchor=anchor,
                               full=(j.semantics == "full"),
                               members=[(j, record)]))
    group_last_use = {}
    for g, group in enumerate(groups):
        group_last_use[group["anchor"]] = g

    written = {id(j): i for i, j in enumerate(joins)}
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
                for j, _ in later["members"]:
                    for a in _join_aliases(j):
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
        for j, record in group["members"]:
            partners = [a for a in _join_aliases(j) if a != anchor]
            stage_labels = _label_counts(j)
            stage_dicts.append(dict(
                written_pos=written[id(j)], exec_idx=exec_idx,
                anchor=anchor, partners=partners,
                semantics=j.semantics, selectivity=j.selectivity,
                expected_tuples=round(record["tuples"], 1),
                anchor_frame_tokens=stage_labels[anchor][1],
                pair_tail_tokens=_question_tokens(j.predicate),
                anchor_resident=record["resident"],
                tuple_tokens=round(record["tokens"], 1)))
            exec_idx += 1
            for a in partners:
                if a not in in_aliases:
                    in_aliases.append(a)
            if j.semantics == "full":
                pairs_edges.append((gid, f"pairs:{written[id(j)]}"))
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
