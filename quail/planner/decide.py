"""Planner decisions: filter order, join order, anchor choice, and
sharding from a logical plan and corpus token counts.
"""

import itertools

from quail.logical import (LogicalPlan, Project, Scan, SemanticFilter,
                           SemanticJoin)
from quail.planner import budgets
from quail.planner.plan import CorpusStats, PhysicalPlan, Refusal
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


def _anchor_prefix_tokens(join, live: dict, stats: dict, anchor: str,
                          pre_tokens: int = 0) -> float:
    """Total prefix tokens for the anchor: preamble + document + frame,
    summed over live anchor documents.
    """
    labels = _label_counts(join)
    return live[anchor] * (stats[anchor].mean_doc_tokens + pre_tokens
                           + labels[anchor][1])


def _frame_only_tokens(join, live: dict, anchor: str) -> float:
    """Frame-only tokens for a later stage continuing the same anchor."""
    labels = _label_counts(join)
    return live[anchor] * labels[anchor][1]


def _pair_tokens(join, live: dict, stats: dict, anchor: str) -> float:
    """Total tokens for partner documents and answer cues across all tuples."""
    labels = _label_counts(join)
    partners = [a for a in _join_aliases(join) if a != anchor]
    per_tuple = _question_tokens(join.predicate)
    for p in partners:
        per_tuple += stats[p].mean_doc_tokens + labels[p][0]
    return _cross_tuples(join, live, anchor) * per_tuple


def _stage_tokens(join, live: dict, stats: dict, anchor: str,
                  pre_tokens: int = 0) -> float:
    """Total tokens for one join stage at the current live counts."""
    return (_anchor_prefix_tokens(join, live, stats, anchor,
                                  pre_tokens)
            + _pair_tokens(join, live, stats, anchor))


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


# ------------------------------ the joint (order x anchor) search

# Token overhead per anchor-switch barrier. Not yet measured.
RESHARD_OVERHEAD_TOKENS = 0.0


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


def _walk(seq, live0: dict, stats: dict, pre_tokens: int = 0):
    """Cost one (join, anchor) sequence.

    Returns:
        (total_tokens, records) where each record is
        (expected_tuples, stage_tokens).
    """
    live = dict(live0)
    total = 0.0
    records = []
    prev_join, prev_anchor = None, None
    for j, anchor in seq:
        same_group = (prev_join is not None and anchor == prev_anchor
                      and j.semantics == "full"
                      and prev_join.semantics == "full")
        if same_group:
            tokens = _frame_only_tokens(j, live, anchor)
        else:
            tokens = _anchor_prefix_tokens(j, live, stats, anchor,
                                           pre_tokens)
            if prev_anchor is not None and anchor != prev_anchor:
                tokens += RESHARD_OVERHEAD_TOKENS
        tokens += _pair_tokens(j, live, stats, anchor)
        records.append((_cross_tuples(j, live, anchor), tokens))
        total += tokens
        _thin(live, j, anchor)
        prev_join, prev_anchor = j, anchor
    return total, records


def plan_joins(joins, rule: str, stats: dict, live0: dict,
               pre_tokens: int = 0):
    """Enumerate stage-order x anchor-choice combinations, return the
    cheapest as ([(join, anchor)], remarks).
    """
    if not joins:
        return [], []

    orders = [list(joins)]
    if rule != "as_written" and len(joins) > 1:
        orders = [list(p) for p in itertools.permutations(joins)]

    def best_seq(honor_forced):
        best, best_total = None, float("inf")
        for order in orders:
            for assign in itertools.product(
                    *[_anchor_candidates(j, honor_forced)
                      for j in order]):
                seq = list(zip(order, assign))
                total, _ = _walk(seq, live0, stats, pre_tokens)
                if total < best_total:
                    best, best_total = seq, total
        return best, best_total

    seq, total = best_seq(True)
    remarks = []
    forced = sorted({j.anchor for j in joins
                     if j.anchor is not None and j.semantics == "full"})
    if forced:
        _, free_total = best_seq(False)
        if free_total < total:
            remarks.append(
                f"anchors {forced} were forced; a free choice prices "
                f"lower ({free_total:,.0f} vs {total:,.0f} tuple "
                f"tokens)")
    return seq, remarks


def pick_runtime_anchor(spec: dict, live_doc_tokens: dict,
                        pre_len: int, chunk_budget: int) -> str:
    """Choose the cheapest anchor at barrier time using live token counts.

    Args:
        spec: Payload stage spec (aliases, labels, frames, tail as
            token id lists).
        live_doc_tokens: alias -> list of per-document token counts.
        pre_len: Preamble length in tokens.
        chunk_budget: Maximum tokens per chunk.

    Returns:
        The alias to anchor on. Only anchors whose worst-case tuple
        fits chunk_budget are candidates.
    """
    tail = len(spec["tail"])
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
        return (counts[a] * (means[a] + pre_len
                             + len(spec["frames"][a]))
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
    pre = _preamble_tokens(filters, joins)

    # ---- the order rule first: the joint search below needs it
    rule, source = (order, f"user: order={order!r}") if order else \
        default_order_rule(filters, joins)

    # ---- joint (order x anchor) search, seeded with post-filter live
    # estimates
    live0 = {a: float(st.n_docs) for a, st in stats.items()}
    for fs in filters.values():
        surv = 1.0
        for p in fs:
            surv *= p.selectivity if p.selectivity is not None else 1.0
        live0[_filter_alias(fs[0])] *= surv
    seq, search_remarks = plan_joins(joins, rule, stats, live0, pre)
    remarks.extend(search_remarks)
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

    admission = budgets.arena_tokens(model, device, chunk)
    _, stage_records = _walk(seq, live0, stats, pre)

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
            # arena writes only needed when a later stage will read
            # the KV back
            writes = len(stages) > 1
            fid = f"filter:{s.alias}"
            nodes.append(dict(id=fid, op="FilterChain",
                              inputs=(ids_src[s.alias],),
                              alias=s.alias, arena_writes=writes,
                              stages=tuple(stages)))
            ids_src[s.alias] = (fid, f"ids:{s.alias}")
            if not writes:
                remarks.append(
                    f"filter on {s.alias!r}: arena writes off (one "
                    f"stage - nothing reads the KV again)")

    # group consecutive full stages on the same anchor; gates run
    # alone; anchor switches become barriers
    groups = []
    for j, anchor in seq:
        merge = (groups and j.semantics == "full" and groups[-1]["full"]
                 and groups[-1]["anchor"] == anchor)
        if merge:
            groups[-1]["members"].append(j)
        else:
            groups.append(dict(anchor=anchor,
                               full=(j.semantics == "full"),
                               members=[j]))

    written = {id(j): i for i, j in enumerate(joins)}
    rec = iter(stage_records)
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
                for j in later["members"]:
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
        for j in group["members"]:
            tuples, tokens = next(rec)
            partners = [a for a in _join_aliases(j) if a != anchor]
            stage_labels = _label_counts(j)
            stage_dicts.append(dict(
                written_pos=written[id(j)], exec_idx=exec_idx,
                anchor=anchor, partners=partners,
                semantics=j.semantics, selectivity=j.selectivity,
                expected_tuples=round(tuples, 1),
                anchor_frame_tokens=stage_labels[anchor][1],
                pair_tail_tokens=_question_tokens(j.predicate),
                tuple_tokens=round(tokens, 1)))
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
