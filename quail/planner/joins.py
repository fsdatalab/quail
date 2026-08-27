"""The shared join search: stage order and anchors from live counts,
live document lengths, and resident KV.

One algorithm, called twice per query with different inputs. At plan
time the inputs are expectations - selectivity-thinned counts, the
corpus length lists, the keep credit - and the output is the
predicted plan (explain, refusals, the SoL comparison). After the
filter round the worker calls it again with the actual survivors and
the KV actually resident, and executes its answer. Barriers reuse
that answer; no new order decision exists below a stage boundary.

Everything here is counted: costs are Work records priced by
speed_of_light against the model architecture and the device
datasheet. No measured constant.
"""

import itertools

from quail.planner import sol
from quail.planner.leftdeep import Extension, optimize_left_deep
from quail.planner.sol import Work


def _mean(lengths) -> float:
    return sum(lengths) / len(lengths) if lengths else 0.0


def surviving_docs(n_docs: float, n_partners: float,
                   tuple_selectivity) -> float:
    """Expected distinct documents with at least one matching tuple."""
    if tuple_selectivity is None:
        return n_docs
    return n_docs * (1.0 - (1.0 - tuple_selectivity)
                     ** max(1.0, n_partners))


def thin(live: dict, spec: dict) -> None:
    """Update live document counts after one join's selectivity."""
    sel = spec["selectivity"]
    aliases = spec["aliases"]

    def others(x):
        out = 1.0
        for a in aliases:
            if a != x:
                out *= live[a]
        return out

    if spec["semantics"] == "full":
        new = {a: surviving_docs(live[a], others(a), sel)
               for a in aliases}
        live.update(new)
    elif sel is not None:
        anchor = spec["anchor"]
        matched = surviving_docs(live[anchor], others(anchor), sel)
        live[anchor] = (matched if spec["semantics"] == "exists"
                        else live[anchor] - matched)


def cross_tuples(spec: dict, live: dict) -> float:
    tuples = 1.0
    for a in spec["aliases"]:
        tuples *= live[a]
    return tuples


def anchor_candidates(spec: dict, honor_forced: bool = True) -> list:
    """Candidate anchors: gates keep their outer table, a forced full
    anchor is honored, a free full join offers every table."""
    if spec["semantics"] != "full":
        return [spec["anchor"]]
    if honor_forced and not spec.get("anchor_free"):
        return [spec["anchor"]]
    return list(spec["aliases"])


def tuple_need(spec: dict, anchor: str, lengths: dict,
               pre: int) -> float:
    """Tokens of the worst live tuple anchored here: the working set
    one admission pins beside whatever the arena is holding. Zero
    when any side has no live documents."""
    if any(not lengths[a] for a in spec["aliases"]):
        return 0.0
    return (pre + max(lengths[anchor]) + spec["frame_tokens"][anchor]
            + spec["tail_tokens"]
            + sum(spec["label_tokens"][p] + max(lengths[p])
                  for p in spec["aliases"] if p != anchor))


def anchor_fits(spec: dict, anchor: str, lengths: dict, pre: int,
                chunk_tokens: int) -> bool:
    """Whether the worst live tuple anchored here fits one chunk."""
    return tuple_need(spec, anchor, lengths, pre) <= chunk_tokens


def _feasible_anchors(spec, honor_forced, lengths, pre, chunk) -> list:
    """Fitting candidates; every candidate when none fits, so the
    caller's refusal check can name the least-bad need."""
    cands = anchor_candidates(spec, honor_forced)
    fits = [a for a in cands
            if anchor_fits(spec, a, lengths, pre, chunk)]
    return fits or cands


def hold_tokens(live: dict, lengths: dict, resident: dict, cached,
                pre: int, page_tokens: int) -> float:
    """Expected resident tokens the priced plan holds at this stage.

    Earlier-anchored aliases count their full live sets; a
    filter-retained alias not yet anchored counts its resident
    positions. Live fractions scale the page-rounded per-document
    masses. Callers pass only aliases some remaining stage can still
    anchor - retained KV with no later reader is freed, not held -
    and add the current stage's tuple_need for the working set.
    """
    total = 0.0
    for a in set(cached) | set(resident):
        rows = lengths.get(a) or ()
        if not rows:
            continue
        frac = live[a] / len(rows)
        if a in cached:
            total += frac * sum(
                -(-(pre + t) // page_tokens) * page_tokens
                for t in rows)
        elif resident[a]:
            total += frac * sum(
                -(-(pre + rows[i]) // page_tokens) * page_tokens
                for i in resident[a])
    return total


# Marker stored in the search's cached-alias state set: some earlier
# stage's assumed-resident mass exceeded the arena. It rides the
# state (not a flag) so extension caching and the Pareto frontier
# keep overflowed and non-overflowed paths apart.
OVERFLOWED = "!hold-overflow"


def residency(anchor: str, cached, resident: dict) -> str:
    """What the anchor's cost assumed: 'kept' (anchored earlier, all
    live prefixes resident), 'filter' (some prefixes resident from
    the filter round), or 'none'."""
    if anchor in cached:
        return "kept"
    if resident.get(anchor):
        return "filter"
    return "none"


def stage_work(spec: dict, anchor: str, live: dict, lengths: dict,
               resident: dict, pre: int, cached) -> Work:
    """Expected Work of one stage at the current live counts.

    Anchor prefixes are priced per document over the live length
    list, scaled by the live fraction: a resident prefix (kept from
    an earlier anchoring, or retained by the filter round) pays its
    frame only, the rest scan preamble + document + frame. Every
    tuple then carries partner labels, partner documents, and the
    answer cue over the resident anchor context.
    """
    rows = lengths[anchor]
    n = live[anchor]
    tuples = cross_tuples(spec, live)
    if not rows or n <= 0:
        return Work()
    partners = [a for a in spec["aliases"] if a != anchor]
    u = spec["tail_tokens"] + sum(
        spec["label_tokens"][p] + _mean(lengths[p]) for p in partners)
    frame = spec["frame_tokens"][anchor]
    per_anchor = tuples / n
    frac = n / len(rows)
    res = resident.get(anchor, ())
    hit_all = anchor in cached
    total = Work()
    for i, doc in enumerate(rows):
        prefix = pre + doc
        start = (sol.ask(prefix, frame) if hit_all or i in res
                 else sol.scan(prefix, frame))
        # the expected form of sol.stream: per_anchor suffixes of
        # mean length u, the prefix read back once
        pairs = Work(tokens=per_anchor * u,
                     pairs=per_anchor * (u * (prefix + frame)
                                         + sol.triangle(u)),
                     kv_written=per_anchor * u,
                     kv_read=prefix + frame)
        total = total + (start + pairs) * frac
    return total


def walk(seq, live0: dict, lengths: dict, resident: dict, pre: int,
         arena_tokens: float | None = None, page_tokens: int = 16):
    """Cost one [(spec, anchor)] sequence.

    Residency credit needs the arena to have held everything the
    plan relies on: retained aliases a remaining stage still
    anchors (KV with no later reader is freed, not held), beside
    the largest tuple the current stage admits. From the first
    stage where that exceeds arena_tokens onward, every prefix
    prices as a scan: something was evicted, the search does not
    model what, so it stops assuming. None disables the check.

    Returns (work, records): per stage, the written position, the
    anchor, the residency its cost assumed, and expected tuples and
    tokens.
    """
    live = dict(live0)
    cached = set()
    overflowed = False
    total = Work()
    records = []
    for k, (spec, anchor) in enumerate(seq):
        if not overflowed and arena_tokens is not None:
            ahead = {a for _, a in seq[k:]}
            hold = hold_tokens(
                live, lengths,
                {a: p for a, p in resident.items() if a in ahead},
                cached & ahead, pre, page_tokens)
            overflowed = hold + tuple_need(
                spec, anchor, lengths, pre) > arena_tokens
        use_resident = {} if overflowed else resident
        use_cached = set() if overflowed else cached
        kind = residency(anchor, use_cached, use_resident)
        w = stage_work(spec, anchor, live, lengths, use_resident,
                       pre, use_cached)
        records.append(dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            tuples=cross_tuples(spec, live),
                            tokens=w.tokens))
        total = total + w
        thin(live, spec)
        cached.add(anchor)
    return total, records


def search_joins(specs, live: dict, lengths: dict, resident: dict,
                 pre: int, chunk_tokens: int, model, device, *,
                 base_work: Work = Work(), fixed_order: bool = False,
                 honor_forced: bool = True,
                 arena_tokens: float | None = None,
                 page_tokens: int = 16):
    """Search stage order and anchor choice; return the cheapest.

    Args:
        specs: One dict per join in written order: aliases
            (placeholder order), anchor, anchor_free, semantics,
            selectivity, written_pos, frame_tokens and label_tokens
            per alias, tail_tokens.
        live: alias -> live document count (float; expected at plan
            time, exact after the filter round).
        lengths: alias -> live documents' token lengths.
        resident: alias -> positions into lengths[alias] whose prefix
            KV is resident.
        base_work: Work outside the joins (the filter round), so
            candidates rank by whole-query predicted seconds.
        fixed_order: keep the written stage order (order=as_written);
            anchors are still chosen.
        honor_forced: honor forced full-join anchors.
        arena_tokens: KV capacity in tokens for the residency
            check. Credit is granted only while what a candidate
            plan holds - retained aliases some remaining stage can
            still anchor, beside the stage's largest tuple - fits
            this budget; past a plan's first stage over it, every
            prefix prices as a scan. None disables the check.
        page_tokens: arena page size, for rounding held masses.

    Returns:
        None when the join graph has no connected left deep order,
        else dict(seq=[(written_pos, anchor)], records, work, states,
        generated).
    """
    if not specs:
        return dict(seq=[], records=[], work=Work(), states=0,
                    generated=0)

    def rank(work):
        seconds = sol.speed_of_light(base_work + work, model, device,
                                     chunk_tokens).seconds
        return (seconds, work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    def run_walk(order_specs, assign):
        seq = list(zip(order_specs, assign))
        work, records = walk(seq, live, lengths, resident, pre,
                             arena_tokens, page_tokens)
        return work, records, seq

    if fixed_order or len(specs) == 1:
        best = None
        for assign in itertools.product(
                *[_feasible_anchors(s, honor_forced, lengths, pre,
                                    chunk_tokens) for s in specs]):
            work, records, seq = run_walk(list(specs), assign)
            key = rank(work)
            if best is None or key < best[0]:
                best = (key, work, records, seq)
        _, work, records, seq = best
        return dict(seq=[(s["written_pos"], a) for s, a in seq],
                    records=records, work=work, states=0, generated=0)

    aliases = []
    for s in specs:
        for a in s["aliases"]:
            if a not in aliases:
                aliases.append(a)
    edge_aliases = [frozenset(s["aliases"]) for s in specs]
    by_pos = {s["written_pos"]: s for s in specs}
    live_cache = {}

    def live_for(applied: frozenset):
        if applied not in live_cache:
            state = dict(live)
            for i in sorted(applied):
                thin(state, specs[i])
            live_cache[applied] = state
        return dict(live_cache[applied])

    claimable_cache = {}

    def claimable_for(applied: frozenset):
        """Aliases an unapplied stage could still anchor - the only
        ones whose retained KV a candidate plan keeps holding."""
        if applied not in claimable_cache:
            out = set()
            for j, s in enumerate(specs):
                if j not in applied:
                    out.update(anchor_candidates(s, honor_forced))
            claimable_cache[applied] = frozenset(out)
        return claimable_cache[applied]

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
            spec = specs[i]
            state = live_for(applied)
            for anchor in _feasible_anchors(
                    spec, honor_forced, lengths, pre, chunk_tokens):
                denied = OVERFLOWED in cached_now
                if not denied and arena_tokens is not None:
                    ahead = claimable_for(applied)
                    hold = hold_tokens(
                        state, lengths,
                        {a: p for a, p in resident.items()
                         if a in ahead},
                        cached_now & ahead, pre, page_tokens)
                    denied = hold + tuple_need(
                        spec, anchor, lengths, pre) > arena_tokens
                use_resident = {} if denied else resident
                use_cached = set() if denied else cached_now
                kind = residency(anchor, use_cached, use_resident)
                w = stage_work(spec, anchor, state, lengths,
                               use_resident, pre, use_cached)
                step = dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            tuples=cross_tuples(spec, state),
                            tokens=w.tokens)
                visit(order, pos + 1, applied | {i},
                      {OVERFLOWED} if denied
                      else cached_now | {anchor}, work + w,
                      steps + [step])

        for order in itertools.permutations(crossing):
            visit(order, 0, applied0, set(cached), Work(), [])
        extension_cache[cache_key] = tuple(extensions)
        return extension_cache[cache_key]

    search = optimize_left_deep(aliases, frozenset(), Work(), extend)
    if not search.candidates:
        return None
    best = min(search.candidates,
               key=lambda c: rank(c.work) + (c.relation_order,))
    return dict(seq=[(s["written_pos"], s["anchor"])
                     for s in best.steps],
                records=[dict(s) for s in best.steps],
                work=best.work, states=search.state_count,
                generated=search.generated_count)
