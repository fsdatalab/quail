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


def anchor_fits(spec: dict, anchor: str, lengths: dict, pre: int,
                chunk_tokens: int) -> bool:
    """Whether the worst live tuple anchored here fits one chunk."""
    if any(not lengths[a] for a in spec["aliases"]):
        return True
    need = (pre + max(lengths[anchor]) + spec["frame_tokens"][anchor]
            + spec["tail_tokens"]
            + sum(spec["label_tokens"][p] + max(lengths[p])
                  for p in spec["aliases"] if p != anchor))
    return need <= chunk_tokens


def _feasible_anchors(spec, honor_forced, lengths, pre, chunk) -> list:
    """Fitting candidates; every candidate when none fits, so the
    caller's refusal check can name the least-bad need."""
    cands = anchor_candidates(spec, honor_forced)
    fits = [a for a in cands
            if anchor_fits(spec, a, lengths, pre, chunk)]
    return fits or cands


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


def walk(seq, live0: dict, lengths: dict, resident: dict, pre: int):
    """Cost one [(spec, anchor)] sequence.

    Returns (work, records): per stage, the written position, the
    anchor, the residency its cost assumed, and expected tuples and
    tokens.
    """
    live = dict(live0)
    cached = set()
    total = Work()
    records = []
    for spec, anchor in seq:
        kind = residency(anchor, cached, resident)
        w = stage_work(spec, anchor, live, lengths, resident, pre,
                       cached)
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
                 honor_forced: bool = True):
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
        work, records = walk(seq, live, lengths, resident, pre)
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
                kind = residency(anchor, cached_now, resident)
                w = stage_work(spec, anchor, state, lengths,
                               resident, pre, cached_now)
                step = dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            tuples=cross_tuples(spec, state),
                            tokens=w.tokens)
                visit(order, pos + 1, applied | {i},
                      cached_now | {anchor}, work + w,
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
