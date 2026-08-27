"""The shared join search: stage order and anchors from live counts,
live document lengths, and resident KV.

One algorithm with different inputs at plan time and at runtime. At
plan time the inputs are expected counts, the corpus length lists,
and the keep credit. At runtime the worker calls it after the filter
round and after every join group with the actual survivors and the
document KV still resident. The runtime executes the next group from
each answer, then searches again when new answers are available.

Everything here is counted: costs are Work records priced by
speed_of_light against the model architecture and the device
datasheet. No measured constant.
"""

import itertools
from dataclasses import dataclass

from quail.planner import sol
from quail.planner.leftdeep import Extension, optimize_left_deep
from quail.planner.sol import Work

DocumentKey = tuple[str, int]


@dataclass(frozen=True)
class KVState:
    """The small physical state needed for guaranteed group reuse."""

    pending_anchor: str | None = None
    group_open: bool = False
    at_start: bool = True


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


def _document_keys(resident: dict) -> frozenset[DocumentKey]:
    return frozenset(
        (alias, position)
        for alias, positions in resident.items()
        for position in positions
    )


def fit_resident_documents(resident: dict, lengths: dict, pre: int,
                           model, device, arena_tokens: float | None,
                           page_tokens: int = 16) -> dict:
    """Apply the runtime value-per-page eviction order to a snapshot."""

    keys = _document_keys(resident)
    if arena_tokens is None:
        kept = keys
    else:
        capacity_pages = max(0, int(arena_tokens) // page_tokens)
        entries = []
        total_pages = 0
        for key in keys:
            alias, position = key
            tokens = pre + lengths[alias][position]
            pages = -(-tokens // page_tokens)
            value = sol.prefix_recompute_seconds(tokens, model, device)
            entries.append((value / pages, key, pages))
            total_pages += pages
        kept = set(keys)
        for _, key, pages in sorted(entries):
            if total_pages <= capacity_pages:
                break
            kept.remove(key)
            total_pages -= pages
    return {
        alias: {position for key_alias, position in kept
                if key_alias == alias}
        for alias in lengths
    }


def residency(anchor: str, state: KVState,
              initially_resident: frozenset[DocumentKey],
              same_group: bool = False) -> str:
    """Name the source of the KV credit recorded for one stage."""
    if same_group:
        return "kept"
    if state.at_start and any(
            key_alias == anchor for key_alias, _ in initially_resident):
        return "filter"
    return "none"


def resident_positions(anchor: str, state: KVState,
                       initially_resident: frozenset[DocumentKey],
                       length_count: int,
                       same_group: bool = False) -> tuple[int, ...]:
    """Document positions guaranteed resident for the next stage."""

    if same_group:
        return tuple(range(length_count))
    if not state.at_start:
        return ()
    return tuple(sorted(
        position for key_alias, position in initially_resident
        if key_alias == anchor
    ))


def stage_work(spec: dict, anchor: str, live: dict, lengths: dict,
               resident_keys, pre: int, same_group: bool = False) -> Work:
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
    res = {position for alias, position in resident_keys
           if alias == anchor}
    total = Work()
    for i, doc in enumerate(rows):
        prefix = pre + doc
        start = (sol.ask(prefix, frame) if same_group or i in res
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
         model, device, *, arena_tokens: float | None = None,
         page_tokens: int = 16):
    """Cost one [(spec, anchor)] sequence.

    Returns (work, records): per stage, the written position, the
    anchor, the residency its cost assumed, and expected tuples and
    tokens.
    """
    live = dict(live0)
    initial = _document_keys(resident)
    state = KVState()
    total = Work()
    records = []
    for spec, anchor in seq:
        same_group = (state.group_open
                      and state.pending_anchor == anchor
                      and spec["semantics"] == "full")
        kind = residency(anchor, state, initial, same_group)
        positions = resident_positions(
            anchor, state, initial, len(lengths[anchor]), same_group)
        resident_keys = frozenset((anchor, i) for i in positions)
        w = stage_work(spec, anchor, live, lengths, resident_keys,
                       pre, same_group)
        records.append(dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            resident_docs=len(positions),
                            resident_positions=positions,
                            tuples=cross_tuples(spec, live),
                            tokens=w.tokens))
        total = total + w
        thin(live, spec)
        state = KVState(anchor, spec["semantics"] == "full", False)
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
        arena_tokens: Accepted for caller compatibility. The resident
            input already records the finite KV state at this planning
            point. The search does not predict later evictions.
        page_tokens: Accepted for caller compatibility.
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
        work, records = walk(
            seq, live, lengths, resident, pre, model, device,
            arena_tokens=arena_tokens, page_tokens=page_tokens)
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
    live_cache = {}
    initial_keys = _document_keys(resident)

    def live_for(applied: frozenset):
        if applied not in live_cache:
            state = dict(live)
            for i in sorted(applied):
                thin(state, specs[i])
            live_cache[applied] = state
        return dict(live_cache[applied])

    extension_cache = {}

    def extend(relations, kv_state, added):
        cache_key = (relations, kv_state, added)
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

        def visit(order, pos, applied, state_now, work, steps):
            if pos == len(order):
                extensions.append(Extension(
                    work=work, state_property=state_now,
                    steps=tuple(steps)))
                return
            i = order[pos]
            spec = specs[i]
            live_now = live_for(applied)
            for anchor in _feasible_anchors(
                    spec, honor_forced, lengths, pre, chunk_tokens):
                same_group = (
                    state_now.group_open
                    and state_now.pending_anchor == anchor
                    and spec["semantics"] == "full"
                )
                kind = residency(anchor, state_now, initial_keys,
                                 same_group)
                positions = resident_positions(
                    anchor, state_now, initial_keys,
                    len(lengths[anchor]), same_group)
                resident_keys = frozenset(
                    (anchor, position) for position in positions)
                w = stage_work(spec, anchor, live_now, lengths,
                               resident_keys, pre, same_group)
                step = dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            resident_docs=len(positions),
                            resident_positions=positions,
                            tuples=cross_tuples(spec, live_now),
                            tokens=w.tokens)
                next_state = KVState(
                    anchor, spec["semantics"] == "full", False)
                visit(order, pos + 1, applied | {i}, next_state,
                      work + w,
                      steps + [step])

        for order in itertools.permutations(crossing):
            visit(order, 0, applied0, kv_state, Work(), [])
        extension_cache[cache_key] = tuple(extensions)
        return extension_cache[cache_key]

    search = optimize_left_deep(
        aliases, KVState(), Work(), extend)
    if not search.candidates:
        return None
    best = min(search.candidates,
               key=lambda c: rank(c.work) + (c.relation_order,))
    return dict(seq=[(s["written_pos"], s["anchor"])
                     for s in best.steps],
                records=[dict(s) for s in best.steps],
                work=best.work, states=search.state_count,
                generated=search.generated_count)
