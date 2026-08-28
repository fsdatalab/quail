"""The shared join search: stage order and anchors from live counts,
live document lengths, and resident KV.

One algorithm with different inputs at plan time and at runtime. At
plan time the inputs are expected counts, corpus length summaries,
and the keep credit. At runtime the worker calls it after the filter
round and after every join group with the actual survivors and the
document KV still resident. The runtime executes the next group from
each answer, then searches again when new answers are available.

Everything here is counted: costs are Work records priced by
speed_of_light against the model architecture and the device
datasheet. No measured constant.
"""

import itertools
from collections import Counter
from dataclasses import dataclass

from quail.planner.sol import prefix_recompute_seconds, speed_of_light
from quail.planner.work import Work, triangle

DocumentKey = tuple[str, int]


@dataclass(frozen=True)
class KVState:
    """The small physical state needed for guaranteed group reuse."""

    pending_anchor: str | None = None
    group_open: bool = False
    at_start: bool = True


@dataclass(frozen=True)
class AliasStats:
    """Document length sums used to cost one alias in constant time."""

    count: int
    total: int
    squared: int
    maximum: int
    histogram: tuple[tuple[int, int], ...] = ()
    resident_count: int = 0
    resident_total: int = 0
    resident_squared: int = 0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def with_resident_min(self, minimum: int) -> "AliasStats":
        """Credit documents at or above one length threshold."""

        kept = [(length, count) for length, count in self.histogram
                if length >= minimum]
        return AliasStats(
            count=self.count,
            total=self.total,
            squared=self.squared,
            maximum=self.maximum,
            histogram=self.histogram,
            resident_count=sum(count for _, count in kept),
            resident_total=sum(length * count for length, count in kept),
            resident_squared=sum(
                length * length * count for length, count in kept),
        )


def summarize_alias(lengths, resident_positions=(), *,
                    resident_flags=None) -> AliasStats:
    """Summarize lengths and a resident position subset in one pass."""

    resident = set(resident_positions) if resident_flags is None else None
    count = total = squared = maximum = 0
    resident_count = resident_total = resident_squared = 0
    histogram = Counter()
    rows = (enumerate(lengths) if resident_flags is None else
            enumerate(zip(lengths, resident_flags, strict=True)))
    for position, raw in rows:
        if resident_flags is None:
            is_resident = position in resident
        else:
            raw, is_resident = raw
        length = int(raw)
        count += 1
        total += length
        squared += length * length
        maximum = max(maximum, length)
        histogram[length] += 1
        if is_resident:
            resident_count += 1
            resident_total += length
            resident_squared += length * length
    return AliasStats(
        count=count,
        total=total,
        squared=squared,
        maximum=maximum,
        histogram=tuple(sorted(histogram.items())),
        resident_count=resident_count,
        resident_total=resident_total,
        resident_squared=resident_squared,
    )


def _alias_stats(lengths: dict, resident: dict) -> dict[str, AliasStats]:
    """Normalize raw length lists or accept summaries from a caller."""

    out = {}
    for alias, values in lengths.items():
        if isinstance(values, AliasStats):
            if resident.get(alias):
                raise ValueError(
                    "resident positions cannot be added to AliasStats")
            out[alias] = values
        else:
            out[alias] = summarize_alias(
                values, resident.get(alias, ()))
    return out


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
    def count(alias):
        value = lengths[alias]
        return value.count if isinstance(value, AliasStats) else len(value)

    def maximum(alias):
        value = lengths[alias]
        if isinstance(value, AliasStats):
            return value.maximum
        return max(value) if value else 0

    if any(count(a) == 0 for a in spec["aliases"]):
        return True
    need = (pre + maximum(anchor) + spec["frame_tokens"][anchor]
            + spec["tail_tokens"]
            + sum(spec["label_tokens"][p] + maximum(p)
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
            value = prefix_recompute_seconds(tokens, model, device)
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


def residency(anchor: str, state: KVState, lengths: dict,
              same_group: bool = False) -> str:
    """Name the source of the KV credit recorded for one stage."""
    if same_group:
        return "kept"
    if state.at_start and lengths[anchor].resident_count:
        return "filter"
    return "none"


def resident_count(anchor: str, state: KVState, lengths: dict,
                   same_group: bool = False) -> int:
    """Number of documents credited as resident for one stage."""
    if same_group:
        return lengths[anchor].count
    if not state.at_start:
        return 0
    return lengths[anchor].resident_count


def stage_work(spec: dict, anchor: str, live: dict, lengths: dict,
               pre: int, *, resident_at_start: bool = False,
               same_group: bool = False) -> Work:
    """Expected Work of one stage at the current live counts.

    The length sums make the calculation constant time in the number
    of documents. A resident prefix pays its frame only. A missing
    prefix scans the preamble, document, and frame. Every tuple then
    carries partner labels, partner documents, and the answer cue.
    """
    stats = lengths[anchor]
    n = live[anchor]
    tuples = cross_tuples(spec, live)
    if stats.count == 0 or n <= 0:
        return Work()
    partners = [a for a in spec["aliases"] if a != anchor]
    u = spec["tail_tokens"] + sum(
        spec["label_tokens"][p] + lengths[p].mean for p in partners)
    frame = spec["frame_tokens"][anchor]
    per_anchor = tuples / n
    frac = n / stats.count

    count = stats.count
    prefix_sum = stats.total + pre * count
    prefix_squared = (
        stats.squared + 2 * pre * stats.total + pre * pre * count)
    if same_group:
        resident_n = count
        resident_prefix = prefix_sum
        resident_prefix_squared = prefix_squared
    elif resident_at_start:
        resident_n = stats.resident_count
        resident_prefix = stats.resident_total + pre * resident_n
        resident_prefix_squared = (
            stats.resident_squared
            + 2 * pre * stats.resident_total
            + pre * pre * resident_n)
    else:
        resident_n = 0
        resident_prefix = 0
        resident_prefix_squared = 0

    resident_start = Work(
        tokens=resident_n * frame,
        pairs=frame * resident_prefix + resident_n * triangle(frame),
        kv_written=resident_n * frame,
        kv_read=resident_prefix,
    )
    missing_n = count - resident_n
    missing_prefix = prefix_sum - resident_prefix
    missing_prefix_squared = prefix_squared - resident_prefix_squared
    scan_sum = missing_prefix + missing_n * frame
    scan_squared = (
        missing_prefix_squared
        + 2 * frame * missing_prefix
        + missing_n * frame * frame)
    missing_start = Work(
        tokens=scan_sum,
        pairs=(scan_squared + scan_sum) / 2,
        kv_written=scan_sum,
    )
    stream = Work(
        tokens=count * per_anchor * u,
        pairs=per_anchor * (
            u * (prefix_sum + count * frame)
            + count * triangle(u)),
        kv_written=count * per_anchor * u,
        kv_read=prefix_sum + count * frame,
    )
    return (resident_start + missing_start + stream) * frac


def walk(seq, live0: dict, lengths: dict, resident: dict, pre: int,
         model, device, *, arena_tokens: float | None = None,
         page_tokens: int = 16):
    """Cost one [(spec, anchor)] sequence.

    Returns (work, records): per stage, the written position, the
    anchor, the residency its cost assumed, and expected tuples and
    tokens.
    """
    live = dict(live0)
    lengths = _alias_stats(lengths, resident)
    state = KVState()
    total = Work()
    records = []
    for spec, anchor in seq:
        same_group = (state.group_open
                      and state.pending_anchor == anchor
                      and spec["semantics"] == "full")
        kind = residency(anchor, state, lengths, same_group)
        kept = resident_count(anchor, state, lengths, same_group)
        w = stage_work(
            spec, anchor, live, lengths, pre,
            resident_at_start=state.at_start, same_group=same_group)
        records.append(dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            resident_docs=kept,
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
                 page_tokens: int = 16,
                 already_joined=()):
    """Search stage order and anchor choice; return the cheapest.

    Args:
        specs: One dict per join in written order: aliases
            (placeholder order), anchor, anchor_free, semantics,
            selectivity, written_pos, frame_tokens and label_tokens
            per alias, tail_tokens.
        live: alias -> live document count (float; expected at plan
            time, exact after the filter round).
        lengths: alias -> live documents' token lengths or AliasStats.
        resident: alias -> positions into lengths[alias] whose prefix
            KV is resident.
        already_joined: aliases connected by completed join stages.
            Runtime replanning starts from this set instead of losing
            the connectivity established by earlier groups.
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

    lengths = _alias_stats(lengths, resident)
    resident = {}

    def rank(work):
        seconds = speed_of_light(base_work + work, model, device,
                                 chunk_tokens).seconds
        return (seconds, work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    def run_walk(order_specs, assign):
        seq = list(zip(order_specs, assign))
        work, records = walk(
            seq, live, lengths, resident, pre, model, device,
            arena_tokens=arena_tokens, page_tokens=page_tokens)
        return work, records, seq

    if fixed_order or (len(specs) == 1 and not already_joined):
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

    def live_for(applied: frozenset):
        if applied not in live_cache:
            state = dict(live)
            for i in sorted(applied):
                thin(state, specs[i])
            live_cache[applied] = state
        return dict(live_cache[applied])

    @dataclass(frozen=True)
    class Candidate:
        work: Work
        state: KVState
        relations: frozenset[str]
        applied: frozenset[int]
        relation_order: tuple[str, ...]
        steps: tuple[dict, ...] = ()

    def insert(frontier, candidate):
        if any(old.work.dominates(candidate.work) for old in frontier):
            return False
        frontier[:] = [old for old in frontier
                       if not candidate.work.dominates(old.work)]
        frontier.append(candidate)
        return True

    seed = frozenset(already_joined)
    unknown = seed - set(aliases)
    if unknown:
        aliases.extend(sorted(unknown))
    initial_relations = ([seed] if seed else
                         [frozenset((alias,)) for alias in aliases])
    states = {}
    generated = 0
    for relations in initial_relations:
        candidate = Candidate(
            Work(), KVState(), relations, frozenset(),
            tuple(sorted(relations)))
        states[(relations, frozenset(), KVState())] = [candidate]
        generated += 1

    for applied_count in range(len(specs)):
        current = [
            (key, tuple(frontier))
            for key, frontier in states.items()
            if len(key[1]) == applied_count
        ]
        for (relations, applied, state_now), frontier in current:
            available = [
                i for i, ends in enumerate(edge_aliases)
                if i not in applied
                and (ends <= relations
                     or (ends & relations
                         and len(ends - relations) == 1))
            ]
            for i in available:
                spec = specs[i]
                live_now = live_for(applied)
                added = tuple(sorted(edge_aliases[i] - relations))
                next_relations = relations | edge_aliases[i]
                next_applied = applied | {i}
                for anchor in _feasible_anchors(
                        spec, honor_forced, lengths, pre, chunk_tokens):
                    same_group = (
                        state_now.group_open
                        and state_now.pending_anchor == anchor
                        and spec["semantics"] == "full")
                    kind = residency(anchor, state_now, lengths,
                                     same_group)
                    kept = resident_count(anchor, state_now, lengths,
                                          same_group)
                    work = stage_work(
                        spec, anchor, live_now, lengths, pre,
                        resident_at_start=state_now.at_start,
                        same_group=same_group)
                    step = dict(
                        written_pos=spec["written_pos"], anchor=anchor,
                        resident=kind, resident_docs=kept,
                        tuples=cross_tuples(spec, live_now),
                        tokens=work.tokens)
                    next_state = KVState(
                        anchor, spec["semantics"] == "full", False)
                    for old in frontier:
                        generated += 1
                        candidate = Candidate(
                            old.work + work, next_state,
                            next_relations, next_applied,
                            old.relation_order + added,
                            old.steps + (step,))
                        target = states.setdefault(
                            (next_relations, next_applied, next_state), [])
                        insert(target, candidate)

    finals = [
        candidate
        for (relations, applied, _), frontier in states.items()
        if len(applied) == len(specs)
        and set(aliases) <= relations
        for candidate in frontier
    ]
    if not finals:
        return None
    best = min(
        finals,
        key=lambda c: rank(c.work) + (
            c.relation_order,
            tuple((s["written_pos"], s["anchor"]) for s in c.steps)))
    return dict(seq=[(s["written_pos"], s["anchor"])
                     for s in best.steps],
                records=[dict(s) for s in best.steps],
                work=best.work, states=len(states), generated=generated)
