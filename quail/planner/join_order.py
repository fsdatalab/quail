"""Choose join order and anchors from estimated survivors.

KV reuse is priced as unlimited, the speed-of-light assumption: a
document prefix computed once, by a filter or by an earlier anchor
use, is resident at every later anchor use. The executor keeps as much
of that KV as the arena holds and recomputes the rest.
"""

import itertools
from dataclasses import dataclass

from quail.cost.joins import stage_work as join_work
from quail.cost.sol import speed_of_light
from quail.cost.work import Work
from quail.planner.statistics import AliasStats, alias_stats, cross_tuples, thin
from quail.planner.validation import anchor_candidates, join_input_tokens


@dataclass(frozen=True)
class KVState:
    """The small physical state needed for guaranteed group reuse."""

    pending_anchor: str | None = None
    group_open: bool = False
    used_anchors: frozenset[str] = frozenset()


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
    need = join_input_tokens(
        spec, anchor, {alias: maximum(alias) for alias in spec["aliases"]}, pre)
    return need <= chunk_tokens


def _feasible_anchors(spec, honor_forced, lengths, pre, chunk) -> list:
    """Fitting candidates, or every candidate when none fits.

    Returning them all lets the caller's refusal check name the
    least-bad need.
    """
    cands = anchor_candidates(spec, honor_forced)
    fits = [a for a in cands
            if anchor_fits(spec, a, lengths, pre, chunk)]
    return fits or cands


def residency(anchor: str, state: KVState, resident_aliases,
              same_group: bool = False) -> str:
    """Name the source of the KV credit recorded for one stage.

    A prefix computed once is resident at every later anchor use: by
    its filter ("filter") or by an earlier anchor use ("kept").
    """
    if same_group or anchor in state.used_anchors:
        return "kept"
    if anchor in resident_aliases:
        return "filter"
    return "none"


def stage_work(spec, anchor, live, lengths, pre, *, resident=False, window=0):
    """Evaluate a prepared join's work at the current survivor counts."""
    if "cost" in spec:
        return spec["cost"].work(live)
    return join_work(spec, anchor, live[anchor], cross_tuples(spec, live),
                     lengths, pre, resident=resident, window=window)


def walk(seq, live0: dict, lengths: dict, resident, pre: int,
         model, device):
    """Cost one [(spec, anchor)] sequence.

    resident names the aliases whose prefix KV a filter computes
    before the joins. Returns (work, records): per stage, the written
    position, the anchor, the residency its cost assumed, and expected
    tuples and tokens.
    """
    lengths = alias_stats(lengths, model.sliding_window)
    resident = set(resident or ())
    state = KVState()
    total = Work()
    records = []
    applied = []
    for spec, anchor in seq:
        # thin in written order, as the search priced it
        live = dict(live0)
        for done in sorted(applied, key=lambda s: s["written_pos"]):
            thin(live, done)
        same_group = (state.group_open
                      and state.pending_anchor == anchor
                      and spec["semantics"] == "full")
        kind = residency(anchor, state, resident, same_group)
        kept = lengths[anchor].count if kind != "none" else 0
        w = stage_work(spec, anchor, live, lengths, pre,
                       resident=kind != "none", window=model.sliding_window)
        records.append(dict(written_pos=spec["written_pos"],
                            anchor=anchor, resident=kind,
                            resident_docs=kept,
                            tuples=cross_tuples(spec, live),
                            tokens=w.tokens, work=w))
        total = total + w
        applied.append(spec)
        state = KVState(anchor, spec["semantics"] == "full",
                        state.used_anchors | {anchor})
    return total, records


def search_joins(specs, live: dict, lengths: dict, resident,
                 pre: int, chunk_tokens: int, model, device, *,
                 base_work: Work = Work(), fixed_order: bool = False,
                 honor_forced: bool = True,
                 already_joined=()):
    """Search stage order and anchor choice; return the cheapest.

    Args:
        specs: One dict per join in written order: aliases
            (placeholder order), anchor, anchor_free, semantics,
            selectivity, written_pos, frame_tokens and label_tokens
            per alias, tail_tokens.
        live: alias -> live document count (float; expected at plan
            time).
        lengths: alias -> live documents' token lengths or AliasStats.
        resident: aliases whose prefix KV a filter computes before the
            joins; their first anchor use pays no prefix.
        pre: Engine preamble token count in front of every prompt.
        chunk_tokens: Chunk token budget.
        model: Model spec, for the cost model.
        device: Device spec, for the cost model.
        already_joined: aliases connected by completed join stages.
            Used when costing a continuation of a partial plan.
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

    lengths = alias_stats(lengths, model.sliding_window)
    resident = set(resident or ())

    def rank(work):
        seconds = speed_of_light(base_work + work, model, device,
                                 chunk_tokens).seconds
        return (seconds, work.tokens, work.pairs, work.kv_written,
                work.kv_read)

    def run_walk(order_specs, assign):
        seq = list(zip(order_specs, assign))
        work, records = walk(
            seq, live, lengths, resident, pre, model, device)
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
                    kind = residency(anchor, state_now, resident,
                                     same_group)
                    kept = lengths[anchor].count if kind != "none" else 0
                    work = stage_work(spec, anchor, live_now, lengths, pre,
                                      resident=kind != "none",
                                      window=model.sliding_window)
                    step = dict(
                        written_pos=spec["written_pos"], anchor=anchor,
                        resident=kind, resident_docs=kept,
                        tuples=cross_tuples(spec, live_now),
                        tokens=work.tokens, work=work)
                    next_state = KVState(
                        anchor, spec["semantics"] == "full",
                        state_now.used_anchors | {anchor})
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
