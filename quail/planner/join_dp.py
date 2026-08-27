"""Fixed left deep join planning after filter answers are known."""

import itertools
import math
from dataclasses import dataclass

from quail.planner.left_deep import Extension, optimize_left_deep
from quail.planner.work import Work, ask, ideal_seconds, scan, stream
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class JoinSearch:
    sequence: tuple[tuple[int, str], ...]
    nodes: tuple[dict, ...]
    work: Work
    states: int
    generated: int


def _surviving_docs(n_docs: float, n_partners: float,
                    selectivity: float | None) -> float:
    if selectivity is None:
        return n_docs
    return n_docs * (1.0 - (1.0 - selectivity)
                     ** max(1.0, n_partners))


def _thin(live: dict[str, float], join: dict) -> None:
    aliases = join["aliases"]
    selectivity = join.get("selectivity")

    def others(alias):
        return math.prod(live[other] for other in aliases
                         if other != alias)

    if join["semantics"] == "full":
        updated = {
            alias: _surviving_docs(live[alias], others(alias), selectivity)
            for alias in aliases
        }
        live.update(updated)
        return
    anchor = join["anchor"]
    matched = _surviving_docs(live[anchor], others(anchor), selectivity)
    live[anchor] = (matched if join["semantics"] == "exists"
                    else live[anchor] - matched)


def _anchors(join: dict) -> tuple[str, ...]:
    if join["semantics"] != "full" or not join.get("anchor_free"):
        return (join["anchor"],)
    return tuple(join["aliases"])


def _fits(join: dict, anchor: str, docs: dict, survivors: dict,
          pre_tokens: int, chunk_tokens: int) -> bool:
    if any(not survivors[alias] for alias in join["aliases"]):
        return True
    partners = [alias for alias in join["aliases"] if alias != anchor]
    anchor_max = max((len(docs[anchor][row])
                      for row in survivors[anchor]), default=0)
    need = pre_tokens + anchor_max + len(join["frames"][anchor]) \
        + len(join["tail"])
    for partner in partners:
        partner_max = max((len(docs[partner][row])
                           for row in survivors[partner]), default=0)
        need += len(join["labels"][partner]) + partner_max
    return need <= chunk_tokens


def _stage_work(join: dict, anchor: str, live: dict[str, float],
                docs: dict, survivors: dict, pre_tokens: int,
                resident_keys: set, same_group: bool) -> Work:
    partners = [alias for alias in join["aliases"] if alias != anchor]
    partner_count = math.prod(live[alias] for alias in partners)
    if live[anchor] <= 0 or partner_count <= 0:
        return Work()
    suffix = len(join["tail"])
    for partner in partners:
        rows = survivors[partner]
        mean = (sum(len(docs[partner][row]) for row in rows) / len(rows)
                if rows else 0.0)
        suffix += len(join["labels"][partner]) + mean
    frame = len(join["frames"][anchor])
    base_rows = survivors[anchor]
    live_fraction = live[anchor] / max(1, len(base_rows))
    total = Work()
    for row in base_rows:
        prefix = pre_tokens + len(docs[anchor][row])
        hit = same_group or (anchor, row) in resident_keys
        start = ask(prefix, frame) if hit else scan(prefix, frame)
        pairs = stream(prefix + frame, suffix, partner_count)
        total += (start + pairs) * live_fraction
    return total


def _plan_nodes(sequence: tuple[tuple[int, str], ...],
                joins: list[dict]) -> tuple[dict, ...]:
    groups = []
    for join_index, anchor in sequence:
        join = joins[join_index]
        merge = (groups and join["semantics"] == "full"
                 and groups[-1]["full"]
                 and groups[-1]["anchor"] == anchor)
        if merge:
            groups[-1]["stage_idxs"].append(join_index)
        else:
            groups.append(dict(anchor=anchor,
                               full=(join["semantics"] == "full"),
                               stage_idxs=[join_index]))

    nodes = []
    barrier = 0
    previous = None
    for index, group in enumerate(groups):
        if previous is not None and previous != group["anchor"]:
            nodes.append(dict(id=f"runtime-barrier:{barrier}", op="Barrier",
                              inputs=(), next_anchor=group["anchor"],
                              thins=()))
            barrier += 1
        nodes.append(dict(id=f"runtime-group:{index}", op="JoinGroup",
                          inputs=(), anchor=group["anchor"],
                          stage_idxs=tuple(group["stage_idxs"]), stages=()))
        previous = group["anchor"]
    return tuple(nodes)


def optimize_joins_after_filters(
    joins: list[dict], docs: dict, survivors: dict, resident_keys: set,
    pre_tokens: int, model: ModelSpec, device: DeviceSpec,
    chunk_tokens: int,
) -> JoinSearch | None:
    """Plan full binary joins once from the actual filter survivors."""

    if not joins:
        return JoinSearch((), (), Work(), 0, 0)
    if any(join["semantics"] != "full"
           or len(join["aliases"]) != 2 for join in joins):
        return None

    aliases = tuple(sorted({alias for join in joins
                            for alias in join["aliases"]}))
    endpoints = [frozenset(join["aliases"]) for join in joins]
    base_live = {alias: float(len(survivors[alias])) for alias in aliases}
    live_cache = {}

    def live_for(relations: frozenset[str]):
        if relations not in live_cache:
            live = dict(base_live)
            active = [index for index, edge in enumerate(endpoints)
                      if edge <= relations]
            for index in sorted(active,
                                key=lambda i: joins[i].get("written_pos", i)):
                _thin(live, joins[index])
            live_cache[relations] = live
        return dict(live_cache[relations])

    def extend(relations: frozenset[str], current_anchor: str | None,
               added: str):
        crossing = [
            index for index, edge in enumerate(endpoints)
            if added in edge and edge & relations
        ]
        if not crossing:
            return ()

        extensions = []
        for order in itertools.permutations(crossing):
            anchor_lists = [_anchors(joins[index]) for index in order]
            for anchors in itertools.product(*anchor_lists):
                if any(not _fits(joins[index], anchor, docs, survivors,
                                 pre_tokens, chunk_tokens)
                       for index, anchor in zip(order, anchors)):
                    continue
                extra = Work()
                live = live_for(relations)
                previous_anchor = current_anchor
                steps = []
                for index, anchor in zip(order, anchors):
                    join = joins[index]
                    same = (previous_anchor == anchor
                            and join["semantics"] == "full")
                    extra += _stage_work(
                        join, anchor, live, docs, survivors,
                        pre_tokens, resident_keys, same)
                    _thin(live, join)
                    steps.append((index, anchor))
                    previous_anchor = anchor
                extensions.append(Extension(
                    work=extra,
                    state_property=previous_anchor,
                    steps=tuple(steps),
                ))
        return tuple(extensions)

    search = optimize_left_deep(aliases, None, Work(), extend)
    if not search.candidates:
        return None
    best = min(
        search.candidates,
        key=lambda candidate: (
            ideal_seconds(candidate.work, model, device, chunk_tokens),
            candidate.work.tokens,
            candidate.work.pairs,
            candidate.steps,
        ),
    )
    return JoinSearch(best.steps, _plan_nodes(best.steps, joins),
                      best.work, search.state_count,
                      search.generated_count)
