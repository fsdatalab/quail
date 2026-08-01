"""Online exact solver: eq. (54) as a stochastic shortest path.

The recurrence has cycles (prefill -> evict -> re-prefill revisits a state),
so naive recursion is not well-founded (convention C4). With positive batch
costs and at least one proper policy, the optimal value function is the
unique fixed point; we compute it by Gauss-Seidel value iteration over the
reachable state graph. Information order matches the paper: choose B,
observe x, choose E."""

from typing import Dict

from ..instance import Instance
from . import engine


def _transitions(inst: Instance, policy: str, kmax: int, evict_mode: str,
                 state, cache: Dict):
    """List of (cost, [(prob, [next_state, ...]), ...]) per feasible batch:
    outer list = batches; per batch, per outcome, the eviction-choice set."""
    if state in cache:
        return cache[state]
    acts = []
    for b in engine.candidate_batches(inst, policy, kmax, state):
        ev = engine.evaluate_batch(inst, policy, state, b)
        if ev is None:
            continue
        _st, cost, _ops = ev
        comps = engine.completions(inst, policy, state, b)
        outs = []
        for prob, outcomes in engine.outcome_branches(inst, comps, s=inst.s):
            mid = engine.apply_batch(inst, policy, state, b, outcomes)
            nexts = list(dict.fromkeys(
                engine.eviction_choices(inst, policy, mid, evict_mode)))
            outs.append((prob, nexts))
        acts.append((cost, outs))
    cache[state] = acts
    return acts


def solve_online(inst: Instance, policy: str, kmax: int = 1,
                 evict_mode: str = "suffix", tol: float = 1e-10,
                 max_iters: int = 100_000, max_states: int = 500_000):
    """Optimal expected makespan OPT_on(d, s) and the reachable state count."""
    inst.check()
    start = engine.initial_state(inst)

    # enumerate reachable states (BFS closure over all actions/outcomes/evictions)
    cache: Dict = {}
    seen = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        if engine.terminal(inst, state):
            continue
        for _cost, outs in _transitions(inst, policy, kmax, evict_mode, state, cache):
            for _prob, nexts in outs:
                for nxt in nexts:
                    if nxt not in seen:
                        seen.add(nxt)
                        if len(seen) > max_states:
                            raise RuntimeError("state explosion in online solve")
                        frontier.append(nxt)

    V = {s: 0.0 if engine.terminal(inst, s) else 1e18 for s in seen}
    states = [s for s in seen if not engine.terminal(inst, s)]
    for _ in range(max_iters):
        delta = 0.0
        for s in states:
            best = 1e18
            for cost, outs in cache.get(s, ()):
                total = cost
                for prob, nexts in outs:
                    total += prob * min(V[nx] for nx in nexts)
                    if total >= best:
                        break
                if total < best:
                    best = total
            if abs(best - V[s]) > delta:
                delta = abs(best - V[s])
            V[s] = best
        if delta < tol:
            break
    else:
        raise RuntimeError("value iteration did not converge")
    return V[start], len(seen)
