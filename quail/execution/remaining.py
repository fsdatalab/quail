"""Estimate a running query's time left from its plan and progress counts.

The planner prices each node in speed of light (SoL) seconds. A
node's slowdown is its measured seconds over its SoL seconds for the
share of its work that has finished. The time left is the running
nodes' remaining work at their measured rate, plus each node not yet
started at its SoL seconds times the slowdown measured so far.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from quail.progress import say

FRACTIONS = (0.1, 0.25, 0.5)


@dataclass(frozen=True)
class RemainingTime:
    """One estimate of a running query's time left.

    Attributes:
        seconds: Estimated seconds until the query's operators finish.
        planned_seconds: The planner's SoL seconds for every operator.
        slowdown: Measured seconds over SoL seconds for the work done.
        planned_total: Whether the running operators' total is the
            planner's expected rows, because the real count is unknown.
    """

    seconds: float
    planned_seconds: float
    slowdown: float
    planned_total: bool = False

    def describe(self) -> str:
        """Return the estimate as a log line."""
        text = (f"about {duration(self.seconds)} left (planned "
                f"{duration(self.planned_seconds)} at SoL, running "
                f"{self.slowdown:.1f}x slower)")
        return text + ", total from the plan" if self.planned_total else text


@dataclass
class _Group:
    """Plan nodes counted by one Progress: one node, or one pipeline."""

    kind: str
    planned_seconds: float
    elapsed: float = 0.0
    fraction: float = 0.0


def duration(seconds: float) -> str:
    """Format seconds as seconds, minutes, or hours."""
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


class RemainingTimeTracker:
    """Correct a plan's SoL seconds with the progress counts of its run.

    Call it with each ``quail.progress.Progress`` that reports, through
    ``quail.progress.listening``. Counts of nodes the plan did not
    price, and counts outside any node, are ignored. One group of nodes
    runs at a time, so a group that stops reporting when another starts
    has finished.

    Args:
        graph: The physical graph that runs.
        estimates: Node id -> its plan estimate entry; an entry with
            ``seconds`` is priced.
        emit: Optional callable run with each RemainingTime, after it
            is logged.
    """

    def __init__(self, graph, estimates, emit=None):
        self.nodes = {node.node_id: node for node in graph.nodes}
        self.seconds = {
            node_id: entry["seconds"] for node_id, entry in estimates.items()
            if entry.get("seconds") and node_id in self.nodes}
        self.planned_seconds = sum(self.seconds.values())
        self.emit = emit
        self.started = time.perf_counter()
        self.groups: dict[tuple, _Group] = {}
        self.history: list[tuple[float, float]] = []

    def start(self) -> None:
        """Log the planner's SoL seconds as a lower bound."""
        self.started = time.perf_counter()
        if self.planned_seconds:
            say(f"planned {duration(self.planned_seconds)} at SoL, "
                f"a lower bound on the time left")

    def __call__(self, progress) -> None:
        """Record one Progress count and report the new estimate."""
        node_ids = tuple(node_id for node_id in progress.node_ids
                         if node_id in self.seconds)
        if not node_ids:
            return
        total, planned_total = progress.total, False
        if total is None:
            total = getattr(self.nodes[node_ids[0]], "expected_rows", None)
            planned_total = True
        group = self.groups.get(node_ids)
        if group is None:
            for other in self.groups.values():
                other.fraction = 1.0
            group = self.groups[node_ids] = _Group(
                kind=_kind(self.nodes[node_id] for node_id in node_ids),
                planned_seconds=sum(self.seconds[i] for i in node_ids))
        group.elapsed = progress.elapsed
        group.fraction = (1.0 if progress.finished
                          else min(1.0, progress.done / total) if total else 0.0)
        estimate = self.estimate(planned_total)
        if estimate is None:
            return
        self.history.append(
            (time.perf_counter() - self.started, estimate.seconds))
        say(f"query: {estimate.describe()}")
        if self.emit is not None:
            self.emit(estimate)

    def estimate(self, planned_total: bool = False) -> RemainingTime | None:
        """Return the time left, or None before any work is measured."""
        measured = [group for group in self.groups.values() if group.fraction]
        done = sum(group.planned_seconds * group.fraction for group in measured)
        if not done:
            return None
        slowdown = sum(group.elapsed for group in measured) / done
        by_kind = {}
        for group in measured:
            elapsed, planned = by_kind.get(group.kind, (0.0, 0.0))
            by_kind[group.kind] = (elapsed + group.elapsed,
                                   planned + group.planned_seconds * group.fraction)
        seconds = 0.0
        for group in self.groups.values():
            if group.fraction >= 1.0:
                continue
            if group.fraction:
                seconds += group.elapsed * (1 - group.fraction) / group.fraction
            else:
                seconds += group.planned_seconds * slowdown
        started = {node_id for node_ids in self.groups for node_id in node_ids}
        for node_id, planned in self.seconds.items():
            if node_id not in started:
                elapsed, kind_planned = by_kind.get(
                    _kind((self.nodes[node_id],)), (0.0, 0.0))
                seconds += planned * (elapsed / kind_planned if kind_planned
                                      else slowdown)
        return RemainingTime(seconds, self.planned_seconds, slowdown,
                             planned_total)

    def report(self) -> dict:
        """Return each group's slowdown and the estimates made during the run."""
        return {
            "planned_seconds": self.planned_seconds,
            "finished_seconds": time.perf_counter() - self.started,
            "groups": [
                {"node_ids": list(node_ids), "kind": group.kind,
                 "planned_seconds": group.planned_seconds,
                 "measured_seconds": group.elapsed,
                 "slowdown": (group.elapsed / group.planned_seconds
                              if group.fraction >= 1.0 else None)}
                for node_ids, group in self.groups.items()],
            "history": [list(point) for point in self.history],
        }


def prediction_errors(report: dict, fractions=FRACTIONS) -> dict:
    """Return the error of the predicted finish at points of a run.

    Args:
        report: A RemainingTimeTracker report saved after the run.
        fractions: Shares of the run's seconds to check at.

    Returns:
        Fraction -> the predicted finish minus the actual finish, over
        the actual finish, from the last estimate made by that point;
        None where no estimate had been made yet.
    """
    finished = report["finished_seconds"]
    errors = {}
    for fraction in fractions:
        made = [(at, left) for at, left in report["history"]
                if at <= fraction * finished]
        if not made:
            errors[fraction] = None
            continue
        at, left = made[-1]
        errors[fraction] = (at + left - finished) / finished
    return errors


def _kind(nodes) -> str:
    return "+".join(dict.fromkeys(node.type_name for node in nodes))
