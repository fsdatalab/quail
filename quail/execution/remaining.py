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
    """Plan nodes counted by one Progress: one node, or one pipeline.

    first is the elapsed seconds and count of the first report with a
    finished row, or None before one. work is the tokens run and the
    tokens still expected, when the Progress reports them.
    """

    kind: str
    planned_seconds: float
    elapsed: float = 0.0
    fraction: float = 0.0
    done: int = 0
    total: int | None = None
    first: tuple[float, int] | None = None
    work: tuple[int, int] | None = None


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

    When the Progress reports work, a running group's share done and
    its rate are in tokens: rows that pass an early stage cost more
    than rows it drops, so finished rows are not an even measure.
    Otherwise the rate is in rows, measured from the group's first
    report with a finished row, because rows in flight finish only
    after every stage runs.

    Args:
        nodes: Priced node id -> {"seconds", "kind", "expected_rows"}.
        on_remaining: Optional callable run with each RemainingTime,
            after it is logged.
        rate_from_first: Measure a running group's rate in rows from its
            first finished row; False measures it from the group's start.
        use_work: Measure a running group in tokens when its Progress
            reports work; False measures it in rows.
    """

    def __init__(self, nodes, on_remaining=None, rate_from_first=True,
                 use_work=True):
        self.nodes = dict(nodes)
        self.planned_seconds = sum(node["seconds"] for node in self.nodes.values())
        self.on_remaining = on_remaining
        self.rate_from_first = rate_from_first
        self.use_work = use_work
        self.started = time.perf_counter()
        self.groups: dict[tuple, _Group] = {}
        self.history: list[tuple[float, float]] = []
        self.counts: list[list] = []

    @classmethod
    def for_plan(cls, graph, estimates,
                 on_remaining=None) -> "RemainingTimeTracker":
        """Return a tracker over the nodes of graph that estimates prices."""
        return cls({
            node.node_id: {
                "seconds": estimates[node.node_id]["seconds"],
                "kind": node.type_name,
                "expected_rows": getattr(node, "expected_rows", None)}
            for node in graph.nodes
            if estimates.get(node.node_id, {}).get("seconds")}, on_remaining)

    def start(self) -> None:
        """Log the planner's SoL seconds as a lower bound."""
        self.started = time.perf_counter()
        if self.planned_seconds:
            say(f"planned {duration(self.planned_seconds)} at SoL, "
                f"a lower bound on the time left")

    def __call__(self, progress) -> None:
        """Record one Progress count and report the new estimate."""
        estimate = self.count(
            time.perf_counter() - self.started, progress.node_ids,
            progress.done, progress.total, progress.elapsed, progress.finished,
            progress.work_done, progress.work_left)
        if estimate is None:
            return
        say(f"query: {estimate.describe()}")
        if self.on_remaining is not None:
            self.on_remaining(estimate)

    def count(self, at, node_ids, done, total, elapsed, finished,
              work_done=None, work_left=None):
        """Record one count made at seconds into the run.

        Returns:
            The new RemainingTime, or None when the count is not of a
            priced node or no work has been measured yet.
        """
        node_ids = tuple(node_id for node_id in node_ids
                         if node_id in self.nodes)
        if not node_ids:
            return None
        self.counts.append([at, list(node_ids), done, total, elapsed, finished,
                            work_done, work_left])
        planned_total = total is None
        if planned_total:
            total = self.nodes[node_ids[0]]["expected_rows"]
        group = self.groups.get(node_ids)
        if group is None:
            for other in self.groups.values():
                other.fraction = 1.0
            group = self.groups[node_ids] = _Group(
                kind=_kind(self.nodes[node_id]["kind"] for node_id in node_ids),
                planned_seconds=sum(self.nodes[i]["seconds"] for i in node_ids))
        group.elapsed, group.done, group.total = elapsed, done, total
        group.work = (None if not self.use_work or work_done is None
                      or not work_done + work_left
                      else (work_done, work_left))
        if finished:
            group.fraction = 1.0
        elif group.work is not None:
            group.fraction = work_done / (work_done + work_left)
        else:
            group.fraction = min(1.0, done / total) if total else 0.0
        if group.first is None and done and not finished:
            group.first = (elapsed, done)
        estimate = self.estimate(planned_total)
        if estimate is not None:
            self.history.append((at, estimate.seconds))
        return estimate

    def _group_seconds(self, group) -> float:
        """Seconds left for a running group with a measured count."""
        if group.work is not None:
            work_done, work_left = group.work
            return group.elapsed * work_left / work_done
        if self.rate_from_first and group.first is not None:
            first_elapsed, first_done = group.first
            if group.done > first_done and group.elapsed > first_elapsed:
                rate = (group.done - first_done) / (group.elapsed - first_elapsed)
                return (group.total - group.done) / rate
        return group.elapsed * (1 - group.fraction) / group.fraction

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
                seconds += self._group_seconds(group)
            else:
                seconds += group.planned_seconds * slowdown
        started = {node_id for node_ids in self.groups for node_id in node_ids}
        for node_id, node in self.nodes.items():
            if node_id not in started:
                elapsed, kind_planned = by_kind.get(node["kind"], (0.0, 0.0))
                seconds += node["seconds"] * (elapsed / kind_planned if kind_planned
                                              else slowdown)
        return RemainingTime(seconds, self.planned_seconds, slowdown,
                             planned_total)

    def report(self) -> dict:
        """Return the run's counts, each group's slowdown, and every estimate."""
        return {
            "planned_seconds": self.planned_seconds,
            "finished_seconds": time.perf_counter() - self.started,
            "nodes": self.nodes,
            "groups": [
                {"node_ids": list(node_ids), "kind": group.kind,
                 "planned_seconds": group.planned_seconds,
                 "measured_seconds": group.elapsed,
                 "slowdown": (group.elapsed / group.planned_seconds
                              if group.fraction >= 1.0 else None)}
                for node_ids, group in self.groups.items()],
            "counts": self.counts,
            "history": [list(point) for point in self.history],
        }


def replay(report: dict, **options) -> dict:
    """Run a saved report's counts through a new tracker on the CPU.

    Args:
        report: A RemainingTimeTracker report saved after the run.
        **options: RemainingTimeTracker options, such as use_work.

    Returns:
        The report with the new tracker's estimates as its history.
    """
    tracker = RemainingTimeTracker(report["nodes"], **options)
    for count in report["counts"]:
        tracker.count(*count)
    return {**report, "history": [list(point) for point in tracker.history]}


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


def _kind(kinds) -> str:
    return "+".join(dict.fromkeys(kinds))
