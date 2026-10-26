"""Time left while a query runs: the plan's SoL seconds corrected by progress."""

import json
import random
import time

import pyarrow as pa
import pytest
from helpers.oracle import Oracle, finish

import quail
from quail.catalog import DocumentProvider
from quail.execution.remaining import (
    RemainingTimeTracker,
    duration,
    prediction_errors,
    replay,
)
from quail.planner.plan import EngineConfig
from quail.progress import Progress, listening, running


class Clock:
    """A perf_counter that moves only when the test advances it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture()
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(time, "perf_counter", value)
    return value


def tracker_for(planned, kinds=None, expected_rows=None, **options):
    """A tracker over nodes priced at planned[node_id] SoL seconds."""
    kinds = kinds or {}
    expected_rows = expected_rows or {}
    return RemainingTimeTracker({
        node_id: {"seconds": seconds,
                  "kind": kinds.get(node_id, "quail.ai_filter"),
                  "expected_rows": expected_rows.get(node_id)}
        for node_id, seconds in planned.items()}, **options)


def run_group(clock, tracker, node_ids, seconds, total=1000, rate=None,
              every=5.0, latency=0):
    """Count one group's work as it runs for seconds, reporting every 5 s.

    rate, when given, is a callable(step) -> a weight for that second's
    share of the work; the weights are scaled to finish at seconds.
    latency is the seconds at the start before any row finishes.
    """
    steps = int(seconds)
    weights = [0.0 if step < latency else rate(step) if rate else 1.0
               for step in range(steps)]
    scale = total / sum(weights)
    with running(node_ids), listening(tracker):
        progress = Progress("op", total=total, every=every,
                            emit=lambda line: None)
        done = 0.0
        for weight in weights:
            clock.now += 1.0
            done += weight * scale
            progress.update(min(total, int(done)))
        progress.done = total
        progress.finish("op done")


def test_no_estimate_exists_before_a_count(clock):
    tracker = tracker_for({"f": 10.0})
    assert tracker.estimate() is None
    with listening(tracker):
        outside = Progress("tokenizing", total=10, every=0,
                           emit=lambda line: None)
        clock.now = 1.0
        outside.update(5)
    assert tracker.estimate() is None, "counts outside a node are ignored"
    assert tracker.report()["history"] == []


def test_one_operator_at_a_steady_rate_predicts_its_finish(clock):
    tracker = tracker_for({"f": 10.0})
    run_group(clock, tracker, ("f",), seconds=30)
    report = tracker.report()
    assert report["finished_seconds"] == 30
    assert report["groups"][0]["slowdown"] == pytest.approx(3.0)
    errors = prediction_errors(report)
    assert errors[0.1] is None, "the first count comes 5 s in, after 3 s"
    assert errors[0.25] == pytest.approx(0.0, abs=0.01)
    assert errors[0.5] == pytest.approx(0.0, abs=0.01)


def test_a_running_operator_is_described_against_its_sol_seconds(clock):
    tracker = tracker_for({"f": 60.0, "j": 120.0},
                          kinds={"j": "quail.ai_join"})
    seen = []
    tracker.on_remaining = seen.append
    with running(("f",)), listening(tracker):
        progress = Progress("op", total=100, every=0, emit=lambda line: None)
        clock.now = 45.0
        progress.update(50)
    # 45 s of filter left at its rate, and the join's 120 s at 1.5x
    assert seen[-1].seconds == pytest.approx(45 + 180)
    assert seen[-1].describe() == (
        "about 3.8 min left (planned 3.0 min at SoL, running 1.5x slower)")


def test_later_operators_of_a_measured_kind_use_that_kind_slowdown(clock):
    tracker = tracker_for({"f1": 10.0, "j": 10.0, "f2": 10.0},
                          kinds={"j": "quail.ai_join"})
    run_group(clock, tracker, ("f1",), seconds=10)       # filters at 1x
    run_group(clock, tracker, ("j",), seconds=50)        # joins at 5x
    # f2 has not started: the filter slowdown applies, not the query's 3x
    assert tracker.estimate().seconds == pytest.approx(10.0)


def test_an_unmeasured_kind_falls_back_to_the_query_slowdown(clock):
    tracker = tracker_for({"f": 10.0, "j": 20.0}, kinds={"j": "quail.ai_join"})
    run_group(clock, tracker, ("f",), seconds=30)        # 3x
    run_group(clock, tracker, ("j",), seconds=20)        # 1x
    errors = prediction_errors(tracker.report())
    # at 5 s, 1/6 of the filter is done: 25 s of it left, the join at 3x;
    # counts are whole documents, so the rate is about 1% off
    assert errors[0.1] == pytest.approx((5 + 25 + 60 - 50) / 50, abs=0.02)
    assert errors[0.5] == pytest.approx((25 + 5 + 60 - 50) / 50, abs=0.02)


def test_a_pipeline_counts_the_sol_seconds_of_every_member(clock):
    tracker = tracker_for({"f": 4.0, "j": 6.0}, kinds={"j": "quail.ai_join"})
    with running(("f", "j")), listening(tracker):
        progress = Progress("pipeline r", total=10, every=0,
                            emit=lambda line: None)
        clock.now = 15.0
        progress.update(5)
    estimate = tracker.estimate()
    assert estimate.slowdown == pytest.approx(3.0)
    assert estimate.seconds == pytest.approx(15.0), "the members overlap"
    assert tracker.report()["groups"][0]["kind"] == "quail.ai_filter+quail.ai_join"


def test_an_unknown_total_uses_the_planned_rows(clock):
    tracker = tracker_for({"j": 10.0}, kinds={"j": "quail.ai_join"},
                          expected_rows={"j": 100})
    seen = []
    tracker.on_remaining = seen.append
    with running(("j",)), listening(tracker):
        progress = Progress("join", total=None, every=0,
                            emit=lambda line: None)
        clock.now = 20.0
        progress.update(50)
    assert seen[-1].planned_total
    assert seen[-1].seconds == pytest.approx(20.0)
    assert seen[-1].describe().endswith(", total from the plan")


def test_a_group_that_stops_reporting_when_another_starts_has_finished(clock):
    tracker = tracker_for({"f1": 10.0, "f2": 10.0})
    with running(("f1",)), listening(tracker):
        first = Progress("f1", total=10, every=0, emit=lambda line: None)
        clock.now = 8.0
        first.update(8)
    with running(("f2",)), listening(tracker):
        second = Progress("f2", total=10, every=0, emit=lambda line: None)
        clock.now = 9.0
        second.update(1)
    groups = tracker.report()["groups"]
    assert groups[0]["slowdown"] == pytest.approx(0.8)
    assert tracker.estimate().seconds == pytest.approx(9.0)


@pytest.mark.parametrize("slowdowns", [
    {"quail.ai_filter": 3.0},
    {"quail.ai_filter": 1.2},
    {"quail.ai_filter": 8.0},
    {"quail.ai_filter": 2.0, "quail.ai_join": 2.0},
])
def test_steady_slowdowns_predict_within_a_few_percent(clock, slowdowns):
    planned = {"f1": 20.0, "f2": 15.0, "j": 30.0}
    kinds = {"j": "quail.ai_join"}
    tracker = tracker_for(planned, kinds=kinds)
    for node_id, seconds in planned.items():
        kind = kinds.get(node_id, "quail.ai_filter")
        run_group(clock, tracker, (node_id,),
                  seconds=seconds * slowdowns.get(kind, slowdowns["quail.ai_filter"]))
    errors = prediction_errors(tracker.report())
    for fraction in (0.25, 0.5):
        assert abs(errors[fraction]) < 0.03, errors


@pytest.mark.parametrize("seed", range(5))
def test_a_noisy_rate_predicts_within_a_quarter_by_half_time(clock, seed):
    noise = random.Random(seed)
    tracker = tracker_for({"f1": 30.0, "f2": 30.0})
    for node_id in ("f1", "f2"):
        run_group(clock, tracker, (node_id,), seconds=90,
                  rate=lambda step: noise.uniform(0.6, 1.4))
    errors = prediction_errors(tracker.report())
    assert abs(errors[0.5]) < 0.25, errors


def test_seconds_before_the_first_finished_row_do_not_slow_the_rate(clock):
    # rows finish only after every stage: none in the first 10 s of 60
    tracker = tracker_for({"c": 30.0}, kinds={"c": "quail.ai_classify"})
    run_group(clock, tracker, ("c",), seconds=60, latency=10)
    report = tracker.report()
    from_start = prediction_errors(replay(report, rate_from_first=False))
    from_first = prediction_errors(report)
    assert from_start[0.5] > 0.1, "the idle start lowers the rate all run"
    assert abs(from_first[0.5]) < 0.03


def test_a_saved_report_replays_to_the_same_estimates(clock):
    tracker = tracker_for({"f": 10.0, "j": 20.0}, kinds={"j": "quail.ai_join"})
    run_group(clock, tracker, ("f",), seconds=30)
    run_group(clock, tracker, ("j",), seconds=20, latency=4)
    report = json.loads(json.dumps(tracker.report()))
    assert replay(report)["history"] == report["history"]


def test_durations_read_as_seconds_minutes_and_hours():
    assert duration(0.25) == "0.2 s"
    assert duration(47) == "47 s"
    assert duration(240) == "4.0 min"
    assert duration(3 * 3600) == "3.0 h"


REVIEWS = pa.table({"id": [1, 2, 3, 4],
                    "text": ["good film", "bad film", "great cast", "dull"]})


def test_a_query_report_holds_each_operator_slowdown():
    progress = []

    def truth(alias, document, position):
        if not progress:
            progress.append(Progress("filter", total=len(REVIEWS), every=0,
                                     emit=lambda line: None))
        progress[0].update(progress[0].done + 1)
        return document % 2 == 0

    with quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm"),
                       tokenizer=str.split) as session:
        session.register("reviews",
                         DocumentProvider.from_table(REVIEWS, id_col="id"))
        query = session.sql(
            "SELECT r.id FROM reviews r WHERE AI.IF(PROMPT('good? {0}', r.text))",
            dialect="bq")
        filter_id = next(node.node_id for node in query.plan().nodes
                         if node.type_name == "quail.ai_filter")
        result = finish(query, Oracle(filters=truth))
    assert result.collect().num_rows == 2
    report = result.report["remaining_time"]
    assert report["planned_seconds"] > 0
    assert [group["node_ids"] for group in report["groups"]] == [[filter_id]]
    assert report["groups"][0]["kind"] == "quail.ai_filter"
    assert report["history"], "an estimate was made while the filter ran"
