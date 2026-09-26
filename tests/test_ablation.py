"""Feature switches for the history ablation: registry, planner, scheduler."""

import pyarrow as pa
import pytest

import quail
import quail_b as benchmark
from quail import ablation
from quail.backends.quail.executor.pack import JoinAdmission
from quail.bench.quailb import build_query
from quail.logical import bind_join_prompt
from quail.physical import AiFilter, AiJoin

CONFIG = dict(model="qwen3-4b-fp8", device="h100-sxm")


@pytest.fixture(autouse=True)
def every_feature_on():
    yield
    ablation.configure(())


def byte_tokens(text):
    return list(text.encode())


def bio4_plan(disabled=()):
    """Plan BIO-4 on small tables with the given features off."""
    config = quail.EngineConfig(**CONFIG, disabled_features=tuple(disabled))
    with quail.Session(config, tokenizer=byte_tokens) as session:
        session.register("reports", quail.DocumentProvider.from_table(pa.table({
            "id": [f"r{i}" for i in range(6)],
            "report": [f"report {i} " + "text " * 400 for i in range(6)],
            "source": ["x"] * 6,
        }), id_col="id"))
        session.register("terms", quail.DocumentProvider.from_table(pa.table({
            "id": [f"t{i}" for i in range(9)],
            "term": [f"term {i}" for i in range(9)],
        }), id_col="id"))
        query = build_query(session, benchmark.get_query("BIO-4"))
        return query.plan(), query


def test_configure_rejects_unknown_names_and_dates_select_later_features():
    with pytest.raises(ValueError, match="unknown features"):
        ablation.configure(("no_such_feature",))
    later = ablation.merged_after("2026-09-08T01:16:41Z")
    assert "filter_join_streaming" in later and "gigatoken" in later
    assert "plan_on_estimates" not in later
    assert ablation.merged_after("2026-12-31T00:00:00Z") == frozenset()
    with pytest.raises(ValueError, match="gpus=1"):
        quail.Session(quail.EngineConfig(
            **CONFIG, gpus=2, disabled_features=("scan_ring",)),
            tokenizer=byte_tokens)


def test_every_feature_on_streams_the_anchor_filter_into_the_join():
    plan, _ = bio4_plan()
    reports = next(n for n in plan.nodes
                   if isinstance(n, AiFilter) and n.alias == "r")
    assert reports.pin_survivors and not reports.keep_kv
    terms = next(n for n in plan.nodes
                 if isinstance(n, AiFilter) and n.alias == "n")
    assert not terms.arena_writes


def test_planner_switches_change_the_bio4_plan():
    plan, _ = bio4_plan(("filter_join_streaming",))
    reports = next(n for n in plan.nodes
                   if isinstance(n, AiFilter) and n.alias == "r")
    assert not reports.pin_survivors and reports.keep_kv
    assert "r" in plan.settings["retention"]["initial"]

    plan, _ = bio4_plan(("filter_kv_reuse", "filter_join_streaming"))
    reports = next(n for n in plan.nodes
                   if isinstance(n, AiFilter) and n.alias == "r")
    assert not reports.pin_survivors and not reports.keep_kv
    assert plan.settings["retention"]["initial"] == {}
    assert plan.settings["filter_order_rule"] == "as_written"
    join = next(n for n in plan.nodes if isinstance(n, AiJoin))
    assert not join.keep_anchor_kv

    on = bio4_plan()[0].settings["retention"]["cap_pages"]
    off = bio4_plan(("scan_ring",))[0].settings["retention"]["cap_pages"]
    assert off > on

    plan, _ = bio4_plan(("skip_arena_writes",))
    assert all(n.arena_writes for n in plan.nodes if isinstance(n, AiFilter))


def test_join_search_off_keeps_written_order_and_the_largest_anchor():
    # the cardiovascular join passes fewer terms, so the search runs it first
    searched = next(n for n in bio4_plan(("filter_kv_reuse",))[0].nodes
                    if isinstance(n, AiJoin))
    written = next(n for n in bio4_plan(("join_search",))[0].nodes
                   if isinstance(n, AiJoin))
    assert written.anchor == searched.anchor == "r"
    assert [stage.written_pos for stage in written.stages] == [0, 1]
    assert [stage.written_pos for stage in searched.stages] == [1, 0]


def test_projection_pushdown_off_keeps_every_source_column():
    _, on = bio4_plan()
    _, off = bio4_plan(("projection_pushdown",))
    scans = {scan.alias: scan for scan in off.logical.operators().scans}
    assert scans["r"].columns == ("id", "report", "source")
    kept = {scan.alias: scan for scan in on.logical.operators().scans}
    assert "source" not in kept["r"].columns


def test_shared_join_prompts_off_moves_the_question_after_every_partner():
    args = (quail.ColumnRef("r", "reports", "report"),
            quail.ColumnRef("n", "terms", "term"))
    template = "Does {0} mention {1}?"
    shared = bind_join_prompt(template, args, byte_tokens)
    ablation.configure(("shared_join_prompts",))
    per_pair = bind_join_prompt(template, args, byte_tokens)
    assert per_pair.frame == ""
    assert per_pair.tail_tokens == shared.tail_tokens + shared.frame_tokens
    shared_frame = dict((a, f) for a, _, f in shared.labels)["r"]
    pair_frame = dict((a, f) for a, _, f in per_pair.labels)["r"]
    assert pair_frame == shared_frame - shared.frame_tokens


def test_stage_barriers_hold_the_next_stage_until_the_group_answers():
    # three anchors, two fit the arena at once; two stages of two partners
    sched = JoinAdmission([100, 100, 100], [[10, 10], [10, 10]], 1000,
                          arena_pages=16, page_tokens=16,
                          stage_barriers=True)
    free = 16
    first = sched.next_chunk(free)
    assert [(a, j) for a, j, *_ in first] == [(0, 0), (1, 0)]
    assert sched.blocked_pages > 0
    # anchor 0 passes its stage; it waits for anchor 1's answers
    assert sched.report(0, 0, 0, 2, [1, 0]) == []
    assert sched.next_chunk(0) == []
    assert sched.report(1, 0, 0, 2, [0, 1]) == []
    second = sched.next_chunk(0)
    assert sorted((a, j) for a, j, *_ in second) == [(0, 1), (1, 1)]
    # anchor 2 waits for the whole group, then starts a new one
    assert sched.report(0, 1, 0, 2, [1, 1]) == [("finished", 0)]
    assert sched.next_chunk(8) == []
    assert sched.report(1, 1, 0, 2, [0, 0]) == [("finished", 1)]
    assert [(a, j) for a, j, *_ in sched.next_chunk(16)] == [(2, 0)]


def test_continuous_admission_starts_the_next_stage_at_once():
    sched = JoinAdmission([100, 100, 100], [[10, 10], [10, 10]], 1000,
                          arena_pages=16, page_tokens=16)
    sched.next_chunk(16)
    sched.report(0, 0, 0, 2, [1, 0])
    assert [(a, j) for a, j, *_ in sched.next_chunk(0)] == [(0, 1)]
