"""waves_logic must plan waves without an engine."""

from quail.engineext.waves_logic import plan_waves


def test_plan_waves_chunks_in_the_order_given():
    cands = [(0, 300), (1, 300), (2, 300), (3, 300)]
    assert plan_waves(cands, wave_tokens=650, budget_tokens=10_000) == [
        [0, 1], [2, 3]]
    assert plan_waves(cands, wave_tokens=300, budget_tokens=10_000) == [
        [0], [1], [2], [3]]


def test_plan_waves_stops_at_the_pool_budget():
    cands = [(0, 300), (1, 300), (2, 300), (3, 300)]
    assert plan_waves(cands, wave_tokens=650, budget_tokens=700) == [[0, 1]]
    assert plan_waves(cands, wave_tokens=650, budget_tokens=0) == []


def test_plan_waves_gives_an_oversized_document_its_own_wave():
    cands = [(7, 5000), (8, 100)]
    assert plan_waves(cands, wave_tokens=650, budget_tokens=10_000) == [
        [7], [8]]
