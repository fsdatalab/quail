"""waves_logic must decide waves and rotation without an engine."""

from quail.engineext.waves_logic import (full_blocks, plan_rotation,
                                         plan_wave, waves_to_gate)


def test_plan_wave_fills_the_budget_in_admission_order():
    cands = [(0, 300), (1, 300), (2, 300), (3, 300)]
    assert plan_wave(cands, wave_tokens=650, in_flight=0) == [0, 1]
    assert plan_wave(cands, wave_tokens=100, in_flight=0) == [0]


def test_plan_wave_respects_the_in_flight_bound():
    cands = [(0, 300)]
    assert plan_wave(cands, 650, in_flight=2) == []
    assert plan_wave(cands, 650, in_flight=1, max_in_flight=2) == [0]


def test_plan_wave_always_takes_one_oversized_document():
    assert plan_wave([(7, 5000)], wave_tokens=650, in_flight=0) == [7]


def test_waves_gate_once_each():
    assert waves_to_gate([3, 3, 5], already_gated={5}) == [3]
    assert waves_to_gate([3], already_gated={3}) == []


def test_full_blocks_drops_the_partial_tail():
    assert full_blocks(320, 16) == 20
    assert full_blocks(321, 16) == 20
    assert full_blocks(15, 16) == 0


def test_rotation_keeps_the_soonest_visits():
    cohorts = [(0, 100, 5), (1, 100, 1), (2, 100, 3)]
    keep, park = plan_rotation(cohorts, resident_capacity_tokens=200)
    assert keep == [1, 2] and park == [0]
    keep, park = plan_rotation(cohorts, resident_capacity_tokens=1000)
    assert keep == [1, 2, 0] and park == []
