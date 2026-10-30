"""Censored answer lengths and eviction priorities."""

from dataclasses import replace

import numpy as np
import pytest

from quail.backends.quail.executor.answer_survival import (
    remaining_tokens,
)
from quail.cost.answers import prior_survival
from quail.cost.decode import DecodeCost
from quail.cost.dense_decoder_cost import dense_decoder_components
from quail.cost.work import Work
from quail.specs import DEVICES, MODELS


def test_cold_start_uses_the_map_length_prior():
    prior = prior_survival(3, 8)
    remaining = remaining_tokens([], [], 3, 8)
    for written in range(8):
        assert np.isclose(remaining[written], prior[written + 1:].sum()
                          / prior[written])
    assert remaining[8] == 0
    np.testing.assert_allclose(remaining, remaining_tokens([], [], 3, 8,
                                                          survival=prior))


def test_censored_answers_leave_the_risk_set_after_their_written_length():
    # The censor at 3 leaves two answers at risk at 4.
    remaining = remaining_tokens([2, -1, 4, 6], [2, 3, 4, 6], 3, 6,
                                 prior_weight=0)
    survival = np.array([1, 1, .75, .75, .375, .375, 0])
    expected = np.r_[np.cumsum(survival[:0:-1])[::-1], 0]
    expected[:-1] /= survival[:-1]
    np.testing.assert_allclose(remaining, expected)
    assert remaining[2] > remaining[4] > remaining[5]


@pytest.mark.parametrize("model_name,device_name", [
    ("qwen3.5-4b-bf16", "h100-sxm"), ("qwen3.5-27b-fp8", "l40s"),
    ("qwen3.8-27b-fp8", "b200"), ("lfm2-2.6b-bf16", "l40s"),
])
def test_hybrid_decode_prices_state_movement_and_compute_from_specs(
        model_name, device_name):
    model, device = MODELS[model_name], DEVICES[device_name]
    cost = DecodeCost.from_specs(model, device)
    work = cost.row_work([16, 16], [0, 0], [16, 16]).total()
    assert work.kv_read == 16
    assert work.state_read == work.state_written == work.tokens == 2
    assert cost.state_seconds == model.checkpoint_bytes / device.hbm_bw > 0
    components = dense_decoder_components(work, model, passes=1)
    memory = sum(component.bytes_moved for component in components) / device.hbm_bw
    compute = sum(component.flops / device.arithmetic_bandwidth(component.precision)
                  for component in components)
    assert cost.seconds(work) == pytest.approx(max(memory, compute))
    assert cost.seconds(work) > replace(cost, state_seconds=0).seconds(work)
    compute_bound = DecodeCost.from_specs(model, replace(device, hbm_bw=float("inf")))
    assert compute_bound.seconds(work) == pytest.approx(compute)
    state_compute = sum(
        component.flops / device.arithmetic_bandwidth(component.precision)
        for component in components if component.name in ("gdn_core", "conv"))
    assert state_compute > 0
    assert cost.recompute_cost.linear_seconds * work.tokens > state_compute
    assert cost.seconds(Work()) == 0
    assert cost.seconds(cost.row_work([]).total()) == 0
    assert cost == DecodeCost.from_specs(
        replace(model, w_mem_bytes=1),
        replace(device, mamf_fp8_flops=1, mamf_bf16_flops=1))
