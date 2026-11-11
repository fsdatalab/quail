"""Censored answer lengths and eviction priorities."""

from dataclasses import replace

import numpy as np
import pytest
from test_kv_reads import named

from quail.common.answers import remaining_tokens, survival_rows
from quail.common.decode import DecodeCost
from quail.common.work import Work
from quail.specs import DEVICES, MODELS


def test_censored_answers_leave_the_risk_set_and_known_lengths_give_the_tokens_left():
    # Answers end at 2, 4, and 6; the censor at 3 leaves two at risk at 4.
    # A zero prior adds no events or risk.
    counts = np.bincount([2, 3, 4, 6], minlength=7)[None]
    events = np.bincount([2, 4, 6], minlength=7)[None]
    curve = survival_rows(counts, events, np.zeros((1, 7)), [6])[0]
    remaining = remaining_tokens(None, None, 3, 6, survival=curve)
    survival = np.array([1, 1, .75, .75, .375, .375, 0])
    expected = np.cumsum(survival[::-1])[::-1]
    expected[:-1] /= survival[:-1]
    np.testing.assert_allclose(remaining, expected)
    assert remaining[2] > remaining[4] > remaining[5]

    # an answer of known length has that length minus its written tokens
    # left: three answers end at 4; a zero prior adds no events or risk
    counts = events = np.bincount([4, 4, 4], minlength=7)[None]
    curve = survival_rows(counts, events, np.zeros((1, 7)), [6])[0]
    remaining = remaining_tokens(None, None, 3, 6, survival=curve)
    np.testing.assert_allclose(remaining[:5], [4, 3, 2, 1, 0])


def check_hybrid_decode_prices_state_movement_and_compute_from_specs(
        model_name, device_name):
    model, device = MODELS[model_name], DEVICES[device_name]
    cost = DecodeCost.from_specs(model, device)
    work = cost.row_work([16, 16], [0, 0], [16, 16]).total()
    assert work.kv_read == 16
    assert work.state_read == work.state_written == work.tokens == 2
    assert cost.state_seconds > 0
    assert cost.state_seconds == pytest.approx(model.checkpoint_bytes / device.hbm_bw)
    assert cost.seconds(work) > replace(cost, state_seconds=0).seconds(work)
    assert cost.seconds(Work()) == 0
    assert cost.seconds(cost.row_work([]).total()) == 0
    assert cost == DecodeCost.from_specs(
        replace(model, w_mem_bytes=1),
        replace(device, mamf_fp8_flops=1, mamf_bf16_flops=1))


def test_hybrid_decode_prices_state_movement_and_compute_from_specs():
    for model_name, device_name in [
        ("qwen3.5-4b-bf16", "h100-sxm"), ("lfm2-2.6b-bf16", "l40s"),
    ]:
        named(
            f"model_name={model_name} device_name={device_name}",
            check_hybrid_decode_prices_state_movement_and_compute_from_specs,
            model_name=model_name, device_name=device_name)
