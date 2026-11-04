"""The Rust cost model (crates/quail-cost) computes Python's numbers exactly."""

from dataclasses import asdict

import numpy as np
import pytest

from quail.common import work as python_work
from quail.common.dense_decoder_cost import (
    attention_pair_flops,
    conv_flops_per_token,
    dense_decoder_components,
    dense_params,
    gdn_core_flops_per_token,
)
from quail.common.sol import compute_seconds, speed_of_light, unrounded_seconds
from quail.specs import DEVICES, MODELS
from quail.specs.loading import DATA

native = pytest.importorskip("_quail_datafusion")


@pytest.fixture(scope="module")
def rust():
    return native.CostModel(str(DATA))


def plain(value):
    """A value with tuples and arrays as lists, for comparison."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    return value


PROPERTIES = (
    "reuses_kv", "attention_layers", "gdn_layers", "short_conv_layers",
    "state_layers", "conv_state_channels", "conv_channels", "gdn_state_bytes",
    "conv_state_bytes", "checkpoint_bytes", "kv_shapes", "widest_projection",
    "kappa", "kv_elements_per_token", "kappa_sliding", "kappa_full", "W_mem",
    "head_mem_bytes", "W_resident", "act_per_token", "intermediate",
)


def test_every_model_reads_with_the_same_derived_properties(rust):
    assert sorted(rust.model_names()) == sorted(MODELS)
    assert sorted(rust.device_names()) == sorted(DEVICES)
    for name, model in MODELS.items():
        expected = {key: plain(getattr(model, key)) for key in PROPERTIES}
        expected.update(
            dense_params=dense_params(model),
            gdn_core_flops_per_token=gdn_core_flops_per_token(model),
            conv_flops_per_token=conv_flops_per_token(model),
            attention_pair_flops=list(attention_pair_flops(model)),
        )
        assert plain(rust.model_properties(name)) == expected, name


def test_every_device_has_the_same_gemm_rates(rust):
    for name, device in DEVICES.items():
        for precision in ("fp8", "bf16", "fp32"):
            assert rust.gemm_flops(name, precision) == device.gemm_flops(precision)


DOCUMENTS = np.array([1.0, 37.0, 300.0, 4096.0, 70_000.0])
PREFIXES = (0.0, 1.0, 37.0, 5_000.0, DOCUMENTS)
SUFFIXES = (1.0, 12.0, 333.0)
WINDOWS = (0, 512, 4096)


def _compare_work(rust, kind, prefix, other, window):
    function = getattr(python_work, kind)
    expected = plain(asdict(function(prefix, other, window=window)))
    rust_other = plain(other) if not isinstance(other, float) else other
    assert plain(rust.work(kind, plain(prefix), rust_other, window)) == expected, (
        kind, prefix, other, window)


def test_the_work_functions_count_the_same_work(rust):
    for prefix in PREFIXES:
        for window in WINDOWS:
            for suffix in SUFFIXES:
                _compare_work(rust, "scan", prefix, suffix, window)
                _compare_work(rust, "ask", prefix, suffix, window)
            _compare_work(rust, "stream", prefix, SUFFIXES, window)
            _compare_work(rust, "packed", prefix, (12.0, 30.0, 7.0), window)
            _compare_work(rust, "packed", prefix, (12.0,), window)
            _compare_work(rust, "decode", prefix, (1.0, 4.0, 4.0), window)


# passes, answer rows, and head rows of a component pricing
READOUTS = ((0.0, 0.0, 0), (3.0, 12.0, 0), (1.5, 4.0, 2))


def _workloads():
    """Summed workloads of several shapes, as the planner prices them."""
    yield python_work.scan(DOCUMENTS, 40.0).total()
    yield python_work.ask(DOCUMENTS, 25.0, window=512).total()
    yield python_work.stream(3_000.0, SUFFIXES, window=4096).total()
    yield python_work.packed(DOCUMENTS, (12.0, 30.0)).total()
    yield python_work.decode(DOCUMENTS, (1.0, 4.0, 4.0)).total()
    yield python_work.Work(tokens=900.0, pairs=4e5, kv_written=900.0,
                           state_written=3.0, state_read=2.0, sae_encoded=900.0)


def test_the_components_and_times_are_the_same(rust):
    for model_name, model in MODELS.items():
        for work in _workloads():
            fields = plain(asdict(work))
            for passes, rows, head_rows in READOUTS:
                expected = [
                    (component.name, component.flops, component.bytes_moved)
                    for component in dense_decoder_components(
                        work, model, passes, rows, head_rows)]
                assert plain(rust.components(fields, model_name, passes, rows,
                                             head_rows)) == plain(expected)
            for device_name, device in DEVICES.items():
                for chunk_tokens, rows in ((1, 0.0), (8192, 12.0)):
                    assert rust.unrounded_seconds(
                        fields, model_name, device_name, chunk_tokens, rows, 0
                    ) == unrounded_seconds(work, model, device, chunk_tokens, rows=rows)
                    estimate = speed_of_light(work, model, device, chunk_tokens,
                                              rows=rows, min_passes=2)
                    assert rust.speed_of_light(
                        fields, model_name, device_name, chunk_tokens, rows, 2) == (
                        estimate.passes, estimate.seconds, estimate.compute,
                        estimate.memory, estimate.bytes_moved)
                for peak in (False, True):
                    assert rust.compute_seconds(
                        fields, model_name, device_name, peak
                    ) == compute_seconds(work, model, device, peak=peak)


def test_the_array_sum_rounds_as_numpy_does():
    generator = np.random.default_rng(0)
    for length in (0, 1, 7, 8, 9, 127, 128, 129, 1000, 4097):
        values = generator.random(length) * 10 ** generator.integers(0, 12, length)
        assert native.CostModel.numpy_sum(values.tolist()) == float(np.sum(values))
