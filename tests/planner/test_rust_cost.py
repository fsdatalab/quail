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


def _outcome(function, *arguments):
    """A function's value, or the message of the ValueError it raises."""
    try:
        return plain(function(*arguments))
    except ValueError as error:
        return ("error", str(error))


def test_the_memory_budgets_are_the_same(rust):
    from quail.common import budgets

    for model_name, model in MODELS.items():
        for device_name, device in DEVICES.items():
            expected = {
                "chunk_budget": budgets.chunk_budget(model, device),
                "chunk_memory_bound": budgets.chunk_memory_bound(model, device),
                "kernel_index_cap": budgets.kernel_index_cap(model),
                "minimum_weight_gpus": budgets.minimum_weight_gpus(model, device),
                "graph_memory_bytes": budgets.graph_memory_bytes(model, device),
                "graph_token_buckets": list(
                    budgets.graph_token_buckets(model, device)),
                "tree_attention_allowed": budgets.tree_attention_allowed(model),
                "sliding_kv_fits": budgets.sliding_kv_fits(model, device),
                "compute_knee": budgets.compute_knee(model, device),
            }
            pair = (model_name, device_name)
            assert plain(rust.model_budgets(model_name, device_name)) == expected, pair
            chunks = (1, 64, 2_048, max(1, expected["chunk_budget"]))
            for chunk in chunks:
                page_tokens = device.page_tokens
                questions = budgets.CHECKPOINT_PASS_QUESTIONS
                assert rust.chunk_budgets(model_name, device_name, chunk) == {
                    "arena_bytes": budgets.arena_bytes(model, device, chunk),
                    "checkpoint_memory_bytes": budgets.checkpoint_memory_bytes(
                        model, chunk, page_tokens),
                    "chunk_memory_bytes": budgets.chunk_memory_bytes(
                        model, chunk, page_tokens),
                    "copy_verification_workspace_bytes":
                        budgets.copy_verification_workspace_bytes(model, chunk),
                    "checkpoint_pass_bytes": budgets.checkpoint_pass_bytes(
                        model, questions, chunk),
                    "hybrid_pool_pages": budgets.hybrid_pool_pages(
                        model, device, chunk),
                    "checkpoint_pages": budgets.checkpoint_pages(model, page_tokens),
                }, (pair, chunk)
                for mean in (None, 1.0, 300.5, 5_000.0):
                    rust_pages = _outcome(
                        rust.arena_pages, model_name, device_name, chunk, mean)
                    python_pages = _outcome(
                        budgets.arena_pages, model, device, chunk, mean)
                    assert rust_pages == python_pages, (pair, chunk, mean)
                for kv_pages in (0, 1_000, 50_000):
                    assert rust.checkpoint_slots(
                        model_name, device_name, chunk, kv_pages
                    ) == budgets.checkpoint_slots(model, device, chunk, kv_pages)
            assert _outcome(rust.arena_pages, model_name, device_name) == _outcome(
                budgets.arena_pages, model, device)
            for readers in (1.0, 2.0, 16.0, 400.0):
                for rows in (1.0, 30.0, 900.0):
                    for node in (0.0, 120.0, 30_000.0):
                        assert rust.cheaper_attention_path(
                            model_name, device_name, readers, rows, node
                        ) == budgets.cheaper_attention_path(
                            model, device, readers=readers, reader_rows=rows,
                            node_tokens=node)
    for gpus in (1, 2, 8):
        for documents in (0.0, 1.0, 3.5, 100.0):
            assert native.CostModel.parallel_copies(gpus, documents) == (
                budgets.parallel_copies(gpus, documents))


def close(rust_values, python_values) -> bool:
    """Whether two curves agree to 1e-12 relative.

    numpy raises an array to a power with its own vectorized code on
    CPUs with AVX-512, which can differ from the C library's pow (and
    from Python's own ``**``) in the last bit, so a curve built with
    ``keep ** np.arange(...)`` differs between machines too.
    """
    return np.allclose(rust_values, python_values, rtol=1e-12, atol=0.0)


def test_the_answer_survival_curves_are_the_same():
    from quail.common import answers

    generator = np.random.default_rng(1)
    for expected in (0.5, 1.0, 4.0, 37.5, 300.0):
        for limit in (1, 8, 64, 512):
            assert close(native.CostModel.prior_survival(expected, limit),
                         answers.prior_survival(expected, limit))
            lengths = generator.integers(-1, limit + 1, 40).tolist()
            written = generator.integers(0, limit + 1, 40).tolist()
            survival = answers.answer_survival(lengths, written, expected, limit)
            assert close(native.CostModel.answer_survival(
                lengths, written, expected, limit), survival)
            assert native.CostModel.remaining_tokens(survival.tolist()) == (
                answers.remaining_tokens(lengths, written, expected, limit,
                                         survival=survival).tolist())


def test_the_draft_lengths_are_the_same():
    from quail.common import draft_policy

    generator = np.random.default_rng(2)
    for streams in (1, 2, 4, 7, 15, 80, 333, 900, 5000):
        assert native.CostModel.live_bucket(streams) == int(
            draft_policy.live_buckets(streams))
    trials = generator.integers(0, 50, len(draft_policy.LIVE_BUCKETS)).tolist()
    accepts = [int(generator.integers(0, count + 1)) for count in trials]
    for streams in (1, 3, 60, 1500):
        assert native.CostModel.source_odds(streams, trials, accepts) == float(
            draft_policy.source_odds([streams], trials, accepts)[0])
    for cap, limit, expected in ((1, 16, 4.0), (8, 64, 12.0), (32, 256, 90.0)):
        policy = draft_policy.DraftPolicy(cap, limit, expected)
        curve = policy.curve()
        assert close(native.CostModel.draft_policy_curve(cap, limit, expected), curve)
        written = generator.integers(0, limit, 30).tolist()
        streams = generator.integers(1, 2000, 30).tolist()
        rho = (generator.random(30) * 500).tolist()
        assert native.CostModel.draft_lengths(
            cap, limit, curve.tolist(), written, streams, rho, trials, accepts
        ) == draft_policy.draft_lengths(
            cap, limit, curve, written, streams, rho, np.array(trials),
            np.array(accepts)).tolist()


def test_the_retention_and_decode_prices_are_the_same(rust):
    from quail.common.decode import DecodeCost
    from quail.common.retention import (
        RetentionPolicy,
        coefficients,
        retention_pages,
    )

    for model_name, model in MODELS.items():
        for device_name, device in DEVICES.items():
            for peak in (False, True):
                assert rust.coefficients(model_name, device_name, peak) == (
                    coefficients(model, device, peak=peak))
            policy = RetentionPolicy(**coefficients(model, device))
            for tokens in (1, 37, 4_096, 70_000):
                assert rust.prefill_seconds(model_name, device_name, tokens) == (
                    policy.prefill_seconds(tokens))
            cost = DecodeCost.from_specs(model, device)
            assert rust.decode_cost(model_name, device_name) == (
                cost.weight_seconds, cost.kv_seconds, cost.sliding_kv_seconds,
                cost.state_seconds)
            for work in _workloads():
                for passes in (1, 7.5, 300):
                    assert rust.decode_lower_bound_seconds(
                        model_name, device_name, plain(asdict(work)), passes
                    ) == cost.lower_bound_seconds(work, passes)
    for arena, chunk, page in ((0, 1, 16), (100_000, 2_048, 16), (5_000, 4_096, 32)):
        assert native.CostModel.retention_pages(arena, chunk, page) == (
            retention_pages(arena, chunk, page))
