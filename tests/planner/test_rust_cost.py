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

native = pytest.importorskip("_quail_native")


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


def same_numbers(rust_value, python_value) -> bool:
    """Whether two results agree: floats to 1e-12 relative, the rest exactly.

    Python's sum() compensates its rounding only while every item is an
    exact float; once a numpy scalar appears, as it does where work.py
    builds a count with numpy, it adds plainly. Which values are numpy
    scalars is an accident of Python's types, so an operator's cost can
    differ from the Rust port's in the last bit.
    """
    rust_value, python_value = plain(rust_value), plain(python_value)
    if isinstance(python_value, float) or isinstance(rust_value, float):
        return bool(np.isclose(rust_value, python_value, rtol=1e-12, atol=0.0))
    if isinstance(python_value, dict):
        return (rust_value.keys() == python_value.keys() and all(
            same_numbers(rust_value[key], python_value[key]) for key in python_value))
    if isinstance(python_value, list):
        return len(rust_value) == len(python_value) and all(
            same_numbers(a, b) for a, b in zip(rust_value, python_value))
    return rust_value == python_value


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


def _stats_cases():
    from quail.planner.statistics.tables import AliasStats

    yield AliasStats(count=0, total=0, squared=0, maximum=0)
    yield AliasStats(count=4, total=1_000, squared=400_000, maximum=600)
    yield AliasStats(count=1_000_000, total=30_000_000_000,
                     squared=10**22 + 7, maximum=90_000)
    yield AliasStats(count=50, total=40_000, squared=40_000_000, maximum=4_000,
                     window=1_024, short_counts=((100, 20), (700, 9), (1_023, 1)))


def test_the_table_statistics_are_the_same():
    from dataclasses import asdict

    from quail.planner.statistics.tables import scale_alias

    for stats in _stats_cases():
        for offset, fraction in ((0.0, 1.0), (12.5, 0.37), (900.0, 0.5)):
            scaled = scale_alias(stats, fraction)
            window = ((stats.window_pairs(offset), stats.window_reads(offset))
                      if stats.window else (0.0, 0.0))
            assert plain(native.CostModel.alias_stats_values(
                plain(asdict(stats)), offset, fraction)) == plain((
                stats.mean, stats.variance, *window,
                (scaled.count, scaled.total, scaled.squared, scaled.short_counts)))


def test_the_filter_score_and_feature_costs_are_the_same(rust):
    from quail.common.budgets import chunk_budget
    from quail.cost.filters import filter_cost
    from quail.cost.packing import choose_pass_groups
    from quail.cost.sae_features import SaeFeaturesCost
    from quail.cost.score import ScoreCost

    stats = list(_stats_cases())[1:]
    for model_name, model in MODELS.items():
        for device_name, device in DEVICES.items():
            chunk = max(1, chunk_budget(model, device))
            for question, mean in ((12, 300.0), (90, 4_500.5)):
                for first in (True, False):
                    assert same_numbers(rust.filter_cost(
                        question, mean, model_name, device_name, chunk, first),
                        filter_cost(question, mean, model, device, chunk, first=first))
            for summaries in ((stats[0],), (stats[0], stats[2]), (stats[2],)):
                for capacity in (500, 10**9):
                    for gpus, draws in ((1, 1), (4, 3)):
                        cost = ScoreCost((40, 9, 15)[:len(summaries) + 1],
                                         summaries, 777.0, model, device, chunk,
                                         capacity, gpus, draws)
                        for rows in (0.0, 3.0, 2_500.0):
                            for prefix_groups in (None, 2.0):
                                work, seconds = cost.estimate(
                                    rows, prefix_groups=prefix_groups)
                                assert same_numbers(rust.score_estimate(
                                    list(cost.token_lengths),
                                    [plain(asdict(s)) for s in summaries],
                                    cost.mean_tokens, model_name, device_name,
                                    chunk, capacity, gpus, draws, rows,
                                    prefix_groups), (asdict(work), seconds))
            for pooling, max_pooling in (("last", False), ("max", False),
                                         ("last", True)):
                cost = SaeFeaturesCost((40, 12), stats[2], pooling, model, device,
                                       chunk, 2, max_pooling)
                for rows in (0.0, 30.0):
                    for resident in (False, True):
                        work, seconds = cost.estimate(rows, resident)
                        assert same_numbers(rust.sae_features_estimate(
                            [40, 12], plain(asdict(stats[2])), pooling, model_name,
                            device_name, chunk, 2, max_pooling, rows, resident
                        ), (asdict(work), seconds))
    model = MODELS["d1-2b-bf16"]
    device = DEVICES["h100-sxm"]
    for chunk in (40, 4_096):
        for header in (0, 3):
            arguments = ((12, 30, 7, 9), (0.5, 0.9, 0.2, 1.0), 1_000.0, 300.0, 600)
            assert _outcome(rust.choose_pass_groups, *arguments, model.name,
                            device.name, chunk, 3, header) == _outcome(
                lambda *values: choose_pass_groups(
                    *values, model, device, chunk, set_size=3, header=header),
                *arguments)


# label token ids: one token each; shared prefixes; a label that ends
# where another continues
LABEL_SETS = (
    [[11], [12], [13]],
    [[5, 6, 7], [5, 6, 8], [5, 9], [4]],
    [[5, 6], [5, 6, 7, 8], [9, 1, 2]],
)


def test_the_label_trie_reads_the_same_nodes():
    from quail import labels as python_labels

    for label_ids in LABEL_SETS:
        for starts in ((), tuple((True,) + (False,) * (len(ids) - 1)
                                 for ids in label_ids)):
            read = python_labels.read_nodes(label_ids, starts)
            assert plain(native.CostModel.label_nodes(
                label_ids, [list(flags) for flags in starts])) == plain((
                sorted(read), sorted(python_labels.fed_nodes(read)),
                python_labels.decode_round_tokens(label_ids)))


# whether the documents' KV is resident, and their borrowed prefixes
RESIDENCY = ((False, []), (False, [0.0, 900.0, 0.0, 30.0]), (True, []))


def test_the_classify_costs_are_the_same(rust):
    from quail.common.budgets import chunk_budget
    from quail.cost import classify
    from quail.labels import DECODE_SCORING, LETTERS_SCORING, TREE_SCORING

    lengths = [120.0, 3_000.0, 45.0, 800.0]
    for model_name, model in MODELS.items():
        for device_name, device in DEVICES.items():
            chunk = max(1, chunk_budget(model, device))
            for scoring in (LETTERS_SCORING, TREE_SCORING, DECODE_SCORING):
                for label_ids in LABEL_SETS:
                    for resident, shared in RESIDENCY:
                        for rows, gpus in ((4.0, 1), (1_000.0, 2)):
                            expected = classify.estimate(
                                scoring, rows, 30, 12, label_ids, lengths=lengths,
                                shared=shared, chunk=chunk, model=model,
                                device=device, resident=resident, draws=2, gpus=gpus)
                            assert same_numbers(rust.classify_estimate(
                                scoring, rows, 30, 12, label_ids, lengths, shared,
                                chunk, model_name, device_name, resident, 2, [], gpus,
                            ), (expected.seconds, expected.passes,
                                asdict(expected.work), expected.suffix_tokens,
                                expected.rounds)), (model_name, scoring, label_ids)


def test_the_join_work_is_the_same():
    from quail.cost.joins import PricedJoin, packed_join_work, stage_work
    from quail.planner.statistics.tables import AliasStats

    for window in (0, 1_024):
        short = ((100, 20), (700, 9)) if window else ()
        lengths = {
            "r": AliasStats(50, 40_000, 40_000_000, 4_000, window, short),
            "a": AliasStats(8, 300, 12_000, 60, window, ((30, 8),) if window else ()),
            "c": AliasStats(1_000_000, 30_000_000_000, 10**22 + 7, 90_000, window,
                            short),
        }
        for aliases in (("r", "a"), ("r", "a", "c")):
            spec = PricedJoin(
                written_position=0, aliases=aliases, anchor=None, anchor_free=True,
                semantics="full", selectivity=0.3,
                frame_tokens={alias: 7 + index for index, alias in enumerate(aliases)},
                label_tokens={alias: 3 + index for index, alias in enumerate(aliases)},
                tail_tokens=11)
            fields = {alias: plain(asdict(stats)) for alias, stats in lengths.items()}
            arguments = (list(aliases), dict(spec.frame_tokens),
                         dict(spec.label_tokens), spec.tail_tokens)
            for anchor in aliases:
                for n, tuples in ((0.0, 0.0), (4.0, 30.0), (40.0, 2_000.5)):
                    for resident in (False, True):
                        expected = stage_work(spec, anchor, n, tuples, lengths, 25,
                                              resident=resident, window=window)
                        assert same_numbers(native.CostModel.join_work(
                            *arguments, anchor, n, tuples, fields, 25, resident,
                            window), asdict(expected)), (aliases, anchor, n, resident)
                    for set_size in (1, 4):
                        expected = packed_join_work(spec, anchor, n, tuples, lengths,
                                                    25, window=window,
                                                    set_size=set_size)
                        assert same_numbers(native.CostModel.join_work(
                            *arguments, anchor, n, tuples, fields, 25, False, window,
                            set_size), asdict(expected))


def test_the_extract_costs_are_the_same(rust):
    from quail.common.budgets import chunk_budget
    from quail.cost import extract

    rng = np.random.default_rng(7)
    tables = {
        "short": ([120.0, 3_000.0, 45.0, 800.0], [0.0, 20.0, 0.0, 40.0]),
        "sampled": (rng.integers(1, 9_000, 2_000).astype(float).tolist(),
                    rng.integers(0, 60, 2_000).astype(float).tolist()),
    }
    windows = {MODELS[name].sliding_window > 0 for name in MODELS}
    recurrent = {bool(MODELS[name].state_layers) for name in MODELS}
    assert windows == recurrent == {False, True}
    device_name = "h100-sxm"
    device = DEVICES[device_name]
    for model_name, model in MODELS.items():
        chunk = max(1, chunk_budget(model, device))
        for table_name, (lengths, shared) in tables.items():
            for policy, draft_tokens in (("fixed", 0), ("fixed", 3), ("adaptive", 6)):
                for answer_tokens, max_answer_tokens in ((16, 40), (1, 8), (0, 8),
                                                         (40, 1)):
                    for resident, shared_lengths in ((False, None), (False, shared),
                                                     (True, None)):
                        assumptions = extract.ExtractAssumptions(answer_tokens)
                        expected = extract.estimate(
                            1_500.0, lengths, 30, 12, chunk=chunk, model=model,
                            device=device, resident=resident, shared=shared_lengths,
                            max_answer_tokens=max_answer_tokens,
                            draft_tokens=draft_tokens, draft_policy=policy,
                            round_cost_per_token=0.002, gpus=2,
                            assumptions=assumptions)
                        assert same_numbers(rust.extract_estimate(
                            1_500.0, lengths, shared_lengths or [], 30, 12, chunk,
                            model_name, device_name, resident, max_answer_tokens,
                            draft_tokens, policy, 0.002, 2, answer_tokens,
                        ), (expected.seconds, expected.passes,
                            asdict(expected.work), expected.suffix_tokens,
                            expected.rows)), (model_name, table_name, policy,
                                              answer_tokens, resident)


def _name(specs, spec) -> str:
    """The registered name of a model or device spec the planner passed."""
    (name,) = [name for name, registered in specs.items() if registered == spec]
    return name


def _replay(rust, kind, call):
    """The Rust value of one recorded cost call, as (rust, python) results."""
    a = call.arguments
    model = _name(MODELS, a["self"].model if "self" in a else a["model"])
    device = _name(DEVICES, a["self"].device if "self" in a else a["device"])
    if kind == "filter_cost":
        return rust.filter_cost(a["question_tokens"], a["mean_tokens"], model, device,
                                a["chunk_tokens"], a["first"])
    if kind == "estimate_suffixes":
        return rust.classify_estimate_suffixes(
            a["head_tokens"], a["frame_tokens"], plain(list(a["suffixes"])),
            a["expected_rows"], plain(list(a["lengths"])), plain(list(a["shared"])),
            a["chunk"], model, device, a["resident"], a["canvas_rows"],
            None if a["runs"] is None else plain(a["runs"]), a["answer_rows"],
            a["head_rows"], a["gpus"])
    if kind == "classify_estimate":
        return rust.classify_estimate(
            a["scoring"], a["expected_rows"], a["head_tokens"], a["frame_tokens"],
            plain(a["labels"]), plain(list(a["lengths"])), plain(list(a["shared"])),
            a["chunk"], model, device, a["resident"], a["draws"], plain(a["starts"]),
            a["gpus"])
    if kind == "extract_estimate":
        assumptions = a["assumptions"]
        return rust.extract_estimate(
            a["expected_rows"], plain(list(a["lengths"])),
            [] if a["shared"] is None else plain(list(a["shared"])), a["head_tokens"],
            a["tail_tokens"], a["chunk"], model, device, a["resident"],
            a["max_answer_tokens"], a["draft_tokens"], a["draft_policy"],
            a["round_cost_per_token"], a["gpus"],
            16 if assumptions is None else assumptions.answer_tokens)
    if kind == "choose_pass_groups":
        return _outcome(rust.choose_pass_groups, plain(a["question_tokens"]),
                        plain(a["selectivities"]), a["rows"], a["state_tokens"],
                        a["longest_state"], model, device, a["chunk_tokens"],
                        a["set_size"], a["header"])
    if kind == "score":
        cost = a["self"]
        return rust.score_estimate(
            list(cost.token_lengths), [plain(asdict(s)) for s in cost.summaries],
            cost.mean_tokens, model, device, cost.chunk_tokens, cost.capacity,
            cost.gpus, cost.draws, a["expected_rows"], a["prefix_groups"])
    if kind == "sae_features":
        cost = a["self"]
        return rust.sae_features_estimate(
            list(cost.token_lengths), plain(asdict(cost.summary)), cost.pooling, model,
            device, cost.chunk_tokens, cost.gpus, cost.max_pooling,
            a["expected_rows"], a["resident"])
    raise AssertionError(kind)


def _join_replay(kind, call):
    """The Rust value of one recorded join work call."""
    a = call.arguments
    spec = a["spec"]
    return native.CostModel.join_work(
        list(spec.aliases), dict(spec.frame_tokens), dict(spec.label_tokens),
        spec.tail_tokens, a["anchor"], a["n"], a["tuples"],
        {alias: plain(asdict(stats)) for alias, stats in a["lengths"].items()},
        a["pre"], a.get("resident", False), a["window"],
        a["set_size"] if kind == "packed_join_work" else None)


def _python_value(kind, result):
    """A recorded Python result in the shape the Rust binding returns."""
    if kind in ("estimate_suffixes", "classify_estimate"):
        return (result.seconds, result.passes, asdict(result.work),
                result.suffix_tokens, result.rounds)
    if kind == "extract_estimate":
        return (result.seconds, result.passes, asdict(result.work),
                result.suffix_tokens, result.rows)
    if kind in ("score", "sae_features"):
        work, seconds = result
        return asdict(work), seconds
    if kind in ("stage_work", "packed_join_work"):
        return asdict(result)
    return plain(result)


def _recorders():
    """(owner, attribute, kind) for each cost function the planner calls."""
    from quail.cost import classify, extract
    from quail.cost.sae_features import SaeFeaturesCost
    from quail.cost.score import ScoreCost
    from quail.planner.graph import passes, pricing
    from quail.planner.operators import filter as filters
    from quail.planner.search import join_order

    return (
        (filters, "filter_cost", "filter_cost"),
        (classify, "estimate_suffixes", "estimate_suffixes"),
        (classify, "estimate", "classify_estimate"),
        (extract, "estimate", "extract_estimate"),
        (pricing, "extract_estimate", "extract_estimate"),
        (passes, "choose_pass_groups", "choose_pass_groups"),
        (join_order, "join_work", "stage_work"),
        (join_order, "packed_join_work", "packed_join_work"),
        (ScoreCost, "estimate", "score"),
        (SaeFeaturesCost, "estimate", "sae_features"),
    )


# a dense model, a sliding-window model, a hybrid model with state layers,
# and a set-attention model
REPLAY_MODELS = ("qwen3-4b-fp8", "diffusion-gemma-26b-a4b-fp8", "qwen3.5-4b-bf16",
                 "d1-2b-bf16")


def test_every_cost_the_planner_asks_for_is_the_same_in_rust(rust, monkeypatch):
    import inspect
    import zlib

    from helpers.sessions import check_table

    import quail
    from quail.planner.plan import EngineConfig
    from quail_b.queries import queries
    from tools.dump_check_queries import check_query_entries

    calls = []
    for owner, attribute, kind in _recorders():
        original = getattr(owner, attribute)
        signature = inspect.signature(original)

        def recorded(*args, _original=original, _signature=signature, _kind=kind,
                     **kwargs):
            result = _original(*args, **kwargs)
            bound = _signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append((_kind, bound, result))
            return result

        monkeypatch.setattr(owner, attribute, recorded)

    def token_ids(text):
        return [zlib.crc32(word.encode()) % 151_936 for word in text.split()]

    entries = check_query_entries(list(queries(include_draft=True)))
    for model in REPLAY_MODELS:
        config = EngineConfig(model=model, device="h100-sxm")
        for entry in entries:
            with quail.Session(config, tokenizer=token_ids) as session:
                for name, columns in entry["tables"].items():
                    session.register(name, quail.DocumentProvider.from_table(
                        check_table(columns), id_col="id"))
                session.sql(entry["sql"]).plan()

    kinds, differ = {}, []
    for kind, call, result in calls:
        kinds[kind] = kinds.get(kind, 0) + 1
        value = (_join_replay(kind, call) if kind in ("stage_work", "packed_join_work")
                 else _replay(rust, kind, call))
        if not same_numbers(value, _python_value(kind, result)):
            differ.append(kind)
    print(kinds)
    assert set(kinds) >= {"filter_cost", "estimate_suffixes", "extract_estimate",
                          "stage_work", "packed_join_work", "choose_pass_groups",
                          "score"}, kinds
    assert differ == [], differ[:10]


def test_a_length_column_summarizes_the_same(rust):
    import pyarrow as pa

    from quail.common.tokens import TokenLengths
    from quail.planner.statistics.tables import summarize_alias

    rng = np.random.default_rng(3)
    columns = (
        pa.chunked_array([], pa.int64()),
        pa.chunked_array([[7]]),
        # a token store yields one chunk per record batch
        pa.chunked_array([rng.integers(1, 3_000, 500), rng.integers(1, 90_000, 700)]),
    )
    for column in columns:
        lengths = TokenLengths(column)
        for window in (0, 512, 4_096):
            expected = summarize_alias(lengths, window)
            fields = native.CostModel.summarize_alias(np.asarray(lengths), window)
            assert plain(fields) == plain(asdict(expected)), (len(lengths), window)
