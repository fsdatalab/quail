"""The Rust cost model (crates/quail-cost) reads the specs and labels as Python does."""


import numpy as np
import pytest

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
        found = plain(rust.model_properties(name))
        assert {key: found[key] for key in PROPERTIES} == expected, name


def test_every_device_has_the_same_gemm_rates(rust):
    for name, device in DEVICES.items():
        for precision in ("fp8", "bf16", "fp32"):
            assert rust.gemm_flops(name, precision) == device.gemm_flops(precision)


def test_the_array_sum_rounds_as_numpy_does():
    generator = np.random.default_rng(0)
    for length in (0, 1, 7, 8, 9, 127, 128, 129, 1000, 4097):
        values = generator.random(length) * 10 ** generator.integers(0, 12, length)
        assert native.CostModel.numpy_sum(values.tolist()) == float(np.sum(values))


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


def test_a_length_column_summarizes_to_its_sums_and_short_counts():
    import pyarrow as pa

    from quail.common.tokens import TokenLengths

    rng = np.random.default_rng(3)
    columns = (
        pa.chunked_array([], pa.int64()),
        pa.chunked_array([[7]]),
        # a token store yields one chunk per record batch
        pa.chunked_array([rng.integers(1, 3_000, 500), rng.integers(1, 90_000, 700)]),
    )
    for column in columns:
        values = np.asarray(TokenLengths(column), dtype=np.int64)
        for window in (0, 512, 4_096):
            short, counts = np.unique(values[values < window], return_counts=True)
            expected = dict(
                count=len(values), total=int(values.sum()),
                squared=sum(int(value) ** 2 for value in values),
                maximum=int(values.max(initial=0)), window=window,
                short_counts=[list(pair) for pair in zip(short.tolist(),
                                                         counts.tolist())])
            fields = native.CostModel.summarize_alias(values, window)
            assert plain(fields) == expected, (len(values), window)
