"""The Rust cost model (quail_planner::cost) reads the specs as Python does."""


import numpy as np
import pyarrow as pa
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


LABEL_SETS = (
    [[11], [12], [13]],
    [[5, 6, 7], [5, 6, 8], [5, 9], [4]],
    [[5, 6], [5, 6, 7, 8], [9, 1, 2]],
)


def test_every_spec_file_is_read(rust):
    assert sorted(rust.model_names()) == sorted(MODELS)
    assert sorted(rust.device_names()) == sorted(DEVICES)


def test_the_rust_sums_label_nodes_and_length_summaries_match_pythons():
    from quail import labels as python_labels
    from quail.common.tokens import TokenLengths

    generator = np.random.default_rng(0)
    for length in (0, 1, 7, 8, 9, 127, 128, 129, 1000, 4097):
        values = generator.random(length) * 10 ** generator.integers(0, 12, length)
        assert native.CostModel.numpy_sum(values.tolist()) == float(np.sum(values))

    for label_ids in LABEL_SETS:
        for starts in ((), tuple((True,) + (False,) * (len(ids) - 1)
                                 for ids in label_ids)):
            read = python_labels.read_nodes(label_ids, starts)
            found = native.CostModel.label_nodes(
                label_ids, [list(flags) for flags in starts])
            assert plain(found[:2]) == plain((
                sorted(read), sorted(python_labels.fed_nodes(read))))

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
