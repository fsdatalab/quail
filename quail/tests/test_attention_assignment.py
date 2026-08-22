"""The workload-to-path assignment: importable without torch, names
real modes, and never puts a join on the unified path. The engine
ships exactly these two modes; the retired split path lives in
ablations/split_reference.py."""

from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION

MODES = ("merge_quant", "unified")


def test_assignment_names_real_modes():
    assert FILTER_ATTENTION in MODES
    assert JOIN_ATTENTION in MODES


def test_joins_never_unified():
    # One causal call per pair cannot share an anchor's KV across the
    # many partner suffixes of a chunk: a later pair's tokens would
    # read the earlier pair's scattered KV. The join path must be a
    # two-call mode.
    assert JOIN_ATTENTION != "unified"
