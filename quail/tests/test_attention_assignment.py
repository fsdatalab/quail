"""The workload-to-path assignment is importable without torch and
names the two attention modes."""

from quail.executor.attention import FILTER_ATTENTION, JOIN_ATTENTION

MODES = ("merge_quant", "unified")


def test_assignment_names_real_modes():
    assert FILTER_ATTENTION in MODES
    assert JOIN_ATTENTION in MODES


def test_join_assignment_uses_measured_faster_path():
    # Packed unified joins are correct, but the 10 x 256 confirming
    # run measured merge_quant 7.4% faster.
    assert JOIN_ATTENTION == "merge_quant"
