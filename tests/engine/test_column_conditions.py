"""Settled column values used by conditions inside a pipeline."""

from types import SimpleNamespace

import numpy as np
import pytest
from helpers.kv_checker import cpu_torch

from quail.backends.quail.executor.operators.filter import ConditionGate
from quail.backends.quail.executor.operators.sae_features import SaeFeaturesPart
from quail.backends.quail.executor.sae import feature_lists_of_rows
from quail.backends.quail.executor.stages import Stage
from quail.logical import ColumnRef, RegularPredicate
from quail.physical import AiSaeFeatures, Filter, SaeFeaturesSpec


@pytest.mark.parametrize("pooling", ["last", "max"])
def test_feature_lists_are_tested_after_settling_and_cached_as_values(pooling):
    packed = np.asarray([[3, 7, 0.5, 0], [3, 7, 0, 0], [3, 7, 0.5, 0]])
    node = AiSaeFeatures(node_id="features", spec=SaeFeaturesSpec(
        aliases=("r",), name="features", pooling=pooling,
        query_template="{0}?", arguments=(("r", "body"),),
        expected_rows=3, estimated_seconds=0, prompt_token_parts=((), (1,))))
    state = SimpleNamespace(torch=cpu_torch(), loaded_model=SimpleNamespace(
        sae=SimpleNamespace(keep=2)))

    def harvest(handles, *, completed_only):
        return (np.arange(len(handles)), *feature_lists_of_rows(packed[handles], 2))

    accumulator = (SimpleNamespace(settle=lambda key: key[1], harvest=harvest)
                   if pooling == "max" else None)
    part = SaeFeaturesPart(state, node, [0, 1, 2], {"r": [[0], [1], [2]]},
                           accumulator=accumulator)
    for position in range(3):
        part.stages[0].decide(position, packed[position:position + 1])
    condition = Filter(node_id="condition", aliases=("r",), predicate=RegularPredicate(
        ColumnRef("r", "reviews", "features"), "is not null"))
    gate = ConditionGate(condition, part, part)
    assert part.tested_columns == ("features",)
    assert gate.gate(("r", 1)) is None
    if pooling == "max":
        assert set(part.pending) == {0, 2}
    assert gate.gate(("r", 0)) is None
    assert gate.gate(("r", 2)) is None
    assert len(gate.accepted) == 2
    rejected = ConditionGate(Filter(
        node_id="null", aliases=("r",), predicate=RegularPredicate(
            ColumnRef("r", "reviews", "features"), "is null")), part, part)
    assert rejected.gate(("r", 1)) is Stage.DROP
    rows = part.result(0, 0, 0, {}).outputs["rows:r"]
    expected = [[{"feature": 3, "strength": 0.5}], [],
                [{"feature": 3, "strength": 0.5}]]
    values = dict(zip(rows.ids["r"].to_pylist(), rows.column("features").to_pylist()))
    assert values == dict(enumerate(expected))
