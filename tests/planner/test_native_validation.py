"""Malformed public native inputs fail normally, without panics or recursion."""

import copy
import json
import subprocess
import sys

import pytest
from helpers.rust_cost import cost_model

from quail.specs import D1_2B_BF16, H100_SXM

native = pytest.importorskip("_quail_native")


def _graph():
    return {
        "nodes": [{
            "type": "quail.scan", "id": "scan:r", "inputs": [],
            "attributes": {
                "alias": "r", "input_id": "r", "num_documents": 1,
                "total_tokens": 1, "predicates": [], "expected_rows": None,
            },
        }],
        "root": {"type": "quail.physical.base.PortRef",
                 "node_id": "scan:r", "port": "rows:r"},
    }


INVALID_GRAPHS = (
    ("cycle", "cycle"), ("dangling", "missing output"),
    ("duplicate", "duplicate physical node id"),
)


def test_stage_cut_rejects_each_invalid_graph_and_accepts_a_valid_one():
    assert json.loads(native.region_stages(json.dumps(_graph()), 2)) == []
    # Keep a regression to stack overflow isolated from the pytest process.
    script_text = """
import resource
import sys
import _quail_native
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
try:
    _quail_native.region_stages(sys.argv[1], 2)
except ValueError as error:
    assert sys.argv[2] in str(error), str(error)
else:
    raise AssertionError('invalid graph was accepted')
"""
    for kind, message in INVALID_GRAPHS:
        graph = _graph()
        if kind == "duplicate":
            graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))
        else:
            source = dict(graph["root"])
            if kind == "dangling":
                source["node_id"] = "missing"
            graph["nodes"][0]["inputs"] = [{
                "type": "quail.physical.base.InputPort", "name": "rows:r",
                "value_type": "rows", "source": source, "schema": [],
            }]
        result = subprocess.run(
            [sys.executable, "-c", script_text, json.dumps(graph), message],
            capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, (kind, result.stderr)


INVALID_PASS_GROUP_INPUTS = [
    ({"selectivities": []}, "one selectivity"),
    ({"selectivities": [1.0, 1.0, 1.0]}, "one selectivity"),
    *[({field: value}, "finite and nonnegative")
      for field in ("rows", "state_tokens")
      for value in (-1.0, float("nan"), float("inf"), -float("inf"))],
    *[({"selectivities": [1.0, value]}, "selectivities must be finite")
      for value in (-0.1, 1.1, float("nan"), float("inf"))],
    ({"chunk_tokens": 0}, "must be positive"),
    ({"chunk_tokens": -1}, "must be positive"),
    ({"set_size": 0}, "must be positive"),
    ({"question_tokens": [-1, 10]}, "must be nonnegative"),
    ({"longest_state": -1}, "must be nonnegative"),
    ({"header": -1}, "must be nonnegative"),
    ({"rows": 1e308, "state_tokens": 1e308}, "cost must be finite"),
    ({"question_tokens": [2**63 - 1, 10]}, "does not fit"),
    ({"longest_state": 2**63 - 1}, "does not fit"),
]


def test_pass_groups_reject_each_invalid_input():
    for overrides, message in INVALID_PASS_GROUP_INPUTS:
        arguments = dict(
            question_tokens=[10, 10], selectivities=[1.0, 1.0], rows=100.0,
            state_tokens=20.0, longest_state=20, model=D1_2B_BF16.name,
            device=H100_SXM.name, chunk_tokens=100, set_size=1, header=0,
        )
        arguments.update(overrides)
        with pytest.raises(ValueError, match=message):
            cost_model().choose_pass_groups(**arguments)
            pytest.fail(f"{overrides} was accepted")


def test_pass_groups_accept_no_questions_and_a_filter_that_keeps_no_row():
    for question_tokens, selectivities, expected in (
            ([], [], []), ([10, 10], [0.0, 1.0], [0, 1])):
        assert cost_model().choose_pass_groups(
            question_tokens, selectivities, 0.0, 0.0, 0,
            D1_2B_BF16.name, H100_SXM.name, 100, 1, 0,
        ) == expected, question_tokens
