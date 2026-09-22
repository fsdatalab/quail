"""The compaction demo exposes only local execution."""

import inspect

from demos import agent_trace_compaction as demo


def test_evaluate_has_no_remote_endpoint():
    assert list(inspect.signature(demo.evaluate).parameters) == [
        "directory",
        "gpus",
    ]
