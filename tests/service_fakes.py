"""Hooks the service tests give the executor instead of a GPU model.

The child process executor imports this module by name, so everything
here must be importable without pytest fixtures.
"""

import time

import pyarrow as pa
from test_session import fake_tok, make_executor

from quail.service.executor import Hooks

TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
               "q2:": [1, 0, 1, 1, 0, 1]}}

FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""

CONFIG = {"model": "qwen3-4b-fp8", "device": "h100-sxm",
          "gpus": 1, "backend": "quail"}


def reviews_table() -> pa.Table:
    return pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
    })


def _executor_with_progress(request):
    from quail.progress import Progress

    progress = Progress("filter (2 stages)", total=6, every=0)
    progress.update(6)
    progress.finish("filter done")
    return make_executor(TRUTH)(request)


hooks = Hooks(physical_executor=_executor_with_progress, tokenizer=fake_tok)


def _sleeping_executor(request):
    time.sleep(120)
    return make_executor(TRUTH)(request)


sleeping_hooks = Hooks(physical_executor=_sleeping_executor,
                       tokenizer=fake_tok)


def _failing_executor(request):
    raise RuntimeError("the model refused to start")


failing_hooks = Hooks(physical_executor=_failing_executor, tokenizer=fake_tok)
