"""Hooks the server tests give the executor instead of a GPU model.

The child process executor imports this module by name, so everything
here must be importable without pytest fixtures.
"""

import threading
import time

import pyarrow as pa
from test_session import fake_tok, make_executor

from quail.server.executor import Hooks

TRUTH = {"r": {"q1:": [1, 1, 0, 1, 1, 0],
               "q2:": [1, 0, 1, 1, 0, 1]}}

FILTER_SQL = """
    SELECT r.id FROM reviews r
    WHERE AI_FILTER(PROMPT('q1: {0}', r.review), {'selectivity': 0.5})
      AND AI_FILTER(PROMPT('q2: {0}', r.review), {'selectivity': 0.5})
"""

JOIN_SQL = """
    SELECT r.id, p.asin FROM reviews r JOIN products p
      ON AI_FILTER(PROMPT('m {0} {1}', r.review, p.description),
                   {'anchor': 'r'})
"""

# answer for the pair (review i, product j); r0 matches p0 and p2
JOIN_TRUTH = {("r", "p"): lambda a, b: (a + b) % 2 == 0}

CONFIG = {"model": "qwen3-4b-fp8", "device": "h100-sxm",
          "gpus": 1, "backend": "quail"}


def reviews_table() -> pa.Table:
    return pa.table({
        "id": [f"r{i}" for i in range(6)],
        "review": [f"review {i} " + "pad " * 20 for i in range(6)],
    })


def products_table() -> pa.Table:
    return pa.table({
        "asin": [f"p{i}" for i in range(4)],
        "description": [f"product {i}" for i in range(4)],
    })


def _executor_with_progress(request):
    from quail.progress import Progress

    progress = Progress("filter (2 stages)", total=6, every=0)
    progress.update(6)
    progress.finish("filter done")
    return make_executor(TRUTH, JOIN_TRUTH)(request)


hooks = Hooks(physical_executor=_executor_with_progress, tokenizer=fake_tok)


def _sleeping_executor(request):
    time.sleep(120)
    return make_executor(TRUTH)(request)


sleeping_hooks = Hooks(physical_executor=_sleeping_executor,
                       tokenizer=fake_tok)


def _failing_executor(request):
    raise RuntimeError("the model refused to start")


failing_hooks = Hooks(physical_executor=_failing_executor, tokenizer=fake_tok)


def start_server(app):
    """Serve ``app`` with uvicorn on a free port; return (url, stop)."""
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                            timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]

    def stop():
        server.should_exit = True
        thread.join(30)

    return f"http://127.0.0.1:{port}", stop
