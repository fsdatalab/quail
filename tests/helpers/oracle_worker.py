"""A GPU worker that answers from an oracle, as its own process.

Quail starts it in place of ``python -m quail.worker``, with a socket
and a GPU index: ``python oracle_worker.py <socket> <gpu index>``. It
keeps every document whose position is even, and labels and joins by
the sum of the documents' positions. ``QUAIL_ORACLE_DELAY`` makes each
filter answer take that many seconds.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quail  # noqa: E402
from helpers.oracle import Oracle, _serve  # noqa: E402
from helpers.sessions import CONFIG, TOKENIZER  # noqa: E402
from quail.api.engine import Engine, engine_description  # noqa: E402


def _keeps(alias, document, position) -> bool:
    """Keep a document at an even position, after the delay a test set."""
    time.sleep(float(os.environ.get("QUAIL_ORACLE_DELAY", "0")))
    return document % 2 == 0


if __name__ == "__main__":
    oracle = Oracle(
        filters=_keeps,
        joins=lambda node, assignment: sum(assignment.values()) % 2 == 0,
        labels=lambda spec, ids: spec.labels[sum(ids) % len(spec.labels)])
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        description = engine_description(Engine(session))
    _serve(oracle, sys.argv[1], description)
