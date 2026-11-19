"""Opt-in record of each TRUE/FALSE answer's margin and answer-row state.

With QUAIL_ANSWER_TRACE set to a directory, every AI filter and AI join
answer read through AsyncAnswers is saved with its TRUE minus FALSE
logit and the final-normed hidden state of its answer row. Each region
writes one Parquet file of rows and one .npy file of states, as a
teacher's labels for training a smaller model.
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ANSWER_TRACE_ENV = "QUAIL_ANSWER_TRACE"

# file names within one process
_FILE_NUMBERS = itertools.count()


def answer_trace_dir() -> Path | None:
    """Return the trace directory, or None when tracing is off."""
    found = os.environ.get(ANSWER_TRACE_ENV)
    return Path(found) if found else None


class AnswerTrace:
    """The traced answers of one region, written when closed.

    A row is one document of a filter, or one (document, partner) pair
    of a join. Its ``state_row`` is its row in the .npy file of states.

    Args:
        directory: Where close() writes ``<name>.parquet`` and
            ``<name>.npy``.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.columns = {"node": [], "stage": [], "document": [], "partner": [],
                        "margin": []}
        self.states = []

    def add(self, node, stage, documents, partners, margins, states):
        """Record one group of answers.

        Args:
            node: The operator's node id.
            stage: The answer's stage within the operator.
            documents: Per row, the document id.
            partners: Per row, the partner id, or -1 without one.
            margins: Per row, the TRUE minus FALSE logit.
            states: (rows, hidden) final-normed answer-row states.
        """
        num_rows = len(margins)
        self.columns["node"].extend([node] * num_rows)
        self.columns["stage"].extend([stage] * num_rows)
        self.columns["document"].extend(int(document) for document in documents)
        self.columns["partner"].extend(int(partner) for partner in partners)
        self.columns["margin"].extend(float(margin) for margin in margins)
        self.states.append(np.array(states, copy=True))

    def close(self):
        """Write the recorded answers; nothing when there are none."""
        if not self.states:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"{os.getpid()}-{next(_FILE_NUMBERS)}"
        states = np.concatenate(self.states)
        np.save(self.directory / f"{name}.npy", states)
        table = pa.table({
            "node": pa.array(self.columns["node"], pa.string()),
            "stage": pa.array(self.columns["stage"], pa.int32()),
            "document": pa.array(self.columns["document"], pa.int64()),
            "partner": pa.array(self.columns["partner"], pa.int64()),
            "margin": pa.array(self.columns["margin"], pa.float32()),
            "state_row": pa.array(np.arange(len(states)), pa.int64()),
        })
        pq.write_table(table, self.directory / f"{name}.parquet")
        self.states = []
