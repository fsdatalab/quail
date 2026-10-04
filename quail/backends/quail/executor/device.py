"""The device implementation: what shared executor code asks of the device.

A device implementation is the code that runs the model on one kind of
device. There are two: CUDA and MLX. The packer, the stage scheduler,
the operators, and the graph executor reach the device only through
the methods of DeviceImplementation, so they import neither torch nor
MLX. The KV pools, the forward pass, and the readouts of a device
implementation are its own classes.

The base class keeps every array on the host as numpy. It runs the
shared code with no device, as the tests do.
"""

import time
from contextlib import nullcontext

import numpy as np


class DeviceImplementation:
    """Host arrays, host clocks, and no readouts.

    Attributes:
        name: The key of this implementation's forward passes in
            executor.models.PIPELINES.
    """

    name = "host"

    # ---- chunk inputs

    def input_staging(self):
        """Return reusable transfer buffers for chunk inputs, or None."""
        return None

    def stage(self, values, dtype, name=None, staging=None):
        """Copy a host array to the device.

        Args:
            values: A numpy array or a list.
            dtype: The numpy dtype of the device array.
            name: Names the array's reusable buffer in staging; None
                copies it without one.
            staging: The buffers input_staging returned, or None.

        Returns:
            The device array, shaped as values.
        """
        return np.asarray(values, dtype=dtype)

    def stage_tokens(self, ids, staging=None):
        """Copy a chunk's token ids, an int64 numpy array, to the device."""
        return ids

    def select_rows(self, rows, index):
        """Return the rows at the index's positions.

        Args:
            rows: A device array, or the list a test model answers with.
            index: Row positions, as a host array or a staged one.
        """
        if isinstance(rows, list):
            return [rows[position] for position in index]
        return rows[np.asarray(index)]

    # ---- timing and memory

    def time_chunks(self, enabled: bool) -> None:
        """Say whether the query reads the time between its chunks' events.

        An implementation whose events cost something records exact
        ones only while this is on.
        """

    def record_event(self):
        """Mark this point in the device's work; returns the mark."""
        return time.perf_counter()

    def elapsed_ms(self, start, end) -> float:
        """Return the milliseconds of device work between two marks."""
        return (end - start) * 1000.0

    def synchronize(self) -> None:
        """Wait for the device to finish the work handed to it."""

    def inference_mode(self):
        """Return a context manager for running the model without gradients."""
        return nullcontext()

    def peak_memory_bytes(self) -> int:
        """Return the most device memory held at once."""
        return 0

    # ---- readouts an operator builds for its own stages

    def label_readout(self, model, targets, *, rows: int, normalize: bool):
        """Return the AI.CLASSIFY readout over the model's whole output head.

        Args:
            model: The loaded model.
            targets: Token ids whose log probabilities are read.
            rows: Rows each answer reads.
            normalize: Whether the targets' logits are normalized over
                the whole vocabulary.
        """
        raise NotImplementedError(f"{self.name} has no label readout")

    def decision_choices(self, head, offsets):
        """Return the AI.CLASSIFY readout of a decision head.

        Args:
            head: The model's decision head.
            offsets: Distances before the last row, options first,
                ending in 0.
        """
        raise NotImplementedError(f"{self.name} has no decision readout")

    def scores(self, answer_rows):
        """Return the AI.SCORE readout over the retained answer rows."""
        raise NotImplementedError(f"{self.name} has no score readout")
