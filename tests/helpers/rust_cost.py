"""The Rust cost model (crates/quail-planner/src/cost/) as tests call it."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import _quail_native as native

from quail.specs.loading import DATA


def cost_model(directory: Path | None = None, **variants) -> "native.CostModel":
    """Return a Rust cost model over Quail's specs.

    Args:
        directory: An empty directory to copy the specs into; needed
            only with variants.
        **variants: New model specs by name, each a dict holding its
            `base` model and the fields it changes.

    Returns:
        A `CostModel` that reads every model and device, and the variants.
    """
    if not variants:
        return native.CostModel(str(DATA))
    data = directory / "data"
    shutil.copytree(DATA, data)
    for name, fields in variants.items():
        lines = [f"{key} = {json.dumps(value)}"
                 for key, value in {"name": name, **fields}.items()]
        (data / "models" / f"{name}.toml").write_text("\n".join(lines) + "\n")
    return native.CostModel(str(data))


@dataclass(frozen=True)
class Cost:
    """One estimate from the Rust cost model.

    Attributes:
        seconds: Roofline time of the expected work.
        passes: Expected fresh tokens divided by the chunk.
        work: The expected work, as a dict of `Work` fields.
        suffix_tokens: Expected tokens fed after the documents.
        rounds: Decode rounds of a classification, or the readout rows
            of an extraction.
    """

    seconds: float
    passes: float
    work: dict
    suffix_tokens: float
    rounds: float


def classify_estimate(rust, scoring, expected_rows, head_tokens, frame_tokens,
                      labels, *, lengths, shared=(), chunk, model, device,
                      resident=False, draws=1, gpus=1) -> Cost:
    """Return the Rust classification estimate, called by keyword."""
    return Cost(*rust.classify_estimate(
        scoring, expected_rows, head_tokens, frame_tokens,
        [list(label) for label in labels], list(lengths), list(shared), chunk,
        model, device, resident, draws, [], gpus))
