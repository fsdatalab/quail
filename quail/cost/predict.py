"""Predicted query time: speed of light seconds scaled by measured ratios.

A table measured on one model and device gives, for a forward pass of a
given shape, the ratio of measured wall seconds to speed of light
seconds. Wall seconds include everything the pass pays beyond the ideal
work, such as host overhead and the GPU waiting for the CPU. The ratio
depends on three numbers: tokens in the pass, attention pairs per token,
and KV read per token.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from quail.cost.sol import speed_of_light
from quail.cost.work import Work
from quail.specs import DeviceSpec, ModelSpec

TABLE_DIR = Path(__file__).parent / "tables"


@dataclass(frozen=True, eq=False)
class PassTable:
    """Measured wall seconds over speed of light seconds, on a grid of shapes.

    Attributes:
        model: The model name the table was measured on.
        device: The device name the table was measured on.
        tokens: Ascending tokens-per-pass grid points.
        pairs_per_token: Ascending attention pairs per token grid points.
        kv_read_per_token: Ascending KV tokens read per token grid points.
        ratios: Array indexed by tokens, pairs per token, KV read per token.
        source: Where the measurements are saved.
    """

    model: str
    device: str
    tokens: np.ndarray
    pairs_per_token: np.ndarray
    kv_read_per_token: np.ndarray
    ratios: np.ndarray
    source: str

    def __post_init__(self):
        axes = (self.tokens, self.pairs_per_token, self.kv_read_per_token)
        if self.ratios.shape != tuple(len(axis) for axis in axes):
            raise ValueError("ratios must have one entry per grid point")
        if any(len(axis) == 0 or np.any(np.diff(axis) <= 0) for axis in axes):
            raise ValueError("each grid axis must be non-empty and ascending")
        if self.tokens[0] <= 0:
            raise ValueError("tokens grid points must be positive")
        if not np.all(np.isfinite(self.ratios)) or np.any(self.ratios <= 0):
            raise ValueError("ratios must be positive and finite")

    @classmethod
    def from_dict(cls, saved: dict) -> "PassTable":
        """Rebuild a table saved by to_dict."""
        return cls(
            model=saved["model"], device=saved["device"],
            tokens=np.asarray(saved["tokens"], dtype=float),
            pairs_per_token=np.asarray(saved["pairs_per_token"], dtype=float),
            kv_read_per_token=np.asarray(saved["kv_read_per_token"], dtype=float),
            ratios=np.asarray(saved["ratios"], dtype=float),
            source=saved["source"])

    def to_dict(self) -> dict:
        """Return the table as JSON-ready lists."""
        return {"model": self.model, "device": self.device,
                "tokens": self.tokens.tolist(),
                "pairs_per_token": self.pairs_per_token.tolist(),
                "kv_read_per_token": self.kv_read_per_token.tolist(),
                "ratios": self.ratios.tolist(), "source": self.source}

    def ratio(self, work: Work) -> float:
        """Return measured over ideal seconds for one pass of this work.

        Outside the grid the ratio of the nearest grid point applies, so
        a shape never measured is priced at the closest measured shape.
        """
        tokens = float(work.tokens)
        if tokens <= 0:
            raise ValueError("a pass must have tokens")
        return interpolate(
            (np.log(self.tokens), self.pairs_per_token, self.kv_read_per_token),
            self.ratios,
            (math.log(tokens), float(work.pairs) / tokens,
             float(work.kv_read) / tokens))


def interpolate(axes, values, point) -> float:
    """Interpolate linearly along each axis, holding the edge value outside.

    Args:
        axes: One ascending array of grid points per dimension.
        values: Array with one dimension per axis.
        point: One coordinate per axis.
    """
    result = np.asarray(values, dtype=float)
    for axis, coordinate in zip(axes, point, strict=True):
        if len(axis) == 1:
            result = result[0]
            continue
        coordinate = min(max(coordinate, axis[0]), axis[-1])
        upper = min(max(int(np.searchsorted(axis, coordinate, side="right")), 1),
                    len(axis) - 1)
        weight = (coordinate - axis[upper - 1]) / (axis[upper] - axis[upper - 1])
        result = (1 - weight) * result[upper - 1] + weight * result[upper]
    return float(result)


def split_passes(work: Work, chunk_tokens: int) -> list[tuple[Work, float]]:
    """Split work into full passes and one partial pass, spread evenly.

    Args:
        work: The work of a node, summed over documents.
        chunk_tokens: The most tokens one forward pass holds.

    Returns:
        Pairs of one pass's work and the number of passes with that work:
        the full passes first, then the remainder when there is one.
    """
    total = work.total()
    if total.tokens <= 0:
        return []
    full = int(total.tokens // chunk_tokens)
    parts = []
    if full:
        parts.append((total * (chunk_tokens / total.tokens), float(full)))
    remainder = total.tokens - full * chunk_tokens
    if remainder > 0:
        parts.append((total * (remainder / total.tokens), 1.0))
    return parts


def predict_seconds(table: PassTable, work: Work, model: ModelSpec,
                    device: DeviceSpec, chunk_tokens: int) -> float:
    """Return predicted wall seconds for work, pass by pass.

    Args:
        table: Measurements for this model and device.
        work: The work of a node, summed over documents.
        model: The model the table was measured on.
        device: The device the table was measured on.
        chunk_tokens: The most tokens one forward pass holds.
    """
    seconds = 0.0
    for part, count in split_passes(work, chunk_tokens):
        ideal = speed_of_light(part, model, device, chunk_tokens).seconds
        seconds += count * ideal * table.ratio(part)
    return seconds


@lru_cache(maxsize=None)
def load_table(model: str, device: str) -> PassTable | None:
    """Return the saved table for a model and device, or None when absent.

    Raises:
        ValueError: A saved table names a different model or device, or
            holds invalid ratios.
    """
    path = TABLE_DIR / f"{model}__{device}.json"
    if not path.exists():
        return None
    table = PassTable.from_dict(json.loads(path.read_text()))
    if (table.model, table.device) != (model, device):
        raise ValueError(f"{path} was measured on {table.model} and "
                         f"{table.device}")
    return table
