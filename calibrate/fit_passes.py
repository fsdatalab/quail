"""Fit a table of measured over ideal seconds from pass measurements.

Needs scikit-learn, which quail itself does not import.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from quail.cost.predict import PassTable

NUM_FOLDS = 10


def load(path: Path) -> list[dict]:
    """Read measurements saved by calibrate.passes.save."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def features(records: list[dict]) -> np.ndarray:
    """Return log tokens and pairs per token for each measurement."""
    return np.array([[math.log(r["tokens"]), r["pairs"] / r["tokens"]]
                     for r in records])


def targets(records: list[dict]) -> np.ndarray:
    """Return the log of measured over ideal seconds for each measurement."""
    return np.log([r["wall_seconds"] / r["sol_seconds"] for r in records])


def cross_validate(records: list[dict], seed: int = 0) -> dict:
    """Return held-out relative error of the forest and of one constant ratio.

    Raises:
        ValueError: Fewer measurements than folds.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    if len(records) < NUM_FOLDS:
        raise ValueError(f"need at least {NUM_FOLDS} measurements")
    x, y = features(records), targets(records)
    forest_error, constant_error = [], []
    for train, test in KFold(NUM_FOLDS, shuffle=True, random_state=seed).split(x):
        forest = RandomForestRegressor(200, min_samples_leaf=1, random_state=seed)
        forest.fit(x[train], y[train])
        forest_error += list(np.abs(np.expm1(forest.predict(x[test]) - y[test])))
        constant_error += list(np.abs(np.expm1(y[train].mean() - y[test])))
    return {"forest_median": float(np.median(forest_error)),
            "forest_p90": float(np.percentile(forest_error, 90)),
            "constant_median": float(np.median(constant_error)),
            "constant_p90": float(np.percentile(constant_error, 90))}


def fit_table(records: list[dict], source: str, seed: int = 0) -> PassTable:
    """Fit a forest and evaluate it on the grid of measured shapes.

    The grid holds the measured token counts and pairs per token; the
    forest does not extrapolate, and the table clamps at its edges.
    """
    from sklearn.ensemble import RandomForestRegressor

    x, y = features(records), targets(records)
    forest = RandomForestRegressor(200, random_state=seed).fit(x, y)
    tokens = np.unique([round(r["tokens"]) for r in records]).astype(float)
    pairs = np.unique([round(r["pairs"] / r["tokens"], 1) for r in records])
    points = np.array([[math.log(t), p] for t in tokens for p in pairs])
    ratios = np.exp(forest.predict(points)).reshape(len(tokens), len(pairs), 1)
    return PassTable(model=records[0]["model"], device=records[0]["device"],
                     tokens=tokens, pairs_per_token=pairs,
                     kv_read_per_token=np.array([0.0]), ratios=ratios,
                     source=source)


def main(measurements: Path, output: Path) -> None:
    """Fit a table, print held-out error, and save the table."""
    records = load(measurements)
    print(json.dumps(cross_validate(records), indent=2))
    table = fit_table(records, str(measurements))
    output.write_text(json.dumps(table.to_dict()))


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]), Path(sys.argv[2]))
