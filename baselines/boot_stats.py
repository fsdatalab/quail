"""Mean/median aggregation for boot-profile trial rows."""

from __future__ import annotations

import statistics


def mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return round(statistics.mean(vals), 2)


def median(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return round(statistics.median(vals), 2)


def summarize(trials: list[dict], phase: str, keys: list[str]) -> dict:
    """Mean and median per key across trials for cold or warm."""
    out = {}
    for key in keys:
        xs = [t[phase].get(key) for t in trials]
        out[key] = dict(
            mean=mean(xs),
            median=median(xs),
            trials=[None if x is None else round(x, 2)
                    if isinstance(x, float) else x
                    for x in xs])
    return out


def aggregate(side: str, trials: list[dict]) -> dict:
    trials = sorted(trials, key=lambda t: t["trial"])
    if side == "quail":
        keys = ["boot_s", "load_model_s", "arena_s", "pipeline_s",
                "warm_kernels_s"]
    else:
        keys = ["boot_s", "llm_init_s", "weight_load_s", "kv_profile_s"]
    return dict(
        side=side,
        n_trials=len(trials),
        trials=trials,
        cold=summarize(trials, "cold", keys),
        warm=summarize(trials, "warm", keys),
    )
