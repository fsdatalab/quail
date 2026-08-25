"""The measured constants the planner's break-even decision consumes.

Exactly two numbers per (model, device) pair, written offline by
`quail.planner.calibrate.measure` (Modal entry: quail/runtime/calibrate.py)
and checked into quail/calibration/:

    a      seconds per fresh token in the packed loop (1/rate; embeds
           the measured efficiency factor)
    a2     seconds per token-pair of attention (the quadratic
           coefficient; refines the restore break-even at long
           documents)

Plus one host table, model-independent: the channel bandwidths from
the pinprobe protocol.

Nothing is ever measured at plan time. A pair without a file gets
spec-ratio-scaled defaults from the anchor measurement (4B/H100), and
the Calibration says so in `source` so explain() can print it.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from quail.specs import DEVICES, MODELS, DeviceSpec, ModelSpec

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"
ANCHOR_FILE = "qwen3-4b-fp8_h100-sxm.json"


@dataclass(frozen=True)
class Calibration:
    a_s_per_token: float
    a2_s_per_token2: float
    source: str            # "calibrated" | "spec-scaled from <anchor>"

    @property
    def rate_tokens_per_s(self) -> float:
        return 1.0 / self.a_s_per_token


def channel_bandwidths() -> dict:
    """Channel name -> bytes/s, from the host table."""
    with open(CALIBRATION_DIR / "channels.json") as f:
        return json.load(f)["bandwidth_bytes_per_s"]


def fit_affine(points) -> tuple[float, float]:
    """Least-squares (a, a2) for t = a + a2*h over (h, t) points -
    measure()'s fit, kept here so it is CPU-testable. Needs at least
    two distinct lengths."""
    n = len(points)
    sx = sum(h for h, _ in points)
    sy = sum(t for _, t in points)
    sxx = sum(h * h for h, _ in points)
    sxy = sum(h * t for h, t in points)
    denom = n * sxx - sx * sx
    if denom <= 0:
        raise ValueError("need at least two distinct lengths")
    a2 = (n * sxy - sx * sy) / denom
    a = (sy - a2 * sx) / n
    return a, a2


def _load_file(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _scale(model: ModelSpec, device: DeviceSpec,
           anchor_model: ModelSpec, anchor_device: DeviceSpec) -> float:
    """Spec-ratio scaling of per-token compute cost from the anchor: a
    model with more params costs proportionally more per token, a
    device with a higher ceiling proportionally less. The measured
    efficiency is assumed to travel; the absolute rates do not."""
    return ((model.params / anchor_model.params)
            * (anchor_device.peak_flops / device.peak_flops))


def load_calibration(model: ModelSpec, device: DeviceSpec) -> Calibration:
    path = CALIBRATION_DIR / f"{model.name}_{device.name}.json"
    if path.exists():
        d = _load_file(path)
        return Calibration(a_s_per_token=d["a_s_per_token"],
                           a2_s_per_token2=d["a2_s_per_token2"],
                           source="calibrated")

    anchor = _load_file(CALIBRATION_DIR / ANCHOR_FILE)
    anchor_model = MODELS[anchor["model"]]
    anchor_device = DEVICES[anchor["device"]]
    s = _scale(model, device, anchor_model, anchor_device)
    return Calibration(
        a_s_per_token=anchor["a_s_per_token"] * s,
        a2_s_per_token2=anchor["a2_s_per_token2"] * s,
        source=f"spec-scaled from {anchor['model']}/{anchor['device']}")


def make_record(model: ModelSpec, device: DeviceSpec,
                a: float, a2: float,
                points: list, channels: dict, loaded: Calibration,
                lengths, tokens_per_point: int) -> dict:
    """The JSON the measure step returns and --commit writes from."""
    return dict(
        model=model.name, device=device.name,
        a_s_per_token=a, a2_s_per_token2=a2,
        provenance=dict(
            a=("wall seconds per fresh token, length sweep "
               f"{list(lengths)} at ~{tokens_per_point} tokens per "
               "point, affine fit intercept"),
            a2="affine fit slope of the same sweep"),
        points=points,
        channels_measured_bytes_per_s=channels,
        loaded_before=dict(a=loaded.a_s_per_token,
                           a2=loaded.a2_s_per_token2,
                           source=loaded.source))


def commit_calibration(record: dict, dest: Path | None = None) -> Path:
    """Write the two constants where load_calibration reads them."""
    dest = dest or (CALIBRATION_DIR
                    / f"{record['model']}_{record['device']}.json")
    keep = {k: record[k] for k in
            ("model", "device", "a_s_per_token",
             "a2_s_per_token2", "provenance")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(keep, f, indent=2)
        f.write("\n")
    return dest
