"""The measured constants the planner's break-even decisions consume.

Exactly three numbers per (model, device) pair, written offline by
`quail calibrate` and checked into quail/calibration/:

    a      seconds per fresh token in the packed loop (1/rate; embeds
           the measured efficiency factor)
    a2     seconds per token-pair of attention (the quadratic
           coefficient; refines both break-evens at long documents)
    q_kv   the fp8-KV conversion tax per fresh token

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
    q_kv_s_per_token: float
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
    the calibrate cell's fit, kept here so it is CPU-testable. Needs
    at least two distinct lengths."""
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
                           q_kv_s_per_token=d["q_kv_s_per_token"],
                           source="calibrated")

    anchor = _load_file(CALIBRATION_DIR / ANCHOR_FILE)
    anchor_model = MODELS[anchor["model"]]
    anchor_device = DEVICES[anchor["device"]]
    s = _scale(model, device, anchor_model, anchor_device)
    # q_kv is an elementwise conversion: bandwidth work, so it scales
    # with KV elements per token and inversely with device memory
    # bandwidth.
    q_scale = ((model.kv_elements_per_token
                / anchor_model.kv_elements_per_token)
               * (anchor_device.hbm_bw / device.hbm_bw))
    return Calibration(
        a_s_per_token=anchor["a_s_per_token"] * s,
        a2_s_per_token2=anchor["a2_s_per_token2"] * s,
        q_kv_s_per_token=anchor["q_kv_s_per_token"] * q_scale,
        source=f"spec-scaled from {anchor['model']}/{anchor['device']}")
