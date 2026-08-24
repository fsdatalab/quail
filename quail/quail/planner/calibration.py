"""The measured constants the planner's break-even decision consumes.

Four constants per (model, device) pair, written offline by
`quail.planner.calibrate.measure` (Modal entry: quail/runtime/calibrate.py)
and checked into quail/calibration/:

    a      seconds per fresh token (linear-layer matmuls; 1/rate)
    a2     seconds per attention pair (one value for both causal
           and paged attention: the FLOPs per pair are the same
           regardless of where the KV lives)
    c      seconds per chunk (CUDA launch floor per forward pass)
    p      seconds per paged-attention suffix dispatch (kernel
           overhead from non-contiguous KV page reads)

The chunk-level cost model:

    gpu_seconds = a * T  +  a2 * S  +  c  +  p * suffixes

    T          fresh tokens in the chunk
    S          total attention pairs: causal (n^2 per segment, the
               /2 absorbed into a2) plus cross-read (suffix_tokens *
               anchor_tokens, no /2)
    suffixes   paged-attention dispatch count in the chunk

The planner uses only a and a2 (for the restore-vs-recompute
break-even); c and p are for prediction and diagnostics.

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
    source: str
    c_s_per_chunk: float = 0.0
    p_s_per_suffix: float = 0.0

    @property
    def rate_tokens_per_s(self) -> float:
        return 1.0 / self.a_s_per_token


def channel_bandwidths() -> dict:
    """Channel name -> bytes/s, from the host table."""
    with open(CALIBRATION_DIR / "channels.json") as f:
        return json.load(f)["bandwidth_bytes_per_s"]


def lstsq(ys, cols):
    """Ordinary least squares via normal equations."""
    k = len(cols[0])
    ata = [[sum(c[i] * c[j] for c in cols) for j in range(k)]
           for i in range(k)]
    atb = [sum(c[i] * y for c, y in zip(cols, ys)) for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            f = ata[j][i] / ata[i][i]
            for m in range(i, k):
                ata[j][m] -= f * ata[i][m]
            atb[j] -= f * atb[i]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (atb[i] - sum(ata[i][j] * x[j]
                for j in range(i + 1, k))) / ata[i][i]
    return x


def fit_cost_model(points):
    """OLS for gpu_s = a*T + a2*S + c + p*suffixes.
    points: list of dicts with keys T, S, suffixes, gpu_s.
    Returns (a, a2, c, p). Needs at least four points."""
    n = len(points)
    if n < 4:
        raise ValueError(f"need at least 4 points, got {n}")
    ys = [p["gpu_s"] for p in points]
    cols = [(p["T"], p["S"], 1.0, float(p["suffixes"])) for p in points]
    a, a2, c, p = lstsq(ys, cols)
    return a, a2, c, p


def _load_file(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _scale(model: ModelSpec, device: DeviceSpec,
           anchor_model: ModelSpec, anchor_device: DeviceSpec) -> float:
    """Spec-ratio scaling of per-token compute cost from the anchor."""
    return ((model.params / anchor_model.params)
            * (anchor_device.peak_flops / device.peak_flops))


def load_calibration(model: ModelSpec, device: DeviceSpec) -> Calibration:
    path = CALIBRATION_DIR / f"{model.name}_{device.name}.json"
    if path.exists():
        d = _load_file(path)
        return Calibration(a_s_per_token=d["a_s_per_token"],
                           a2_s_per_token2=d["a2_s_per_token2"],
                           source="calibrated",
                           c_s_per_chunk=d.get("c_s_per_chunk", 0.0),
                           p_s_per_suffix=d.get("p_s_per_suffix", 0.0))

    anchor = _load_file(CALIBRATION_DIR / ANCHOR_FILE)
    anchor_model = MODELS[anchor["model"]]
    anchor_device = DEVICES[anchor["device"]]
    s = _scale(model, device, anchor_model, anchor_device)
    return Calibration(
        a_s_per_token=anchor["a_s_per_token"] * s,
        a2_s_per_token2=anchor["a2_s_per_token2"] * s,
        source=f"spec-scaled from {anchor['model']}/{anchor['device']}")


def make_record(model: ModelSpec, device: DeviceSpec,
                a: float, a2: float, c: float, p: float,
                points: list, channels: dict,
                loaded: Calibration) -> dict:
    """The JSON the measure step returns and --commit writes from."""
    return dict(
        model=model.name, device=device.name,
        a_s_per_token=a, a2_s_per_token2=a2,
        c_s_per_chunk=c, p_s_per_suffix=p,
        provenance=dict(
            a="seconds per fresh token, OLS over per-chunk GPU time",
            a2="seconds per attention pair (one coefficient, "
               "causal and paged), same fit",
            c="per-chunk CUDA launch floor, same fit intercept",
            p="per paged-attention suffix dispatch, same fit"),
        points=points,
        channels_measured_bytes_per_s=channels,
        loaded_before=dict(a=loaded.a_s_per_token,
                           a2=loaded.a2_s_per_token2,
                           c=loaded.c_s_per_chunk,
                           p=loaded.p_s_per_suffix,
                           source=loaded.source))


def commit_calibration(record: dict, dest: Path | None = None) -> Path:
    """Write the four constants where load_calibration reads them."""
    dest = dest or (CALIBRATION_DIR
                    / f"{record['model']}_{record['device']}.json")
    keep = {k: record[k] for k in
            ("model", "device", "a_s_per_token",
             "a2_s_per_token2", "c_s_per_chunk",
             "p_s_per_suffix", "provenance")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(keep, f, indent=2)
        f.write("\n")
    return dest
