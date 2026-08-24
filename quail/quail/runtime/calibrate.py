"""Modal entry for quail.planner.calibrate.measure.

Attached to quail-engine (same app as the worker). No new app name.

    uv run modal run quail/runtime/calibrate.py --model qwen3-4b-fp8 \
        --device h100-sxm 2>&1 | tee results/calibrate.log

--commit writes quail/calibration/{model}_{device}.json, where
load_calibration reads. Without it the measurement only lands in
results/. The local process loads the previous constants so the
container does not need the JSON files, and so loaded_before is the
prediction the house rule wants printed next to the fresh fit.

This function is wired to H100. A new device spec is not enough:
the gpu= decorator has to match the hardware.
"""

import json
import os

from quail.planner.calibrate import resolve_pair
from quail.planner.calibration import (Calibration, commit_calibration,
                                       load_calibration)
from quail.runtime.worker import (app, hf_cache, image, kernel_cache,
                                  results_vol)

_MODAL_DEVICE = "h100-sxm"


@app.function(image=image, gpu="H100!", timeout=3600, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/kernels": kernel_cache,
                       "/results": results_vol})
def calibrate_run(model: str, device: str, loaded_before: dict,
                  tokens_per_point: int = 1_500_000) -> str:
    if device != _MODAL_DEVICE:
        raise ValueError(
            f"this entry is wired to {_MODAL_DEVICE} (gpu=H100!); "
            f"got {device!r}. Add a gpu= mapping before calibrating "
            "another device.")
    from quail.planner.calibrate import measure
    spec, dev = resolve_pair(model, device)
    loaded = Calibration(a_s_per_token=loaded_before["a"],
                         a2_s_per_token2=loaded_before["a2"],
                         source=loaded_before["source"],
                         c_s_per_chunk=loaded_before.get("c", 0.0),
                         p_s_per_suffix=loaded_before.get("p", 0.0))
    result = measure(spec, dev, tokens_per_point=tokens_per_point,
                     loaded=loaded)
    print(json.dumps(result, indent=2), flush=True)
    os.makedirs("/results/calibrate", exist_ok=True)
    remote = f"/results/calibrate/{model}_{device}.json"
    with open(remote, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    kernel_cache.commit()
    return json.dumps(result)


@app.local_entrypoint()
def run(model: str = "qwen3-4b-fp8", device: str = "h100-sxm",
        commit: bool = False, out: str = "results/calibrate.json"):
    spec, dev = resolve_pair(model, device)
    loaded = load_calibration(spec, dev)
    print(f"[calibrate] loaded_before a={loaded.a_s_per_token} "
          f"a2={loaded.a2_s_per_token2} c={loaded.c_s_per_chunk} "
          f"p={loaded.p_s_per_suffix} source={loaded.source}",
          flush=True)
    payload = calibrate_run.remote(
        model, device,
        dict(a=loaded.a_s_per_token, a2=loaded.a2_s_per_token2,
             c=loaded.c_s_per_chunk, p=loaded.p_s_per_suffix,
             source=loaded.source))
    result = json.loads(payload)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"saved {out}")
    print(f"[calibrate] measured a={result['a_s_per_token']} "
          f"a2={result['a2_s_per_token2']} "
          f"c={result['c_s_per_chunk']} "
          f"p={result['p_s_per_suffix']} "
          f"(was a={loaded.a_s_per_token}, a2={loaded.a2_s_per_token2})")
    if commit:
        dest = commit_calibration(result)
        print(f"committed {dest}")
