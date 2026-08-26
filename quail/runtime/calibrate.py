"""Modal entry point for running calibration measurements on H100."""

import json
import os

from quail.planner.calibrate import resolve_pair
from quail.planner.calibration import (Calibration, commit_calibration,
                                       load_calibration)
from quail.runtime.worker import (app, hf_cache, image, kernel_cache,
                                  results_vol)

# This entry is H100-only until another device is wired here.
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
                         source=loaded_before["source"])
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
          f"a2={loaded.a2_s_per_token2} source={loaded.source}",
          flush=True)
    payload = calibrate_run.remote(
        model, device,
        dict(a=loaded.a_s_per_token, a2=loaded.a2_s_per_token2,
             source=loaded.source))
    result = json.loads(payload)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"saved {out}")
    print(f"[calibrate] measured a={result['a_s_per_token']} "
          f"a2={result['a2_s_per_token2']} "
          f"(was {loaded.a_s_per_token}, {loaded.a2_s_per_token2})")
    if commit:
        dest = commit_calibration(result)
        print(f"committed {dest}")
