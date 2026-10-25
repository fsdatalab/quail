"""Run Quail and stock vLLM on the same H100 over the map configurations.

One Modal function call runs both engines, one after the other, each
in its own child process so each gets a fresh CUDA context and the
whole GPU: Quail first, then vLLM. Each child records the GPU's UUID,
and the result asserts the two match. The configurations are the four
map quadrants of experiments/cells/map_quadrants.py and two with three
maps over one table in one query: Quail runs the three as one map
pipeline, and vLLM gets every (document, instruction) prompt in one
generate call. Quail runs the adaptive draft policy with GPU timing;
vLLM decodes greedily without speculation, with its default CUDA graph
capture sizes, prefix caching on, and its prefix cache reset before
each configuration. Query time excludes model startup for both;
startup is reported separately.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-map-same-node.log"
    uv run modal run --detach \
      experiments/cells/map_same_node.py::launch_same_node \
      2>&1 | tee "$run_log"

The result JSON lands under /results/ablations/ on the quail-results
volume; the function prints its path.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from bench.quailb_parallel import VOLUMES, app, image, results_vol
from experiments.cells.map_quadrants import (
    CONFIGURATIONS,
    LONG_MAP,
    RESULT_DIR,
    SHORT_MAP,
    _ensure_tables,
    _run_one,
    answer_stats,
)
from experiments.cells.map_vllm_baseline import _prompts
from experiments.cells.map_vllm_baseline import _run_one as _run_vllm

TOPIC_MAP = "Name the main topic of the document in a few words."

# (name, table, column, ((result column, instruction, max answer tokens), ...))
SAME_NODE_CONFIGURATIONS = tuple(
    (name, table, column, (("answer", instruction, cap),))
    for name, table, column, instruction, cap in CONFIGURATIONS) + tuple(
    (name, table, column, (("negatives", SHORT_MAP, 64),
                           ("summary", LONG_MAP, 256),
                           ("topic", TOPIC_MAP, 32)))
    for name, table, column in (("imdb_three_maps", "reviews", "body"),
                                ("cuad_three_maps", "contracts", "text")))

SINGLE_SUBMISSION = "stock vLLM, operator-at-a-time submission"
MULTI_SUBMISSION = "stock vLLM, pipelining submission"


def _gpu() -> str:
    """Return this process's GPU as nvidia-smi names it: UUID, then model."""
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True).stdout.strip()


def _chosen(only: str) -> tuple:
    """Return the configurations the comma-separated names choose; all when empty."""
    names = {name.strip() for name in only.split(",") if name.strip()}
    unknown = names - {config[0] for config in SAME_NODE_CONFIGURATIONS}
    if unknown:
        raise ValueError(f"unknown configurations: {sorted(unknown)}")
    return tuple(config for config in SAME_NODE_CONFIGURATIONS
                 if not names or config[0] in names)


def _quail(model: str, directory: Path, configurations) -> dict:
    """Run the configurations on Quail; return the GPU, startup, and runs."""
    import quail
    from bench.quailb import register_tables
    from quail.specs import H100_USD_PER_HOUR

    result = {"engine": "Quail", "gpu": _gpu(), "draft_policy": "adaptive",
              "runs": {}}
    with quail.Session(quail.EngineConfig(
            model=model, device="h100-sxm", backend="quail",
            gpu_timing=True)) as session:
        register_tables(session, directory)
        for name, table, column, maps in configurations:
            print(f"quail: running {name}", flush=True)
            run = _run_one(session, table, column, maps, "adaptive",
                           H100_USD_PER_HOUR)
            # the session boots the engine in its first query; the
            # engine's own wall time excludes that startup
            run["startup_s"] = round(run["wall_s"] - (run["engine_wall_s"] or 0.0), 2)
            result["runs"][name] = run
            print(json.dumps({name: run}, default=str), flush=True)
    first = next(iter(result["runs"].values()), {})
    result["startup_s"] = first.get("startup_s")
    return result


def _vllm(model: str, directory: Path, configurations) -> dict:
    """Run the configurations on stock vLLM; return the GPU, boot, and runs."""
    # the engine core runs in process, as map_vllm_baseline sets it: a
    # spawned core would not see the gigatoken tokenizer registration
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.pop("QUAIL_VLLM_SPECULATIVE", None)
    from quail.baselines.vllm import VLLMEngine, parse_capture_sizes
    from quail.specs import H100_USD_PER_HOUR, MODELS

    spec = MODELS[model]
    result = {"engine": "stock vLLM", "gpu": _gpu(), "speculative": None,
              "capture_sizes": "vllm", "runs": {}}
    with ExitStack() as resources:
        engine = VLLMEngine(capture_sizes=parse_capture_sizes("vllm"))
        settings, boot = engine.boot(spec, [0], resources=resources)
        result["boot"] = boot
        result["engine_settings"] = {
            key: value for key, value in settings["capacity"].items()
            if isinstance(value, (int, float, str, bool))}
        client = settings["client"]
        for name, table, column, maps in configurations:
            print(f"vllm: running {name}", flush=True)
            # each configuration prefills its documents itself, as
            # Quail does: the KV an earlier one cached is dropped
            if client.reset_prefix_cache() is False:
                raise RuntimeError("vLLM did not reset its prefix cache")
            per_map = [_prompts(directory, table, column, instruction, spec)
                       for _, instruction, _ in maps]
            # document-major order: a document's prompts arrive together,
            # so prefix caching can share its KV across them
            prompts = [prompt for document in zip(*per_map) for prompt in document]
            caps = [cap for _ in per_map[0] for _, _, cap in maps]
            run = _run_vllm(client, prompts, caps, keep_answers=True)
            answers = run.pop("answers")
            run["engine"] = SINGLE_SUBMISSION if len(maps) == 1 else MULTI_SUBMISSION
            run["num_requests"] = len(prompts)
            run["num_documents"] = len(per_map[0])
            run["usd_per_query"] = run["wall_s"] / 3600 * H100_USD_PER_HOUR
            if len(maps) > 1:
                run["maps"] = {
                    map_name: answer_stats(answers[index::len(maps)])
                    for index, (map_name, _, _) in enumerate(maps)}
            result["runs"][name] = run
            print(json.dumps({name: run}, default=str), flush=True)
    return result


_ENGINES = {"quail": _quail, "vllm": _vllm}


def _child(engine: str, model: str, directory: str, only: str, out: str) -> None:
    """Run one engine in this process and write its result JSON to out."""
    result = _ENGINES[engine](model, Path(directory), _chosen(only))
    Path(out).write_text(json.dumps(result, default=str))


def _run_child(engine: str, model: str, directory: Path, only: str) -> dict:
    """Run one engine in a fresh interpreter on this GPU; return its result."""
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / f"{engine}.json"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", "experiments.cells.map_same_node", engine,
             model, str(directory), only, str(out)],
            check=True, env=environment)
        result = json.loads(out.read_text())
    result["process_s"] = round(time.perf_counter() - started, 2)
    return result


@app.function(image=image, gpu="H100!", memory=98304, timeout=14400,
              volumes=VOLUMES)
def map_same_node(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                  only: str = "", engines: str = "quail,vllm") -> str:
    """Run the engines one after the other on this GPU; return the result's path.

    Args:
        model: The model to run.
        scale_factor: The corpus scale factor.
        only: Comma-separated configuration names to run; empty runs
            all six.
        engines: Comma-separated engines to run, from "quail" and
            "vllm"; they run in that order.

    Raises:
        RuntimeError: The engines ran on different GPUs.
    """
    chosen_engines = [engine for engine in ("quail", "vllm")
                      if engine in {e.strip() for e in engines.split(",")}]
    if not chosen_engines:
        raise ValueError(f"no known engine in {engines!r}")
    configurations = _chosen(only)
    directory = _ensure_tables(scale_factor,
                               [config[1] for config in configurations])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    measurements = {"model": model, "scale_factor": scale_factor,
                    "configurations": [config[0] for config in configurations],
                    "engines": {}}
    for engine in chosen_engines:
        print(f"starting {engine}", flush=True)
        measurements["engines"][engine] = _run_child(engine, model, directory, only)
    gpus = {engine: result["gpu"]
            for engine, result in measurements["engines"].items()}
    uuids = {gpu.split(",")[0] for gpu in gpus.values()}
    measurements["gpus"] = gpus
    measurements["same_gpu"] = len(uuids) == 1
    out = Path(RESULT_DIR) / f"map_same_node_{model}_sf{scale_factor}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurements, indent=2, default=str))
    results_vol.commit()
    print(f"saved {out}")
    if len(uuids) != 1:
        raise RuntimeError(f"the engines ran on different GPUs: {gpus}")
    return str(out)


@app.local_entrypoint()
def launch_same_node(model: str = "qwen3-4b-fp8", scale_factor: float = 0.1,
                     only: str = "", engines: str = "quail,vllm"):
    call = map_same_node.spawn(model=model, scale_factor=scale_factor, only=only,
                               engines=engines)
    print(f"function call id: {call.object_id}")
    print(call.get())


if __name__ == "__main__":
    _child(*sys.argv[1:6])
