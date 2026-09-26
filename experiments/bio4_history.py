r"""Run BIO-4 once per dated configuration of Quail's history.

Each Quail configuration turns on the features in quail.ablation.FEATURES
that merged by its date and turns the rest off. The vLLM configurations
are the baselines as they stood on their dates. Every configuration runs
BIO-4 in a fresh container on its own H100, all at once.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-bio4-history.log"
    uv run modal run --detach experiments/bio4_history.py::history \
      --sf 0.1 2>&1 | tee "$run_log"

Pass --only with comma-separated configuration names to rerun some of
them into a new run directory, or into an earlier one with --run-dir.
--startup-samples boots each configuration that many more times in
fresh containers without running the query, since startup time varies
from machine to machine. The log prints every function call id.

Saved on the quail-results volume under the printed run directory:
configurations/<name>.json holds one configuration's summary, <name>/
holds its QUAIL-B run with answers and scores, and
startup/<name>-<sample>.json holds each extra startup sample.
"""

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quail import ablation
from quail.bench.images import gpu_image
from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    kernel_cache,
    results_vol,
)
from quail.bench.results import write_json

QUERY = "BIO-4"
MODEL = "qwen3-4b-fp8"
# the tokenizer Quail used for documents before Gigatoken (#103)
image = gpu_image(packages=("bpe-qwen==0.1.5",))


@dataclass(frozen=True)
class Configuration:
    """One dated point in the history.

    Args:
        name: Short name used for files and the command line.
        label: What the configuration adds, for tables and plots.
        backend: The EngineConfig backend.
        as_of: UTC timestamp the configuration stands for; features
            merged after it are off.
        pull_request: The pull request that added the step, or None.
        extra_disabled: Features off even though they merged earlier.
    """

    name: str
    label: str
    backend: str
    as_of: str
    pull_request: int | None = None
    extra_disabled: tuple = ()

    @property
    def disabled(self) -> tuple:
        return tuple(sorted(
            ablation.merged_after(self.as_of) | set(self.extra_disabled)))


# the first Quail engine commit (44eaf81) and the first pipelined vLLM
# baseline commit (ba4173e) have no pull request
ENGINE_COMMIT = "2026-08-18T08:21:00Z"
BASELINE_COMMIT = "2026-08-18T09:30:18Z"
TODAY = "2026-09-26T00:00:00Z"
QUAIL_STEPS = (
    "pinned_staging", "attention_paths", "skip_arena_writes", "join_search",
    "compile_once", "shared_join_prompts", "filter_kv_reuse", "scan_ring",
    "boot_cache", "shared_retention", "join_continuous_batching",
    "projection_pushdown", "plan_on_estimates", "filter_join_streaming",
    "gigatoken",
)
CONFIGURATIONS = {c.name: c for c in (
    Configuration("vllm-defaults", "vLLM at default settings", "dumb_vllm",
                  "2026-08-01T18:27:42Z"),
    Configuration("vllm-tuned", "tuned vLLM, operator-at-a-time",
                  "stock_vllm", "2026-08-13T06:54:33Z", 2),
    Configuration("vllm-pipelined", "tuned vLLM, pipelined filters",
                  "pipelined_vllm", BASELINE_COMMIT),
    Configuration("vllm-today", "pipelined vLLM with Gigatoken",
                  "pipelined_vllm", ablation.FEATURES["vllm_gigatoken"].merged,
                  167),
    Configuration("quail-engine-vllm-kernels",
                  "Quail engine with vLLM's kernels", "quail", ENGINE_COMMIT,
                  extra_disabled=("triton_kernels",)),
    Configuration("quail-engine", "fused Triton kernels", "quail",
                  ENGINE_COMMIT, 3),
    *(Configuration(name, ablation.FEATURES[name].summary, "quail",
                    ablation.FEATURES[name].merged,
                    ablation.FEATURES[name].pull_request)
      for name in QUAIL_STEPS),
    Configuration("quail-today-repeat", "repeat of the last configuration",
                  "quail", TODAY),
)}
# these write the kernel and vLLM caches every later boot reads
CACHE_WRITERS = ("gigatoken", "vllm-today")

PREDICTION_TEXT = {
    0.1: (
        "sf=0.1 check: each configuration should finish. gigatoken and "
        "quail-today-repeat should take close to the saved 72.5 s for Quail "
        "on BIO-4, plus planning and tokenization now inside the timing; "
        "vllm-today close to the saved 430 s. Configurations before "
        "shared_join_prompts should compute about 3 times the fresh join "
        "tokens of the later ones, because the 38-token question then "
        "follows every partner instead of being written once per anchor."),
    0.5: (
        "sf=0.5: 2,500 reports and 2,934 terms give about 3.2 million "
        "evaluated pairs, 13 times sf=0.1. Quail today (gigatoken, "
        "quail-today-repeat) about 700 s of query time, from 67.8 s of join "
        "at sf=0.1 times 13. Configurations before shared_join_prompts about "
        "2,100 s; from shared_join_prompts to projection_pushdown about "
        "900 s; filter_join_streaming close to Quail today. vllm-today "
        "5,000 to 10,000 s. Startup does not depend on the scale factor: "
        "within the sf=0.1 spread of each configuration."),
}


def summarize(configuration, suite, result) -> dict:
    """The configuration's time, startup, tokens, and accuracy in one record."""
    (item,) = suite["queries"]
    measurements = item.get("measurements", {})
    session_ready_s = suite["metadata"].get("session_ready_s")
    boot_s = measurements.get("boot_s")
    return {
        "name": configuration.name,
        "label": configuration.label,
        "backend": configuration.backend,
        "as_of": configuration.as_of,
        "pull_request": configuration.pull_request,
        "disabled_features": list(configuration.disabled),
        "status": item["status"],
        "error": item.get("error"),
        "metrics": item.get("metrics"),
        "startup": {
            "session_ready_s": session_ready_s,
            "boot_s": boot_s,
            "startup_s": (None if session_ready_s is None or boot_s is None
                          else session_ready_s + boot_s),
            "boot": measurements.get("boot"),
        },
        "timing": {key: measurements.get(key) for key in (
            "submission_to_answer_s", "frontend_s", "planning_s",
            "input_ready_s", "token_wait_s", "physical_prepare_s",
            "model_wall_s", "finish_s", "answer_prepare_s", "collection_s")},
        "tokens": {key: measurements.get(key) for key in (
            "fresh_tokens", "cached_tokens", "input_tokens")},
        "kv_manager": measurements.get("kv_manager"),
        "gpu_uuids": result["gpu_uuids"],
        "process_cleanup": result["process_cleanup"],
        "collection_id": suite["collection_id"],
    }


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=86400,
    volumes=VOLUMES,
)
def run_configuration(name: str, sf: float, run_dir: str, collection: str,
                      commit_caches: bool = False) -> str:
    """Run BIO-4 under one configuration in a fresh process on this H100."""
    from quail.bench.process_isolation import run_backend_group_in_fresh_process

    configuration = CONFIGURATIONS[name]
    if "boot_cache" in configuration.disabled:
        # before #78 vLLM's cache lived in the container and started empty
        os.environ["VLLM_CACHE_ROOT"] = tempfile.mkdtemp(prefix="vllm-cache-")
    try:
        result = run_backend_group_in_fresh_process(
            data_dir=DATA_DIR, model=MODEL, sf=sf, query_ids=(QUERY,),
            run_dir=run_dir, ground_truth_collection=collection,
            methods=(configuration.backend,),
            disabled_features=configuration.disabled,
            output_name=name, started_at=time.time())
        summary = summarize(
            configuration, result["suites"][configuration.backend], result)
        write_json(Path(run_dir) / "configurations" / f"{name}.json", summary)
        return json.dumps(summary)
    finally:
        results_vol.commit()
        if commit_caches:
            kernel_cache.commit()


@app.function(
    image=image,
    gpu="H100!",
    memory=98304,
    timeout=3600,
    volumes=VOLUMES,
)
def sample_startup(name: str, sf: float, run_dir: str, sample: int) -> str:
    """Start one configuration's session and engine in a fresh process."""
    from quail.bench.process_isolation import (
        run_backend_group_in_fresh_process,
        startup_sample,
    )

    configuration = CONFIGURATIONS[name]
    if "boot_cache" in configuration.disabled:
        os.environ["VLLM_CACHE_ROOT"] = tempfile.mkdtemp(prefix="vllm-cache-")
    try:
        result = run_backend_group_in_fresh_process(
            target=startup_sample, data_dir=DATA_DIR, model=MODEL, sf=sf,
            query_id=QUERY, backend=configuration.backend,
            disabled_features=configuration.disabled, started_at=time.time())
        record = {"name": name, "sample": sample, **result}
        write_json(Path(run_dir) / "startup" / f"{name}-{sample}.json", record)
        return json.dumps(record)
    finally:
        results_vol.commit()


def _line(summary) -> str:
    metrics = summary.get("metrics") or {}
    startup = summary["startup"]
    return json.dumps({
        "name": summary["name"], "status": summary["status"],
        "runtime_s": metrics.get("runtime_s"),
        "startup_s": startup["startup_s"],
        "fresh_tokens": metrics.get("fresh_tokens"),
        "regret_tokens": metrics.get("regret_tokens"),
        "pairs": metrics.get("evaluated_document_pairs"),
        "agreement": ((metrics.get("accuracy") or {}).get("answer_accuracy")
                      or {}).get("accuracy"),
    })


@app.local_entrypoint()
def history(sf: float = 0.1, only: str = "", startup_samples: int = 0,
            run_dir: str = ""):
    names = ([item.strip() for item in only.split(",") if item.strip()]
             or list(CONFIGURATIONS))
    unknown = sorted(set(names) - set(CONFIGURATIONS))
    if unknown:
        raise ValueError(f"unknown configurations {unknown}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_dir or f"/results/ablations/bio4-history-sf{sf}-{stamp}"
    print(f"run directory: {run_dir}", flush=True)
    print(f"prediction: {PREDICTION_TEXT.get(sf, 'none stated')}",
          flush=True)
    for name in names:
        configuration = CONFIGURATIONS[name]
        print(f"configuration {name}: backend={configuration.backend} "
              f"as_of={configuration.as_of} "
              f"disabled={','.join(configuration.disabled) or '-'}",
              flush=True)
    data_call = ensure_data.spawn(sf, [QUERY], "")
    print(f"function call id: {data_call.object_id} (data)", flush=True)
    collection = data_call.get()
    print(f"ground truth collection: {collection}", flush=True)
    calls = {}
    for name in names:
        call = run_configuration.spawn(
            name, sf, run_dir, collection,
            commit_caches=name in CACHE_WRITERS)
        calls[name] = call
        print(f"function call id: {call.object_id} ({name})", flush=True)
    samples = {}
    for sample in range(1, startup_samples + 1):
        for name in names:
            call = sample_startup.spawn(name, sf, run_dir, sample)
            samples[(name, sample)] = call
            print(f"function call id: {call.object_id} "
                  f"(startup {name} {sample})", flush=True)
    for (name, sample), call in samples.items():
        try:
            record = json.loads(call.get())
            print(f"startup: {name} {sample} {record['startup_s']:.1f} s",
                  flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"failed: startup {name} {sample}: "
                  f"{type(error).__name__}: {error}", flush=True)
    for name, call in calls.items():
        try:
            print(f"finished: {_line(json.loads(call.get()))}", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"failed: {name}: {type(error).__name__}: {error}",
                  flush=True)
    print(f"summaries: {run_dir}/configurations/", flush=True)
