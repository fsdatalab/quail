"""Time BIO-2's pipelined vLLM join with raw and chat prompts in one container.

The September 12 run with raw prompts took 1069.16 seconds on BIO-2's
563,500 pairs; the September 18 run with non-thinking chat prompts took
933.23 seconds, 13% less, on a different physical GPU. This cell runs
the same join both ways in one container, so the host is held fixed and
only the prompt layout changes.

Prediction: the two layouts take the same time per pair within 5%, with
the chat layout no faster than the raw layout, because vLLM's time on
this join is per-request scheduler work and the chat layout adds 9
fresh tokens to every pair. If the chat layout is instead about 13%
faster here too, the cause is the batch composition (fewer requests per
step under the 25,305-token step budget), not the host.

Each measured join runs on a fresh vLLM engine in a fresh child
process, as the benchmark measures, with BIO-1 run first on that
engine so kernels are warm, as in both saved runs. The BIO tables are
already on quail-results and the references are read from the volume,
as the September 18 run did. The join call is wrapped to submit the
requested layout. The layout the checked out code renders natively is
measured as is; the other is derived from it by adding or removing the
chat wrapper tokens, which is exactly how the two branches differ.
Round one runs raw then chat, round two chat then raw. An earlier
version ran all joins on one engine; vLLM 0.26.0 crashed in a model
step at the start of the second join, so each child's output is saved
to worker.log on the volume. A container whose GPU starts hot or
throttled is refused and the entrypoint spawns again; the GPU's
temperature, clock, and throttle flags are sampled every 30 seconds
and saved with the result.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-bio2-prompt-layout.log"
    cell=experiments/bio2_prompt_layout.py::compare_prompt_layouts
    uv run modal run --detach "$cell" 2>&1 | tee "$run_log"

The run prints its Modal function call id. The result is saved on
quail-results under /results/ablations/bio2-prompt-layout-<UTC>/result.json;
pull it with:

    modal volume get quail-results ablations/bio2-prompt-layout-<UTC>/result.json $W
"""

import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from quail.bench.quailb_parallel import DATA_DIR, VOLUMES, app, image, results_vol

QUERIES = ("BIO-1", "BIO-2")
MEASURED = "BIO-2"
SCALE_FACTOR = 0.1
COLLECTION = "gt_91df55461cea394013812087a6ca6625"
REFERENCE_ROOT = "/results"    # the collection is on quail-results
LAYOUTS = ("raw", "chat")
# Qwen3 apply_chat_template(enable_thinking=False), as the runtime renders it.
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PREDICTION_TEXT = (
    "Raw and chat layouts take the same time per pair within 5% in one "
    "container, with chat no faster than raw. A 13% chat advantage here "
    "means batch composition, not the host, explains the September 18 "
    "vLLM time."
)
# A container whose GPU is already hot or throttled is refused; the
# entrypoint spawns again, which usually lands on another machine.
MAX_TEMPERATURE_C = 85
GPU_QUERY = ("temperature.gpu,clocks.sm,clocks.max.sm,power.draw,"
             "clocks_throttle_reasons.hw_slowdown,"
             "clocks_throttle_reasons.sw_thermal_slowdown,"
             "clocks_throttle_reasons.hw_thermal_slowdown")


class GpuUnhealthyError(RuntimeError):
    """The container's GPU is hot or throttled before any work starts."""


def gpu_sample() -> dict:
    """Return one nvidia-smi reading of temperature, clocks, and throttling."""
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0]
    temperature, clock, max_clock, power, *reasons = [
        field.strip() for field in output.split(",")]
    return {
        "time": time.time(), "temperature_c": int(temperature),
        "clock_mhz": int(clock), "max_clock_mhz": int(max_clock),
        "power_w": float(power),
        "throttled": any(reason == "Active" for reason in reasons),
    }


def require_healthy_gpu() -> dict:
    """Raise GpuUnhealthyError unless the GPU is cool and not throttled."""
    sample = gpu_sample()
    if sample["throttled"] or sample["temperature_c"] > MAX_TEMPERATURE_C:
        raise GpuUnhealthyError(f"GPU unhealthy at start: {sample}")
    return sample


def sample_gpu_forever(samples: list, stop: threading.Event, every_s=30.0):
    """Append a GPU reading every interval until stopped."""
    while not stop.is_set():
        try:
            samples.append(gpu_sample())
        except Exception as error:  # noqa: BLE001
            samples.append({"time": time.time(), "error": str(error)})
        stop.wait(every_s)


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def _strip(ids: list, head: list, tail: list) -> list:
    """Remove the chat wrapper tokens from one prefix or suffix."""
    if head:
        if list(ids[:len(head)]) != head:
            raise ValueError("prefix does not start with the chat prefix")
        ids = ids[len(head):]
    if tail:
        if list(ids[-len(tail):]) != tail:
            raise ValueError("suffix does not end with the chat suffix")
        ids = ids[:-len(tail)]
    return list(ids)


def layout_variants(prefixes, suffixes, prefix_ids, suffix_ids) -> dict:
    """Return the raw and chat token layouts, whichever one came in.

    Args:
        prefixes: Per-anchor prefix token lists as the runner built them.
        suffixes: Per-partner suffix token lists as the runner built them.
        prefix_ids: Token ids of the chat prefix.
        suffix_ids: Token ids of the chat suffix.
    """
    native_is_chat = bool(prefixes) and list(prefixes[0][:len(prefix_ids)]) == list(
        prefix_ids)
    if native_is_chat:
        raw = ([_strip(p, prefix_ids, []) for p in prefixes],
               [_strip(s, [], suffix_ids) for s in suffixes])
        chat = (list(prefixes), list(suffixes))
    else:
        raw = (list(prefixes), list(suffixes))
        chat = ([list(prefix_ids) + list(p) for p in prefixes],
                [list(s) + list(suffix_ids) for s in suffixes])
    return {"raw": raw, "chat": chat, "native": "chat" if native_is_chat else "raw"}


def _measure_one(directory: str, layout: str, connection) -> None:
    """Run BIO-1 then BIO-2 on a fresh engine, submitting BIO-2 in one layout."""
    os.setsid()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    # vLLM's engine process inherits these descriptors, so its crash text
    # lands in the file whatever the log stream drops.
    log = os.open(str(root / "worker.log"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    record = {"layout": layout}
    try:
        import quail.backends.request as request_module
        from quail.bench.process_isolation import (
            run_backend_group,
            visible_gpu_uuids,
        )

        original_join = request_module.run_join_grouped
        calls = []

        @wraps(original_join)
        def measured_join(client, sampling_params, prefixes, suffixes,
                          true_ids, *args, **kwargs):
            if calls:
                raise RuntimeError("BIO-2 should submit exactly one join")
            tokenizer = client.llm.get_tokenizer()

            def encode(text: str) -> list:
                return list(tokenizer.encode(text, add_special_tokens=False))

            variants = layout_variants(
                prefixes, suffixes, encode(CHAT_PREFIX), encode(CHAT_SUFFIX))
            layout_prefixes, layout_suffixes = variants[layout]
            started = time.perf_counter()
            result = original_join(
                client, sampling_params, layout_prefixes, layout_suffixes,
                true_ids, *args, **kwargs)
            elapsed = time.perf_counter() - started
            pairs = len(result["answers"])
            record.update({
                "native_layout": variants["native"],
                "wall_s": elapsed, "generate_wall_s": result["wall"],
                "pairs": pairs, "ms_per_pair": 1000.0 * elapsed / max(1, pairs),
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "submission": result["submission"],
                "true_pairs": int(sum(result["answers"])),
                "anchor_prefix_tokens": sum(len(p) for p in layout_prefixes),
                "partner_suffix_tokens": sum(len(s) for s in layout_suffixes),
                "capacity": client.capacity,
                "gpu_uuids": list(visible_gpu_uuids()),
            })
            calls.append(record)
            (root / "join.json").write_text(json.dumps(record, indent=2))
            # Sent now: the runner's scoring rejects the layout it did not
            # render (fresh tokens below the minimum of its own prompts).
            connection.send(("ok", record))
            connection.close()
            return result

        request_module.run_join_grouped = measured_join
        suite = run_backend_group(
            data_dir=DATA_DIR, model="qwen3-4b-fp8", sf=SCALE_FACTOR,
            query_ids=QUERIES, run_dir=str(root),
            ground_truth_collection=COLLECTION, methods=("pipelined_vllm",),
            root=REFERENCE_ROOT,
        )
        if len(calls) != 1:
            raise RuntimeError(f"expected one join run, got {len(calls)}")
        record["benchmark_status"] = suite["suites"]["pipelined_vllm"].get(
            "status")
    except BaseException:
        (root / "error.txt").write_text(traceback.format_exc())
        if not calls:
            connection.send(("error", traceback.format_exc()))
            connection.close()
        raise


def _run_child(root: Path, name: str, layout: str) -> dict:
    """Run one measurement in its own process group and return its record."""
    from quail.bench.process_isolation import _stop_process_group

    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_measure_one, args=(str(root / name), layout, sender))
    process.start()
    sender.close()
    try:
        status, payload = receiver.recv()
    finally:
        receiver.close()
        cleanup = _stop_process_group(process)
        results_vol.commit()
    if status != "ok":
        raise RuntimeError(f"{name} failed:\n{payload}")
    payload["run"] = name
    payload["process_cleanup"] = cleanup
    return payload


@app.function(
    image=image.add_local_python_source("experiments"),
    gpu="H100!", memory=98304, timeout=14400, volumes=VOLUMES,
)
def compare_layouts(rounds: int = 2) -> str:
    """Measure each layout on fresh engines, in alternating order per round."""
    import torch
    import vllm

    first_sample = require_healthy_gpu()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(f"/results/ablations/bio2-prompt-layout-{stamp}")
    root.mkdir(parents=True)
    samples = [first_sample]
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_gpu_forever, args=(samples, stop), daemon=True)
    sampler.start()
    joins = []
    for round_index in range(rounds):
        order = LAYOUTS if round_index % 2 == 0 else tuple(reversed(LAYOUTS))
        for layout in order:
            record = _run_child(root, f"round{round_index}-{layout}", layout)
            record["round"] = round_index
            record["gpu_samples_before"] = len(samples)
            joins.append(record)
            (root / "joins.json").write_text(json.dumps(joins, indent=2))
            print(f"[layout] {json.dumps(record)}", flush=True)
            results_vol.commit()
    stop.set()
    sampler.join(timeout=60)
    by_layout = {
        name: [j["ms_per_pair"] for j in joins if j["layout"] == name]
        for name in LAYOUTS}
    summary = {
        name: {"ms_per_pair_mean": sum(values) / len(values),
               "ms_per_pair_runs": values}
        for name, values in by_layout.items()}
    summary["chat_over_raw"] = (summary["chat"]["ms_per_pair_mean"]
                                / summary["raw"]["ms_per_pair_mean"])
    result = {
        "query": MEASURED, "warmup_query": QUERIES[0],
        "scale_factor": SCALE_FACTOR, "rounds": rounds,
        "prediction": PREDICTION_TEXT, "joins": joins, "summary": summary,
        "host": {"cpu_model": _cpu_model(), "cpu_count": os.cpu_count(),
                 "gpu_uuids": sorted({u for j in joins for u in j["gpu_uuids"]})},
        "versions": {"torch": torch.__version__, "vllm": vllm.__version__},
        "gpu_samples": samples,
        "gpu_throttled_samples": sum(1 for s in samples if s.get("throttled")),
        "result_volume_path": str(root / "result.json"),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2))
    results_vol.commit()
    return str(root / "result.json")


@app.local_entrypoint()
def compare_prompt_layouts(rounds: int = 2, attempts: int = 4):
    """Start the comparison, spawning again when a container's GPU is unhealthy."""
    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    for attempt in range(attempts):
        call = compare_layouts.spawn(rounds)
        print(f"function call id: {call.object_id} (attempt {attempt + 1})",
              flush=True)
        try:
            print(f"result volume path: {call.get()}", flush=True)
            return
        except Exception as error:  # noqa: BLE001
            if "GPU unhealthy at start" not in str(error):
                raise
            print(f"refused container: {error}", flush=True)
    raise RuntimeError(f"no healthy GPU in {attempts} attempts")
