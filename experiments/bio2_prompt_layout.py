"""Time BIO-2's pipelined vLLM join with raw and chat prompts in one container.

The September 12 run with raw prompts took 1069.16 seconds on BIO-2's
563,500 pairs; the September 18 run with non-thinking chat prompts took
933.23 seconds, 13% less, on a different container. This cell runs the
same join both ways in one process on one vLLM engine, so the host and
the engine are held fixed and only the prompt layout changes.

Prediction: the two layouts take the same time per pair within 5%, with
the chat layout no faster than the raw layout, because vLLM's time on
this join is per-request scheduler work and the chat layout adds 9
fresh tokens to every pair. If the chat layout is instead about 13%
faster here too, the cause is the batch composition (fewer requests per
step under the 25,305-token step budget), not the host.

The query runs through the real runner and planner. The join call is
wrapped: for each round it runs the raw layout, resets the prefix
cache, runs the chat layout, and resets again. The layout the checked
out code renders natively is measured as is; the other is derived from
it by adding or removing the chat wrapper tokens, which is exactly how
the two branches differ. The runner scores the native layout's answers.

    run_log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-bio2-prompt-layout.log"
    uv run modal run --detach experiments/bio2_prompt_layout.py \
      2>&1 | tee "$run_log"

The run prints its Modal function call id. The result is saved on
quail-results under /results/ablations/bio2-prompt-layout-<UTC>/result.json;
pull it with:

    modal volume get quail-results ablations/bio2-prompt-layout-<UTC>/result.json $W
"""

import json
import multiprocessing as mp
import os
import time
import traceback
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from quail.bench.quailb_parallel import (
    DATA_DIR,
    VOLUMES,
    app,
    ensure_data,
    image,
    results_vol,
)

QUERY = "BIO-2"
SCALE_FACTOR = 0.1
COLLECTION = "gt_91df55461cea394013812087a6ca6625"
# Qwen3 apply_chat_template(enable_thinking=False), as the runtime renders it.
CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PREDICTION_TEXT = (
    "Raw and chat layouts take the same time per pair within 5% in one "
    "container, with chat no faster than raw. A 13% chat advantage here "
    "means batch composition, not the host, explains the September 18 "
    "vLLM time."
)


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


def _worker(directory: str, connection, rounds: int) -> None:
    """Run BIO-2 on pipelined vLLM with the join measured under both layouts."""
    os.setsid()
    root = Path(directory)
    try:
        import torch
        import vllm

        import quail.backends.request as request_module
        from quail.bench.process_isolation import run_backend_group

        original_join = request_module.run_join_grouped
        joins = []
        engine_info = {}

        @wraps(original_join)
        def measured_join(client, sampling_params, prefixes, suffixes,
                          true_ids, *args, **kwargs):
            if joins:
                raise RuntimeError("BIO-2 should submit exactly one join")
            tokenizer = client.llm.get_tokenizer()

            def encode(text: str) -> list:
                return list(tokenizer.encode(text, add_special_tokens=False))

            variants = layout_variants(
                prefixes, suffixes, encode(CHAT_PREFIX), encode(CHAT_SUFFIX))
            native = variants["native"]
            engine_info.update({
                "capacity": client.capacity, "native_layout": native,
                "anchor_prefix_tokens": {
                    name: sum(len(p) for p in variants[name][0])
                    for name in ("raw", "chat")},
                "partner_suffix_tokens": {
                    name: sum(len(s) for s in variants[name][1])
                    for name in ("raw", "chat")},
            })
            native_result = None
            for round_index in range(rounds):
                for name in ("raw", "chat"):
                    if client.reset_prefix_cache() is False:
                        raise RuntimeError("vLLM did not reset its prefix cache")
                    layout_prefixes, layout_suffixes = variants[name]
                    started = time.perf_counter()
                    result = original_join(
                        client, sampling_params, layout_prefixes,
                        layout_suffixes, true_ids, *args, **kwargs)
                    elapsed = time.perf_counter() - started
                    pairs = len(result["answers"])
                    record = {
                        "round": round_index, "layout": name,
                        "wall_s": elapsed, "generate_wall_s": result["wall"],
                        "pairs": pairs,
                        "ms_per_pair": 1000.0 * elapsed / max(1, pairs),
                        "fresh_tokens": result["fresh_tokens"],
                        "cached_tokens": result["cached_tokens"],
                        "submission": result["submission"],
                        "true_pairs": int(sum(result["answers"])),
                    }
                    joins.append(record)
                    (root / "joins.json").write_text(json.dumps(joins, indent=2))
                    print(f"[layout] {json.dumps(record)}", flush=True)
                    if name == native:
                        native_result = result
            return native_result

        request_module.run_join_grouped = measured_join
        suite = run_backend_group(
            data_dir=DATA_DIR, model="qwen3-4b-fp8", sf=SCALE_FACTOR,
            query_ids=(QUERY,), run_dir=str(root),
            ground_truth_collection=COLLECTION, methods=("pipelined_vllm",),
        )
        if len(joins) != 2 * rounds:
            raise RuntimeError(f"expected {2 * rounds} join runs, got {len(joins)}")
        by_layout = {
            name: [j["ms_per_pair"] for j in joins if j["layout"] == name]
            for name in ("raw", "chat")}
        summary = {
            name: {"ms_per_pair_mean": sum(values) / len(values),
                   "ms_per_pair_runs": values}
            for name, values in by_layout.items()}
        summary["chat_over_raw"] = (summary["chat"]["ms_per_pair_mean"]
                                    / summary["raw"]["ms_per_pair_mean"])
        result = {
            "query": QUERY, "scale_factor": SCALE_FACTOR, "rounds": rounds,
            "prediction": PREDICTION_TEXT, "joins": joins, "summary": summary,
            "engine": engine_info,
            "host": {"cpu_model": _cpu_model(), "cpu_count": os.cpu_count(),
                     "gpu_uuids": suite["gpu_uuids"]},
            "versions": {"torch": torch.__version__, "vllm": vllm.__version__},
            "benchmark": suite,
        }
        (root / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException:
        (root / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        connection.send(None)
        connection.close()


@app.function(
    image=image.add_local_python_source("experiments"),
    gpu="H100!", memory=98304, timeout=7200, volumes=VOLUMES,
)
def compare_layouts(rounds: int = 2) -> str:
    """Run the comparison in a child process and save its result."""
    from quail.bench.process_isolation import _stop_process_group

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(f"/results/ablations/bio2-prompt-layout-{stamp}")
    root.mkdir(parents=True)
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(str(root), sender, rounds))
    process.start()
    sender.close()
    try:
        receiver.recv()
    finally:
        receiver.close()
        cleanup = _stop_process_group(process)
        results_vol.commit()
    if (root / "error.txt").exists():
        raise RuntimeError((root / "error.txt").read_text())
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    result["process_cleanup"] = cleanup
    result["result_volume_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2))
    results_vol.commit()
    return str(result_path)


@app.local_entrypoint()
def main(rounds: int = 2):
    """Start the comparison and print its function call id and result path."""
    print(f"prediction: {PREDICTION_TEXT}", flush=True)
    collection = ensure_data.remote(SCALE_FACTOR, [QUERY], COLLECTION)
    print(f"label collection: {collection}", flush=True)
    call = compare_layouts.spawn(rounds)
    print(f"function call id: {call.object_id}", flush=True)
    print(f"result volume path: {call.get()}", flush=True)
