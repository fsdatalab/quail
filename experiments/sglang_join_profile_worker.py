"""Worker instrumentation for the SGLang join profiling experiment."""

from __future__ import annotations

import json
import os
import time
import traceback
from functools import wraps
from pathlib import Path

PREDICTION_TEXT = (
    "CPU request handling and batch scheduling leave substantial gaps between "
    "GPU operations in the SGLang joins. Expect GPU kernels and transfers to "
    "occupy less than half of the captured join wall time. Compare profiled "
    "answers and work counts with the saved 135.74-second unprofiled FEV-9 run; "
    "profiling overhead must not replace the benchmark result."
)
BASELINE_PATH = (
    "/results/benchmarks/quailb/families/20260906T222629Z-sglang-suffix-major/"
    "fever-sglang-process.json"
)


def process_cpu_times(engine):
    """Read CPU seconds for the driver and SGLang child processes."""
    import psutil

    result = {}
    for pid in [os.getpid(), *engine.get_all_child_pids()]:
        process = psutil.Process(pid)
        value = process.cpu_times()
        result[str(pid)] = {
            "name": process.name(),
            "user_s": value.user,
            "system_s": value.system,
        }
    return result


def profile_worker(directory, connection, input_detail=False):
    """Run FEV-9 with profiling scoped to each join call."""
    os.setsid()
    root = Path(directory)
    try:
        import sglang
        import torch

        import quail.backends.request as request_module
        from quail.bench.process_isolation import run_backend_group

        original = request_module.run_join_grouped
        joins = []
        model_info = {}
        details = None
        if input_detail:
            from experiments.sglang_input_profile_worker import InputProfiler

            details = InputProfiler()

        @wraps(original)
        def profiled_join(client, sampling_params, prefixes, suffixes, true_ids,
                          **kwargs):
            number = len(joins)
            destination = root / f"join-{number}"
            destination.mkdir(parents=True)
            engine = client.engine
            if details is not None:
                details.reset(number, destination)
            if not model_info:
                config = engine.tokenizer_manager.model_config.hf_config
                model_info.update({
                    "model_path": engine.server_args.model_path,
                    "revision": getattr(config, "_commit_hash", None),
                    "quantization_config": getattr(config, "quantization_config", None),
                    "capacity": client.capacity,
                })
                assert model_info["model_path"] == "Qwen/Qwen3-4B-FP8"
            engine.start_profile(
                output_dir=str(destination),
                activities=["GPU"] if input_detail else ["CPU", "GPU"],
                with_stack=False, record_shapes=False,
                profile_id=f"join-{number}",
            )
            before = process_cpu_times(engine)
            started_ns = time.time_ns()
            started = time.perf_counter()
            driver_path = destination / "driver.trace.json.gz"
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    with_stack=False, record_shapes=False,
                ) as driver:
                    with torch.profiler.record_function(f"quail.join-{number}"):
                        result = original(
                            client, sampling_params, prefixes, suffixes,
                            true_ids, **kwargs,
                        )
                elapsed = time.perf_counter() - started
                finished_ns = time.time_ns()
                after = process_cpu_times(engine)
            finally:
                engine.stop_profile()
            driver.export_chrome_trace(str(driver_path))
            record = {
                "join": number, "anchors": len(prefixes), "partners": len(suffixes),
                "pairs": len(prefixes) * len(suffixes),
                "submission": result["submission"],
                "wall_s": elapsed, "generate_wall_s": result["wall"],
                "started_unix_ns": started_ns, "finished_unix_ns": finished_ns,
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "true_pairs": sum(result["answers"]),
                "process_cpu": {
                    pid: {
                        "name": value["name"],
                        "user_s": value["user_s"] - before[pid]["user_s"],
                        "system_s": value["system_s"] - before[pid]["system_s"],
                    }
                    for pid, value in after.items()
                },
                "traces": [str(path) for path
                           in sorted(destination.glob("*.trace.json.gz"))],
            }
            if details is not None:
                record["input_preparation"] = details.summary()
            assert len(record["traces"]) == 2, record["traces"]
            joins.append(record)
            (root / "joins.json").write_text(json.dumps(joins, indent=2))
            print(f"[profile] join {number}: {json.dumps(record)}", flush=True)
            return result

        request_module.run_join_grouped = profiled_join
        result = run_backend_group(
            data_dir="/results/quailb_data", model="qwen3-4b-fp8", sf=0.1,
            query_ids=("FEV-9",), run_dir=str(root),
            ground_truth_collection="gt_77bb8b128743a79aedddaa24c808c3f8",
            methods=("pipelined_sglang",),
        )
        assert len(joins) == 3
        result.update({
            "prediction": PREDICTION_TEXT, "baseline_volume_path": BASELINE_PATH,
            "profiled": True, "joins": joins,
            "input_detail": input_detail,
            "model_info": model_info,
            "versions": {"torch": torch.__version__, "sglang": sglang.__version__},
        })
        (root / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException:
        (root / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        connection.send(None)
        connection.close()
