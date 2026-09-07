"""Capture vLLM CPU and GPU activity during benchmark joins."""

import json
import os
import time
import traceback
from functools import wraps
from pathlib import Path

PREDICTION_TEXT = (
    "vLLM will spend a larger fraction of join time running GPU operations "
    "than SGLang's measured 25.6%, with less time in request preparation "
    "and scheduling. Profiling must preserve the saved vLLM answers."
)
PREDICTION_TEXTS = {
    "FEV-9": PREDICTION_TEXT,
    "AGENT-1": (
        "vLLM will compute fewer fresh tokens than Quail by reusing shared "
        "prefixes across agent snapshots. The saved unprofiled filter took "
        "99.15 seconds, compared with Quail's 240.49 seconds."
    ),
    "BIO-3": (
        "Request handling and scheduling leave substantial GPU idle time "
        "during BIO-3's join. The saved unprofiled join took 474.04 seconds "
        "for 311,052 pairs. Run the filter normally to preserve its KV, then "
        "record the full join. Compare answers and work counts with the baseline."
    ),
}


def install_scheduler_scopes(worker):
    """Annotate scheduler operations in the vLLM engine process."""
    import torch
    from vllm.v1.core.sched.scheduler import Scheduler

    for name in ("schedule", "update_from_output", "add_request"):
        original = getattr(Scheduler, name)

        def annotated(self, *args, _method=original, _name=name, **kwargs):
            with torch.profiler.record_function(f"vllm.scheduler.{_name}"):
                return _method(self, *args, **kwargs)

        setattr(Scheduler, name, annotated)
    return os.getpid()


class ProfileExtension:
    """Expose scheduler instrumentation through vLLM's worker extension API."""

    def install_join_profile_scopes(self):
        """Add CPU intervals to the scheduler methods."""
        return install_scheduler_scopes(self)


def profile_worker(directory, connection, query="FEV-9"):
    """Run the existing vLLM baseline with profiling around each join."""
    os.setsid()
    root = Path(directory)
    traces = root / "worker-traces"
    traces.mkdir()
    try:
        import torch
        import vllm

        import quail.backends.request as request_module
        from quail.bench.process_isolation import run_backend_group

        original_init = vllm.LLM.__init__

        @wraps(original_init)
        def initialize(llm, *args, **kwargs):
            from quail.specs import MODELS

            kwargs["revision"] = MODELS["qwen3-4b-fp8"].revision
            kwargs["worker_extension_cls"] = (
                "experiments.vllm_join_profile_worker.ProfileExtension"
            )
            kwargs["profiler_config"] = {
                "profiler": "torch", "torch_profiler_dir": str(traces),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": False,
                "torch_profiler_dump_cuda_time_total": False,
            }
            return original_init(llm, *args, **kwargs)

        vllm.LLM.__init__ = initialize
        phase = "filter" if query == "AGENT-1" else "join"
        function = "_pipelined_filter" if phase == "filter" else "run_join_grouped"
        original_join = getattr(request_module, function)
        joins = []
        model_info = {}

        @wraps(original_join)
        def profile_join(client, sampling_params, prefixes, suffixes, true_ids,
                         *args, **kwargs):
            number = len(joins)
            destination = root / f"{phase}-{number}"
            destination.mkdir()
            llm = client.llm
            if not model_info:
                config = llm.llm_engine.vllm_config.model_config
                model_info.update({
                    "model_path": config.model,
                    "revision": getattr(config.hf_config, "_commit_hash", None),
                    "quantization_config": getattr(
                        config.hf_config, "quantization_config", None),
                    "capacity": client.capacity,
                    "scheduler_process_ids": llm.collective_rpc(
                        "install_join_profile_scopes"),
                })
                assert config.model == "Qwen/Qwen3-4B-FP8"
            before = set(traces.glob("*.trace.json.gz"))
            llm.start_profile(profile_prefix=f"{phase}-{number}")
            started_ns = time.time_ns()
            started = time.perf_counter()
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    with_stack=False, record_shapes=False,
                ) as driver:
                    with torch.profiler.record_function(f"quail.{phase}-{number}"):
                        result = original_join(
                            client, sampling_params, prefixes, suffixes,
                            true_ids, *args, **kwargs,
                        )
                elapsed = time.perf_counter() - started
                finished_ns = time.time_ns()
            finally:
                llm.stop_profile()
            produced = set(traces.glob("*.trace.json.gz")) - before
            assert len(produced) == 1, produced
            next(iter(produced)).rename(destination / "worker.trace.json.gz")
            driver.export_chrome_trace(str(destination / "driver.trace.json.gz"))
            record = {
                "phase": phase, "wall_s": elapsed,
                "generate_wall_s": result["wall_s" if phase == "filter" else "wall"],
                "started_unix_ns": started_ns, "finished_unix_ns": finished_ns,
                "fresh_tokens": result["fresh_tokens"],
                "cached_tokens": result["cached_tokens"],
                "traces": [str(p) for p in sorted(destination.glob("*.trace.json.gz"))],
            }
            if phase == "join":
                record.update({
                    "join": number, "anchors": len(prefixes),
                    "partners": len(suffixes), "pairs": len(prefixes) * len(suffixes),
                    "submission": result["submission"],
                    "true_pairs": sum(result["answers"]),
                })
            else:
                record["documents"] = len(prefixes)
            joins.append(record)
            (root / f"{phase}s.json").write_text(json.dumps(joins, indent=2))
            print(f"[profile] {phase} {number}: {json.dumps(record)}", flush=True)
            return result

        setattr(request_module, function, profile_join)
        result = run_backend_group(
            data_dir="/results/quailb_data", model="qwen3-4b-fp8", sf=0.1, lf=1,
            query_ids=(query,), run_label=root.name,
            prediction=PREDICTION_TEXTS[query],
            ground_truth_collection="gt_77bb8b128743a79aedddaa24c808c3f8",
            methods=("pipelined_vllm",),
        )
        assert len(joins) == {"FEV-9": 3, "BIO-3": 1, "AGENT-1": 1}[query]
        result.update({
            "prediction": PREDICTION_TEXTS[query], "profiled": True, f"{phase}s": joins,
            "query": query,
            "model_info": model_info,
            "versions": {"torch": torch.__version__, "vllm": vllm.__version__},
        })
        (root / "result.json").write_text(json.dumps(result, indent=2))
    except BaseException:
        (root / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        connection.send(None)
        connection.close()
