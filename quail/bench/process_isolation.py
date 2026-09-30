"""Run benchmark backend groups in fresh child processes."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import time
import traceback
from collections.abc import Sequence
from pathlib import Path


def visible_gpu_uuids() -> tuple[str, ...]:
    """Return the physical UUIDs visible to the current process."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def run_backend_group(
    *,
    data_dir: str,
    model: str,
    sf: float,
    query_ids: Sequence[str],
    run_dir: str,
    ground_truth_collection: str,
    methods: Sequence[str],
    root: str | None = None,
    label_scoring: str | None = None,
    attention: str | None = None,
    gpu_timing: bool = False,
    suite_name: str | None = None,
    canvas_draws: str | None = None,
) -> dict:
    """Run backend methods while sharing one loaded model when possible.

    suite_name is the directory the suites go under per method; it
    defaults to the queries' family, which two groups of one family
    (BIO-5 and BIO-6) must not share.
    """
    from quail import EngineConfig
    from quail.bench.quailb import run_suite
    from quail_b.queries import query_family_name

    query_ids = tuple(query_ids)
    methods = tuple(methods)
    if not methods:
        raise ValueError("at least one backend is required")
    family = query_family_name(query_ids)
    run_dir = Path(run_dir)
    suites = {}
    # several label scoring rules, comma-separated, run the Quail
    # method once per rule in this one process, on this one GPU, so
    # their times compare without container-to-container variance
    rules = [rule.strip() for rule in (label_scoring or "").split(",")
             if rule.strip()] or [None]
    # so do several canvas draw counts
    draw_counts = [int(count) for count in (canvas_draws or "").split(",")
                   if count.strip()] or [None]
    settings = [(rule, draws) for rule in rules for draws in draw_counts]
    for method in methods:
        for rule, draws in settings:
            name = method
            if method == "quail" and len(rules) > 1:
                name += f"-{rule}"
            if method == "quail" and len(draw_counts) > 1:
                name += f"-draws{draws}"
            print(
                f"[{family}] running {name} for {len(query_ids)} queries",
                flush=True,
            )
            suite = run_suite(
                query_ids, sf=sf,
                config=EngineConfig(
                    gpus=1,
                    model=model,
                    backend=method,
                    device="h100-sxm",
                    label_scoring=rule,
                    attention=attention,
                    gpu_timing=gpu_timing,
                    **({} if draws is None else {"canvas_draws": draws}),
                ),
                data_dir=Path(data_dir) / f"sf{sf}",
                ground_truth_collection=ground_truth_collection or None,
                root=root,
                output_dir=run_dir / name / (suite_name or family),
            )
            suite["run_id"] = run_dir.name
            suite["query_family"] = {"name": family, "query_ids": list(query_ids)}
            suites[name] = suite
            if method != "quail":
                break
    return {
        "query_family": family,
        "query_ids": list(query_ids),
        "ground_truth_collection": next(iter(suites.values()))["collection_id"],
        "gpu_uuids": list(visible_gpu_uuids()),
        "methods": list(suites),
        "suites": suites,
    }


def _child_main(connection, arguments: dict) -> None:
    """Run one backend group and send its result to the parent."""
    os.setsid()
    try:
        connection.send(("ok", run_backend_group(**arguments)))
    except BaseException:  # noqa: BLE001
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _gpu_memory_used_mib() -> tuple[int, ...]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return tuple(
        int(line.strip()) for line in output.splitlines() if line.strip()
    )


def _stop_process_group(process) -> dict:
    """Stop one backend process group and wait for GPU memory release."""
    process_group = process.pid
    signal_name = "SIGTERM"
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5.0
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_group_exists(process_group):
        signal_name = "SIGKILL"
        os.killpg(process_group, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while (
            _process_group_exists(process_group)
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
    process.join(timeout=1.0)
    if _process_group_exists(process_group):
        raise RuntimeError(
            f"backend process group {process_group} did not exit"
        )

    deadline = time.monotonic() + 30.0
    memory_used = _gpu_memory_used_mib()
    while any(value >= 1_024 for value in memory_used):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "GPU memory remained allocated after backend process exit: "
                f"{memory_used} MiB"
            )
        time.sleep(0.1)
        memory_used = _gpu_memory_used_mib()
    return {
        "signal": signal_name,
        "exit_code": process.exitcode,
        "gpu_memory_used_mib_after_exit": list(memory_used),
    }


def run_backend_group_in_fresh_process(**arguments) -> dict:
    """Run one backend group and destroy its CUDA context afterward."""
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_main,
        args=(child_connection, arguments),
    )
    process.start()
    child_connection.close()
    message = None
    receive_error = None
    try:
        try:
            message = parent_connection.recv()
        except EOFError as error:
            receive_error = error
    finally:
        parent_connection.close()
        cleanup = _stop_process_group(process)
    if receive_error is not None:
        raise RuntimeError(
            f"backend process exited with code {process.exitcode}"
        ) from receive_error
    status, result = message
    if status != "ok":
        raise RuntimeError(f"backend process failed:\n{result}")
    result["process_cleanup"] = cleanup
    return result
