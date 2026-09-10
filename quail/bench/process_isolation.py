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
) -> dict:
    """Run backend methods while sharing one loaded model when possible."""
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
    for method in methods:
        print(
            f"[{family}] running {method} for {len(query_ids)} queries",
            flush=True,
        )
        suite = run_suite(
            query_ids, sf=sf,
            config=EngineConfig(model=model, backend=method, gpus=1),
            data_dir=Path(data_dir) / f"sf{sf}",
            ground_truth_collection=ground_truth_collection or None,
            output_dir=run_dir / method / family,
        )
        suite["run_id"] = run_dir.name
        suite["query_family"] = {"name": family, "query_ids": list(query_ids)}
        suites[method] = suite
    return {
        "query_family": family,
        "query_ids": list(query_ids),
        "ground_truth_collection": suites[methods[0]]["collection_id"],
        "gpu_uuids": list(visible_gpu_uuids()),
        "methods": list(methods),
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
