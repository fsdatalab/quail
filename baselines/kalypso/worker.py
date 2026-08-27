"""Kalypso baseline worker: starts Kalypso's vLLM fork as a server
process, then sends join queries via its /v1/semantic/query HTTP API.

    Modal GPU cell attached to the existing quail-milestone1 app.
"""

import json
import os
import subprocess
import sys
import time

import modal
import requests

from .config import (
    APP_NAME,
    ENABLE_PREFIX_CACHING,
    GPU_MEMORY_UTILIZATION,
    HEALTH_POLL_S,
    HEALTH_TIMEOUT_S,
    MODEL_NAMES,
    SERVER_PORT,
)

app = modal.App(APP_NAME)

kalypso_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "ninja-build", "wget")
    .run_commands(
        "git clone --depth 1 https://github.com/goodluck-hojae/kalypso.git"
        " /opt/kalypso"
    )
    .run_commands(
        "pip install -r /opt/kalypso/requirements/build.txt",
    )
    .run_commands(
        "pip install -r /opt/kalypso/requirements/cuda.txt",
    )
    .run_commands(
        "cd /opt/kalypso && VLLM_USE_PRECOMPILED=1"
        " pip install . --no-build-isolation",
    )
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("baselines")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)


@app.cls(
    image=kalypso_image,
    gpu="H100!",
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/results": results_vol,
    },
)
class KalypsoWorker:
    GPU = "H100!"
    model: str = modal.parameter(default="qwen3-4b-fp8")

    @modal.enter()
    def start_server(self):
        model_name = MODEL_NAMES[self.model]
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--tensor-parallel-size", "1",
        ]
        if ENABLE_PREFIX_CACHING:
            cmd.append("--enable-prefix-caching")

        print(f"[kalypso-worker] starting server: {' '.join(cmd)}",
              flush=True)
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        health_url = f"http://localhost:{SERVER_PORT}/v1/semantic/healthz"
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                r = requests.get(health_url, timeout=3)
                if r.ok:
                    print("[kalypso-worker] server healthy", flush=True)
                    return
            except requests.ConnectionError:
                pass
            if self._server_proc.poll() is not None:
                raise RuntimeError(
                    f"Kalypso server exited with code "
                    f"{self._server_proc.returncode}"
                )
            time.sleep(HEALTH_POLL_S)

        raise TimeoutError(
            f"Kalypso server did not become healthy within "
            f"{HEALTH_TIMEOUT_S}s"
        )

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "_server_proc") and self._server_proc.poll() is None:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=10)

    @modal.method()
    def run_join(
        self,
        left_path: str,
        right_path: str,
        instruction: str,
        query_name: str,
    ) -> dict:
        results_vol.reload()

        query_payload = {
            "ops": [
                {
                    "op": "join",
                    "args": {
                        "instruction": instruction,
                        "right_table": right_path,
                    },
                }
            ],
            "data_path": left_path,
        }

        url = f"http://localhost:{SERVER_PORT}/v1/semantic/query"
        print(
            f"[kalypso-worker] {query_name}: POST {url} "
            f"left={left_path} right={right_path}",
            flush=True,
        )

        t0 = time.perf_counter()
        resp = requests.post(url, json=query_payload, timeout=3600)
        wall_s = time.perf_counter() - t0

        resp.raise_for_status()
        body = resp.json()

        n_output = body.get("num_output_rows", 0)
        server_latency = body.get("latency_sec")
        print(
            f"[kalypso-worker] {query_name}: done "
            f"wall={wall_s:.2f}s server_latency={server_latency}s "
            f"output_rows={n_output}",
            flush=True,
        )

        return {
            "query_name": query_name,
            "wall_time_s": wall_s,
            "server_latency_s": server_latency,
            "num_output_rows": n_output,
            "model": self.model,
            "results": body.get("results", []),
        }
