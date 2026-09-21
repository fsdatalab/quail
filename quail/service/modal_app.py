"""Deploy the query service on Modal.

    modal deploy -m quail.service.modal_app 2>&1 | tee service-deploy.log

Deploys one web endpoint on the existing ``quail-engine`` app, backed by
one H100 container. The URL is printed by ``modal deploy``; pass it as
``endpoint`` to ``quail.Session``.

Where the data lives:

- Results and uploaded inputs go straight onto the ``quail-results``
  Volume under ``/results/quail-service``. They are write-once files.
- The live SQLite file stays on the container's local disk. A Volume
  has no file locking and rewrites a file on in-place writes, so SQLite
  must not run there. A checkpoint thread copies the database to the
  Volume after each change and calls ``volume.commit()``. On start
  the copy is restored. ``max_containers=1`` keeps one writer.

The endpoint is a public URL, so a bearer token is required. Create it
once before deploying:

    modal secret create quail-service-token QUAIL_SERVICE_TOKEN=<token>

Clients read the same variable, or pass ``token`` to ``ServiceClient``.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from quail.bench.images import gpu_image

APP_NAME = "quail-engine"
MODELS = ("qwen3-4b-fp8", "qwen3-32b-fp8", "diffusion-gemma-26b-a4b-fp8",
          "qwen3-reranker-0.6b-bf16", "qwen3-reranker-4b-bf16")
DEVICE = "h100-sxm"
GPUS = 1

VOLUME_DIR = Path("/results")
DATA_DIR = VOLUME_DIR / "quail-service"
DB_COPY = DATA_DIR / "quail.sqlite3"
LOCAL_DB = Path("/tmp/quail-service/quail.sqlite3")

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)


@app.cls(
    image=gpu_image(),
    gpu=f"H100!:{GPUS}",
    volumes={
        str(VOLUME_DIR): results_volume,
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
    },
    secrets=[modal.Secret.from_name("quail-service-token",
                                    required_keys=["QUAIL_SERVICE_TOKEN"])],
    timeout=24 * 3600,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
class QueryService:
    """One service container: web app, scheduler, and the executor child."""

    @modal.enter()
    def start(self) -> None:
        from quail.service.app import ServiceSettings, create_app
        from quail.service.checkpoint import Checkpoint, restore

        results_volume.reload()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if restore(DB_COPY, LOCAL_DB):
            print(f"restored {DB_COPY} to {LOCAL_DB}", flush=True)
        settings = ServiceSettings(
            data_dir=DATA_DIR,
            db_path=LOCAL_DB,
            models=MODELS,
            device=DEVICE,
            gpus=(GPUS,),
            token=os.environ["QUAIL_SERVICE_TOKEN"],
        )
        self.asgi = create_app(settings)
        self.service = self.asgi.state.service
        # the ASGI lifespan starts the scheduler; the checkpoint thread is
        # ours because it needs the Volume
        self.checkpoint = Checkpoint(self.service.store, DB_COPY,
                                     commit=results_volume.commit)
        self.checkpoint.start()
        self.service.add_closer(self.checkpoint.stop)
        if self.service.recovered:
            print(f"marked interrupted: {self.service.recovered}", flush=True)

    @modal.asgi_app()
    def web(self):
        return self.asgi

    @modal.exit()
    def stop(self) -> None:
        self.checkpoint.stop()
