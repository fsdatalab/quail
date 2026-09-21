"""Deploy Quail Server on Modal.

    modal deploy -m quail.server.modal_app 2>&1 | tee server-deploy.log

Deploys one web endpoint on the existing ``quail-engine`` app, backed by
one H100 container. The URL is printed by ``modal deploy``; pass it as
``endpoint`` to ``quail.Session``.

Where the data lives:

- Results and uploaded inputs go straight onto the ``quail-results``
  Volume under ``/results/quail-server``. They are write-once files.
- The live SQLite file stays on the container's local disk. A Volume
  has no file locking and rewrites a file on in-place writes, so SQLite
  must not run there. A checkpoint thread copies the database to the
  Volume after each change and calls ``volume.commit()``. A
  submission is copied and committed before it is acknowledged. On
  start the copy is restored. ``max_containers=1`` keeps one writer.

The endpoint is a public URL, so a bearer token is required. Create it
once before deploying:

    modal secret create quail-server-token QUAIL_SERVER_TOKEN=<token>

Clients read the same variable, or pass ``token`` to ``ServerClient``.
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
DATA_DIR = VOLUME_DIR / "quail-server"
DB_COPY = DATA_DIR / "quail.sqlite3"
LOCAL_DB = Path("/tmp/quail-server/quail.sqlite3")

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
    secrets=[modal.Secret.from_name("quail-server-token",
                                    required_keys=["QUAIL_SERVER_TOKEN"])],
    timeout=24 * 3600,
    scaledown_window=15 * 60,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
class QuailServer:
    """One server container: web app, scheduler, and the executor child."""

    @modal.enter()
    def start(self) -> None:
        from quail.server.app import ServerSettings, create_app
        from quail.server.checkpoint import Checkpoint, restore

        results_volume.reload()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if restore(DB_COPY, LOCAL_DB):
            print(f"restored {DB_COPY} to {LOCAL_DB}", flush=True)
        settings = ServerSettings(
            data_dir=DATA_DIR,
            db_path=LOCAL_DB,
            models=MODELS,
            device=DEVICE,
            gpus=(GPUS,),
            token=os.environ["QUAIL_SERVER_TOKEN"],
        )
        self.asgi = create_app(settings)
        self.server = self.asgi.state.server
        # the ASGI lifespan starts the scheduler; the checkpoint thread is
        # ours because it needs the Volume
        self.checkpoint = Checkpoint(self.server.store, DB_COPY,
                                     commit=results_volume.commit)
        self.checkpoint.start()
        self.server.add_sync(self.checkpoint.sync)
        self.server.add_closer(self.checkpoint.stop)
        if self.server.recovered:
            print(f"marked interrupted: {self.server.recovered}", flush=True)

    @modal.asgi_app()
    def web(self):
        return self.asgi

    @modal.exit()
    def stop(self) -> None:
        self.checkpoint.stop()
