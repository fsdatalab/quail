"""Download a registered model's checkpoint into the quail-hf-cache volume.

    MODAL_PROFILE=fsdatalab uv run modal run experiments/cells/fetch_model.py \
        --model qwen3.5-4b-bf16 2>&1 | tee /tmp/fetch_model.log

The checkpoint lands at the spec's pinned revision, where vLLM and the
Quail loader find it without another download.
"""

import modal

try:
    from quail.bench.images import cpu_image
    image = cpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def fetch(model: str) -> dict:
    """Download the checkpoint and list its files."""
    import os

    from huggingface_hub import snapshot_download

    from quail.specs import MODELS

    spec = MODELS[model]
    path = snapshot_download(spec.hf_name, revision=spec.revision or None)
    hf_cache.commit()
    sizes = {name: os.path.getsize(os.path.join(path, name))
             for name in sorted(os.listdir(path))}
    return {"hf_name": spec.hf_name, "revision": spec.revision, "path": path,
            "bytes": sum(sizes.values()), "files": sizes}


@app.local_entrypoint()
def main(model: str = "qwen3.5-4b-bf16"):
    call = fetch.spawn(model)
    print(f"fetch function call id: {call.object_id}", flush=True)
    print(call.get(), flush=True)
