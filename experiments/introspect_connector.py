"""One-off: the KV-connector API surface in pinned vLLM 0.26, for the
fork's partial-block copy. Run: modal run experiments/introspect_connector.py"""
import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
image = (modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
         .entrypoint([]).pip_install("vllm==0.26.0"))
app = modal.App("docengine-introspect2")


@app.function(image=image, timeout=600)
def show() -> str:
    import inspect
    out = []

    def dump(label, obj, limit=5000):
        try:
            src = inspect.getsource(obj)
        except Exception as e:
            src = f"<no source: {e}>"
        out.append(f"===== {label} =====\n{src[:limit]}")

    from vllm.distributed.kv_transfer.kv_connector.v1 import base as b
    dump("KVConnectorBase_V1 methods",
         b.KVConnectorBase_V1, 12000)
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory)
        dump("KVConnectorFactory.register_connector",
             KVConnectorFactory.register_connector, 2000)
    except ImportError as e:
        out.append(f"===== factory import failed: {e} =====")
    from vllm.config import KVTransferConfig
    fields = [f for f in dir(KVTransferConfig) if not f.startswith("_")]
    out.append("===== KVTransferConfig fields =====\n" + str(fields))
    return "\n\n".join(out)


@app.local_entrypoint()
def main():
    print(show.remote())
