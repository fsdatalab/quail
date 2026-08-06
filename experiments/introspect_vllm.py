"""One-off: print the pinned vLLM 0.26 internals the fork surgery
touches, so the sibling construction matches reality. Run:
modal run experiments/introspect_vllm.py"""
import modal

IMAGE_BASE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .pip_install("vllm==0.26.0")
)

app = modal.App("docengine-introspect")


@app.function(image=image, timeout=600)
def show() -> str:
    import inspect

    out = []

    def dump(label, obj, limit=7000):
        try:
            src = inspect.getsource(obj)
        except Exception as e:
            src = f"<no source: {e}>"
        out.append(f"===== {label} =====\n{src[:limit]}")

    from vllm.v1.request import Request
    dump("Request.__init__", Request.__init__)
    for name in ("append_output_token_ids", "get_hash_new_full_blocks",
                 "block_hashes"):
        if hasattr(Request, name):
            dump(f"Request.{name}", getattr(Request, name), 2500)
    from vllm.v1.core import block_pool as bp
    dump("BlockPool.cache_full_blocks", bp.BlockPool.cache_full_blocks,
         4000)
    import vllm.v1.engine as ve
    if hasattr(ve, "EngineCoreOutput"):
        try:
            out.append("===== EngineCoreOutput fields =====\n"
                       + str(ve.EngineCoreOutput.__struct_fields__))
        except Exception:
            dump("EngineCoreOutput", ve.EngineCoreOutput, 2500)
    from vllm.v1.core.sched.scheduler import Scheduler
    for name in ("_free_request",):
        if hasattr(Scheduler, name):
            dump(f"Scheduler.{name}", getattr(Scheduler, name), 2500)
    # the partial-block KV copy surgery: the hit lookup, allocation,
    # the block wrapper type, manual block acquisition, and the
    # compiled whole-block copy op
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    for name in ("get_computed_blocks", "allocate_slots"):
        if hasattr(KVCacheManager, name):
            dump(f"KVCacheManager.{name}", getattr(KVCacheManager, name),
                 9000)
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks
        dump("KVCacheBlocks", KVCacheBlocks, 3000)
    except ImportError:
        out.append("===== KVCacheBlocks: not in kv_cache_manager =====")
    from vllm.v1.core import block_pool as bp
    for name in ("get_new_blocks",):
        dump(f"BlockPool.{name}", getattr(bp.BlockPool, name), 2500)
    import torch
    out.append("===== cache copy op =====\n"
               + str(hasattr(torch.ops, "_C_cache_ops") and
                     hasattr(torch.ops._C_cache_ops, "copy_blocks")))
    try:
        from vllm.config import get_current_vllm_config  # noqa: F401
        out.append("===== get_current_vllm_config importable =====\nTrue")
    except ImportError:
        out.append("===== get_current_vllm_config =====\nFalse")
    return "\n\n".join(out)


@app.local_entrypoint()
def main():
    print(show.remote())
