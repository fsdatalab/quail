"""Release loaded engines before another configuration uses the GPUs."""

import gc


def _release_vllm_parallel_state() -> None:
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
    except ImportError:
        return
    destroy_model_parallel()
    destroy_distributed_environment()


def freeze_loaded_objects() -> None:
    """Collect garbage, then leave every object alive now out of later collections.

    Python's full collections walk every tracked object. After a model
    loads, the objects of the model, its kernels, and their libraries
    stay alive for the process's life; frozen, a full collection that
    falls inside a query walks only the objects made since.
    """
    gc.collect()
    gc.freeze()


def release_models(state: dict) -> None:
    """Close every engine in state and return its CUDA memory to the driver.

    Args:
        state: The runtime state of one backend configuration; each
            value has a close method. The dict is emptied.
    """
    import torch

    initialized = torch.cuda.is_initialized()
    if initialized:
        torch.cuda.synchronize()
    for key in tuple(state):
        state[key].close()
        del state[key]
    _release_vllm_parallel_state()
    # the released engines' objects may be frozen
    gc.unfreeze()
    gc.collect()
    if initialized:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
