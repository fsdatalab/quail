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
    gc.collect()
    if initialized:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
