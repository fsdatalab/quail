"""Weight loading through vLLM's public surface: the checkpoint as
vLLM's processed module - merged qkv and gate_up, fp8 weights and
block scales laid out for DeepGEMM. No engine, no scheduler, no KV
pool. vLLM is a library here (loader and kernels), nothing more."""


def load_model(model_name: str, backend: str = "nccl"):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.utils.network_utils import get_open_port

    config = EngineArgs(model=model_name, dtype="auto",
                        enforce_eager=True).create_engine_config()
    with set_current_vllm_config(config):
        import torch.distributed as dist
        if not dist.is_initialized():
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
                local_rank=0, backend=backend)
            ensure_model_parallel_initialized(1, 1)
        model = get_model(vllm_config=config)
    torch.cuda.synchronize()
    return model
