"""Weight loading through vLLM's public surface: the checkpoint as
vLLM's processed module - merged qkv and gate_up, fp8 weights and
block scales laid out for DeepGEMM. No engine, no scheduler, no KV
pool. vLLM is a library here (loader and kernels), nothing more.

After load, an untied lm_head weight moves to CPU memory
(move_untied_head_to_host): the engine only ever reads its TRUE/FALSE
rows, and the freed GPU bytes go to the KV arena.

get_model reads tensor-parallel group objects. Those collectives are
no-ops at world size 1, so this path installs single-rank stubs
instead of starting NCCL or gloo."""


class _SingleRank:
    """The group object get_model reads. Collectives are identity."""

    world_size = 1
    rank = 0
    rank_in_group = 0
    local_rank = 0
    ranks = [0]
    first_rank = 0
    last_rank = 0
    device_index = 0
    cpu_group = None
    device_group = None
    device_communicator = None
    unique_name = "single"
    torch_distributed_backend = None

    def __init__(self, torch):
        self.device = torch.device("cuda:0")

    @property
    def is_first_rank(self):
        return True

    @property
    def is_last_rank(self):
        return True

    def all_reduce(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x

    def reduce_scatter(self, x, dim=-1):
        return x

    def gather(self, x, dst=0, dim=-1):
        return x

    def broadcast(self, x, src=0):
        return x

    def broadcast_object(self, obj=None, src=0):
        return obj

    def destroy(self):
        return None

    def prepare_communication_buffer_for_model(self, model):
        return None

    def graph_capture(self, graph_capture_context=None):
        from contextlib import nullcontext
        return nullcontext(graph_capture_context)


def _install_single_rank_groups(torch):
    from vllm.distributed import parallel_state as ps
    if ps._TP is not None:
        return
    group = _SingleRank(torch)
    ps._WORLD = group
    ps._INNER_DP_WORLD = group
    ps._TP = group
    ps._PP = group
    ps._DP = group
    ps._PCP = group
    ps._DCP = group
    ps._NODE_COUNT = 1


def move_untied_head_to_host(torch, model):
    """Move an untied lm_head weight to CPU memory, freeing GPU memory.

    The engine reads at most a dozen TRUE/FALSE rows of the head (the
    answerers slice them out), so the full-vocabulary matrix never
    earns its place on the GPU. A tied head shares the input
    embedding's tensor and stays. The freed bytes go to the KV arena:
    budgets subtract ModelSpec.head_mem_bytes from resident weights.

    Returns:
        Bytes freed on the GPU; 0 when the head is tied or already
        on the CPU.
    """
    weight = model.lm_head.weight
    if weight.device.type != "cuda":
        return 0
    if weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        return 0
    freed = weight.numel() * weight.element_size()
    model.lm_head.weight = torch.nn.Parameter(
        weight.detach().to("cpu"), requires_grad=False)
    del weight
    torch.cuda.empty_cache()
    return freed


def load_model(model_name: str):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    config = EngineArgs(model=model_name, dtype="auto",
                        enforce_eager=True).create_engine_config()
    _install_single_rank_groups(torch)
    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    move_untied_head_to_host(torch, model)
    torch.cuda.synchronize()
    return model
