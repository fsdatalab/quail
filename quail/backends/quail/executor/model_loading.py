"""Weight loading through vLLM's public surface.

The checkpoint becomes vLLM's processed module - merged qkv and
gate_up, fp8 weights and block scales laid out for DeepGEMM. No
engine, no scheduler, no KV pool. vLLM is a library here (loader and
kernels), nothing more.

After load, only the TRUE/FALSE output rows are retained. The full
output head is discarded; shared input embeddings remain available.

A Decision 2.0 checkpoint keeps its Qwen3 backbone in a subfolder
under its own tensor names; `engine_args` points vLLM at it.

get_model reads tensor-parallel group objects. Those collectives are
no-ops at world size 1, so this path installs single-rank stubs
instead of starting NCCL or gloo.
"""

import json
from pathlib import Path

from quail.backends.quail.executor.models.inkling2 import (
    is_inkling2,
    load_inkling2,
)
from quail.backends.quail.executor.models.lfm3 import is_lfm3, load_lfm3
from quail.backends.quail.executor.moe_configs import write_configs
from quail.specs.files import resolve_model_path


def is_decision2(path) -> bool:
    """Whether a checkpoint directory holds a Decision 2.0 package."""
    config = Path(path) / "config.json"
    return (config.is_file()
            and json.loads(config.read_text()).get("model_type") == "decision2")


def engine_args(path, architecture: str = "") -> dict:
    """Return the vLLM EngineArgs that load the checkpoint at path.

    A Decision 2.0 package keeps its Qwen3 backbone in backbone/, in
    fp32 and without the `model.` tensor prefix; vLLM loads it as bf16
    through a class that maps the names, which the caller registers
    with `decision2.register_backbone()`. The tokenizer stays at the
    package root.

    A checkpoint whose text config names another dtype than its top
    level loads in the text config's dtype.

    Args:
        path: The local checkpoint directory.
        architecture: The registered class for a Decision 2.0 package;
            empty uses the generative backbone class.
    """
    if is_decision2(path):
        from quail.backends.quail.executor.models.decision2 import (
            BACKBONE_ARCHITECTURE,
        )

        return {"model": str(Path(path) / "backbone"),
                "tokenizer": str(path), "dtype": "bfloat16",
                "hf_overrides": {"architectures": [
                    architecture or BACKBONE_ARCHITECTURE]}}
    settings = {"model": str(path)}
    config = Path(path) / "config.json"
    if not config.is_file():
        return settings
    # mammoth-1's config says float32 at the top and bfloat16, the
    # weights' dtype, in its text config; vLLM's "auto" reads the top
    fields = json.loads(config.read_text())
    text_dtype = fields.get("text_config", {}).get("dtype")
    if text_dtype and fields.get("dtype", text_dtype) != text_dtype:
        settings["dtype"] = text_dtype
    return settings


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
        self.device_group = self

    def size(self):
        return self.world_size

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
    ps._EP = group
    ps._PCP = group
    ps._DCP = group
    ps._NODE_COUNT = 1


def retain_answer_head(torch, model, token_ids, head=None):
    """Keep the TRUE/FALSE answer rows and the full output head in bf16.

    Args:
        torch: The torch module.
        model: The loaded model.
        token_ids: The answer token ids whose rows to keep.
        head: The output head's weights, for a head the loaded module
            stores packed; None reads the module's weight.
    """
    allowed = tuple(sorted(set(token_ids)))
    if not allowed:
        raise ValueError("TRUE/FALSE token ids must not be empty")
    if hasattr(model, "quail_answer_token_ids"):
        answer_weights(model, allowed)
        return
    # a checkpoint with a vision tower keeps its head in the language model
    owner = model if hasattr(model, "lm_head") else model.language_model
    weight = owner.lm_head.weight if head is None else head
    indices = torch.tensor(allowed, device=weight.device, dtype=torch.long)
    weights = weight.detach().index_select(0, indices).to(dtype=torch.bfloat16)
    model.register_buffer("quail_answer_weights", weights, persistent=False)
    model.quail_answer_token_ids = allowed
    # AI.CLASSIFY reads the whole head: bf16 logits, as in vLLM,
    # normalized in float32. A head tied to the embedding shares its
    # memory; the model spec prices an untied head into the KV arena.
    model.quail_full_head = weight.detach().to(dtype=torch.bfloat16)
    # lm_head can be the same module as embed_tokens; drop only this reference.
    owner.lm_head = None


# the 16 FP4 E2M1 values, by their 4-bit code: sign bit, then magnitude
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
NVFP4_GROUP = 16


def dequantize_nvfp4(torch, packed, scale, global_scale, rows_per_step=8192):
    """Return NVFP4 weights in bf16.

    Args:
        torch: The torch module.
        packed: The (rows, columns / 2) uint8 weights, two E2M1 codes
            per byte, the even column in the low four bits.
        scale: The (rows, columns / 16) fp8 scale of each 16 columns.
        global_scale: The weight's float32 scale.
        rows_per_step: Rows dequantized at once, which bounds the
            float32 scratch.
    """
    rows = packed.shape[0]
    columns = 2 * packed.shape[1]
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    weights = torch.empty(rows, columns, dtype=torch.bfloat16, device=packed.device)
    global_scale = global_scale.to(device=packed.device, dtype=torch.float32)
    for start in range(0, rows, rows_per_step):
        block = packed[start:start + rows_per_step]
        codes = torch.stack((block & 0xF, block >> 4), dim=-1).reshape(len(block), -1)
        scales = scale[start:start + rows_per_step].float() * global_scale
        weights[start:start + len(block)] = (
            values[codes.long()].view(len(block), -1, NVFP4_GROUP)
            * scales.unsqueeze(-1)).view(len(block), columns)
    return weights


def nvfp4_output_head(torch, model_path, device="cuda"):
    """Return a ModelOpt checkpoint's NVFP4 output head in bf16.

    Args:
        torch: The torch module.
        model_path: The local checkpoint directory.
        device: The device to read and dequantize the head on.

    Returns:
        The (vocab, hidden) head, or None when the checkpoint does not
        store its head in NVFP4.
    """
    from safetensors import safe_open

    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text())
    layers = (config.get("quantization_config") or {}).get("quantized_layers", {})
    if layers.get("lm_head", {}).get("quant_algo", "").upper() != "NVFP4":
        return None
    index = model_path / "model.safetensors.index.json"
    files = json.loads(index.read_text())["weight_map"] if index.is_file() else {}
    tensors = []
    for name in ("lm_head.weight", "lm_head.weight_scale", "lm_head.weight_scale_2"):
        file = model_path / files.get(name, "model.safetensors")
        with safe_open(file, framework="pt", device=device) as checkpoint:
            tensors.append(checkpoint.get_tensor(name))
    return dequantize_nvfp4(torch, *tensors)


def full_output_head(model):
    """Return the whole output head for full-vocabulary log probabilities.

    Raises:
        ValueError: The model booted without keeping its head.
    """
    head = getattr(model, "quail_full_head", None)
    if head is None:
        raise ValueError("this model's full output head was not kept")
    return head


def logit_softcap(model) -> float:
    """The cap of the model's output logits, or 0 for uncapped logits."""
    return float(getattr(model, "quail_logit_softcap", 0.0))


def cap_logits(logits, cap):
    """Return cap * tanh(logits / cap), the logits a capped model reads."""
    return logits if not cap else cap * (logits / cap).tanh()


def answer_weights(model, token_ids):
    """Return the retained rows for a query's answer token ids."""
    if tuple(token_ids) != model.quail_answer_token_ids:
        raise ValueError("Query answer token ids differ from the loaded "
                         "model's retained rows")
    return model.quail_answer_weights


def keep_layers(torch, model, num_layers: int):
    """Keep the first num_layers decoder layers in the forward pass.

    The whole checkpoint is loaded first, then the module list is cut:
    vLLM's load_weights maps every checkpoint tensor onto a parameter,
    so a module list shorter than the checkpoint fails the load. The
    cut does not release the dropped layers' memory: measured on
    Qwen3.5-2B-Base, the memory after the cut is the full model's, so
    something still references their weights.

    Raises:
        ValueError: The checkpoint has fewer than num_layers layers.
    """
    text = getattr(model, "language_model", model)
    layers = text.model.layers
    if len(layers) < num_layers:
        raise ValueError(
            f"the checkpoint has {len(layers)} layers; cannot keep "
            f"{num_layers}")
    if len(layers) == num_layers:
        return
    text.model.layers = layers[:num_layers]
    torch.cuda.empty_cache()


def load_model(model_name: str, revision: str | None = None, *,
               answer_token_ids, max_batched_tokens=None,
               moe_backend=None, text_only=False, images=False,
               num_layers=None, quantize_fp8=False, quantize_fp4=False):
    """Load model weights and retain the answer rows and the full output head.

    max_batched_tokens is the largest chunk the model will see. vLLM's
    fused MoE kernels size their scratch buffers from it; a dense model
    ignores it. moe_backend is passed through as vLLM's moe_backend
    setting (see ModelSpec.moe_backend); None lets vLLM pick. text_only
    takes no video input, for a checkpoint with a vision tower, and no
    image input unless images is True, which loads its vision encoder.
    num_layers keeps only the first decoder layers after the load, for a
    spec that reads an intermediate layer's residual stream.
    quantize_fp8 quantizes a bf16 LFM3 checkpoint's linear weights to fp8
    as they load; vLLM's checkpoints carry their own precision.
    quantize_fp4 quantizes an Inkling checkpoint's bf16 attention
    projections to NVFP4 as they load.
    """
    import os
    import tempfile

    import torch
    import vllm

    # vLLM's Triton fused MoE reads tuned tile configs from this
    # folder before its own; ours add entries for Quail's chunk sizes
    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        base = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
        folder = write_configs(
            Path(tempfile.gettempdir()) / "quail-moe-configs", base)
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(folder)
    model_path = resolve_model_path(model_name, revision)
    if is_lfm3(model_path):
        # vLLM has no LFM3 model; its loader would also refuse the config
        model = load_lfm3(model_path, quantize_fp8=quantize_fp8)
    elif is_inkling2(model_path):
        # vLLM 0.29 has no Inkling model
        model = load_inkling2(model_path, quantize_fp4=quantize_fp4)
    else:
        model = _load_vllm_model(model_path, max_batched_tokens, moe_backend,
                                 text_only, images)
    if num_layers is not None:
        keep_layers(torch, model, num_layers)
    retain_answer_head(torch, model, answer_token_ids,
                       head=nvfp4_output_head(torch, model_path))
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return model


def _load_vllm_model(model_path, max_batched_tokens, moe_backend, text_only,
                     images=False):
    """Load a checkpoint through vLLM's model loader, with no engine."""
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    if is_decision2(model_path):
        from quail.backends.quail.executor.models.decision2 import (
            register_backbone,
        )

        register_backbone()
    args = {"dtype": "auto", "enforce_eager": True} | engine_args(model_path)
    if max_batched_tokens is not None:
        args["max_num_batched_tokens"] = int(max_batched_tokens)
    if moe_backend is not None:
        args["moe_backend"] = moe_backend
    if text_only:
        # an image limit of 0 leaves the vision encoder out
        args["limit_mm_per_prompt"] = {"image": int(images), "video": 0}
    config = EngineArgs(**args).create_engine_config()
    _install_single_rank_groups(torch)
    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    # vLLM's fused MoE kernels read the forward context, which is
    # built from this config
    model.quail_vllm_config = config
    return model
