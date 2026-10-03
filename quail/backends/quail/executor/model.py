"""Weight loading through vLLM's public surface.

The checkpoint becomes vLLM's processed module - merged qkv and
gate_up, fp8 weights and block scales laid out for DeepGEMM. No
engine, no scheduler, no KV pool. vLLM is a library here (loader and
kernels), nothing more.

After load, only the TRUE/FALSE output rows are retained. The full
output head is discarded; shared input embeddings remain available.

A Decision 2.0 checkpoint keeps its Qwen3 backbone in a subfolder
under its own tensor names. It is converted once into a Qwen3
checkpoint folder beside the Hugging Face cache, with bf16 weights and
its decision head copied alongside, and loads from there.

get_model reads tensor-parallel group objects. Those collectives are
no-ops at world size 1, so this path installs single-rank stubs
instead of starting NCCL or gloo.
"""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

from quail.backends.quail.executor.moe_configs import write_configs
from quail.progress import say


@lru_cache(maxsize=8)
def resolve_model_path(model_name: str, revision: str | None = None) -> str:
    """Download one model revision and return its local directory."""
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from huggingface_hub import snapshot_download

    say(f"preparing model files: {model_name}")
    path = snapshot_download(
        repo_id=model_name, revision=revision or None,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
    )
    say(f"model files ready: {path}")
    return path


# files a converted Decision 2.0 checkpoint keeps from the original;
# vLLM loads every safetensors file at the top level as model weights,
# so the head goes in a subfolder
DECISION2_COPIED = {"tokenizer.json": "tokenizer.json",
                    "tokenizer_config.json": "tokenizer_config.json",
                    "decision_config.json": "decision_config.json",
                    "decision_head.safetensors": "head/decision_head.safetensors"}
# names the converted layout; a new layout converts again
DECISION2_FORMAT = "qwen3-bf16-v1"


def is_decision2(path) -> bool:
    """Whether a checkpoint directory holds a Decision 2.0 package."""
    config = Path(path) / "config.json"
    return (config.is_file()
            and json.loads(config.read_text()).get("model_type") == "decision2")


@lru_cache(maxsize=8)
def checkpoint_path(model_name: str, revision: str | None = None) -> str:
    """The local directory vLLM loads for a model, converted when needed."""
    path = resolve_model_path(model_name, revision)
    if not is_decision2(path):
        return path
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    snapshot = Path(path).resolve()
    dest = (Path(hf_home).expanduser() / "quail-checkpoints" / DECISION2_FORMAT
            / snapshot.parent.parent.name / snapshot.name)
    if not (dest / "model.safetensors").is_file():
        say(f"converting {model_name} to a Qwen3 checkpoint at {dest}")
        convert_decision2(snapshot, dest)
    return str(dest)


def convert_decision2(src, dest) -> None:
    """Write a Decision 2.0 package as a bf16 Qwen3ForCausalLM checkpoint.

    Args:
        src: The package directory, with backbone/config.json and
            backbone/model.safetensors.
        dest: The directory to create. It appears only once complete.
    """
    import torch
    from safetensors.torch import load_file, save_file

    src, dest = Path(src), Path(dest)
    config = json.loads((src / "backbone" / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError(
            f"{src}: the backbone is {config.get('model_type')!r}, not 'qwen3'")
    config.update(architectures=["Qwen3ForCausalLM"], torch_dtype="bfloat16",
                  dtype="bfloat16")
    tensors = {
        f"model.{name}": tensor.to(torch.bfloat16).contiguous()
        for name, tensor in load_file(
            str(src / "backbone" / "model.safetensors")).items()}
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    save_file(tensors, str(tmp / "model.safetensors"))
    (tmp / "head").mkdir()
    for name, copy in DECISION2_COPIED.items():
        shutil.copyfile(src / name, tmp / copy)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


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


def retain_answer_head(torch, model, token_ids):
    """Keep the TRUE/FALSE answer rows and the full output head in bf16."""
    allowed = tuple(sorted(set(token_ids)))
    if not allowed:
        raise ValueError("TRUE/FALSE token ids must not be empty")
    if hasattr(model, "quail_answer_token_ids"):
        answer_weights(model, allowed)
        return
    weight = model.lm_head.weight
    indices = torch.tensor(allowed, device=weight.device, dtype=torch.long)
    weights = weight.detach().index_select(0, indices).to(dtype=torch.bfloat16)
    model.register_buffer("quail_answer_weights", weights, persistent=False)
    model.quail_answer_token_ids = allowed
    # AI.CLASSIFY reads the whole head: bf16 logits, as in vLLM,
    # normalized in float32. A head tied to the embedding shares its
    # memory; the model spec prices an untied head into the KV arena.
    model.quail_full_head = weight.detach().to(dtype=torch.bfloat16)
    # lm_head can be the same module as embed_tokens; drop only this reference.
    model.lm_head = None


def full_output_head(model):
    """Return the whole output head for full-vocabulary log probabilities.

    Raises:
        ValueError: The model booted without keeping its head.
    """
    head = getattr(model, "quail_full_head", None)
    if head is None:
        raise ValueError("this model's full output head was not kept")
    return head


def answer_weights(model, token_ids):
    """Return the retained rows for a query's answer token ids."""
    if tuple(token_ids) != model.quail_answer_token_ids:
        raise ValueError("Query answer token ids differ from the loaded "
                         "model's retained rows")
    return model.quail_answer_weights


def load_model(model_name: str, revision: str | None = None, *,
               answer_token_ids=None, max_batched_tokens=None,
               moe_backend=None):
    """Load model weights and retain the answer rows and the full output head.

    max_batched_tokens is the largest chunk the model will see. vLLM's
    fused MoE kernels size their scratch buffers from it; a dense model
    ignores it. moe_backend is passed through as vLLM's moe_backend
    setting (see ModelSpec.moe_backend); None lets vLLM pick.
    """
    import os
    import tempfile

    import torch
    import vllm
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    # vLLM's Triton fused MoE reads tuned tile configs from this
    # folder before its own; ours add entries for Quail's chunk sizes
    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        base = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
        folder = write_configs(
            Path(tempfile.gettempdir()) / "quail-moe-configs", base)
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(folder)
    model_path = checkpoint_path(model_name, revision)
    args = dict(model=model_path, dtype="auto", enforce_eager=True)
    if max_batched_tokens is not None:
        args["max_num_batched_tokens"] = int(max_batched_tokens)
    if moe_backend is not None:
        args["moe_backend"] = moe_backend
    config = EngineArgs(**args).create_engine_config()
    _install_single_rank_groups(torch)
    with set_current_vllm_config(config):
        model = get_model(vllm_config=config)
    # vLLM's fused MoE kernels read the forward context, which is
    # built from this config
    model.quail_vllm_config = config
    if answer_token_ids is None:
        from gigatoken import Tokenizer

        from quail.logical import true_false_ids

        tokenizer = Tokenizer(model_path).as_hf()
        true_ids, false_ids = true_false_ids(tokenizer)
        answer_token_ids = true_ids | false_ids
    retain_answer_head(torch, model, answer_token_ids)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return model
