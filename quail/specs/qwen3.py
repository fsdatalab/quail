"""Model specs for Qwen3 checkpoints, read from their config.json."""

import json
from dataclasses import replace
from pathlib import Path

from .base import ModelSpec

# The FP8 layout the executor's DeepGEMM path reads.
FP8_BLOCK = [128, 128]


def qwen3_spec(name: str, hf_name: str, revision: str = "", *,
               config: dict | str | Path | None = None,
               config_file: str = "config.json", **fields) -> ModelSpec:
    """Build a ModelSpec for a Qwen3 checkpoint from its config.json.

    For example, imagine registering a Qwen3 fine-tune that Quail does
    not ship a spec for:

        spec = qwen3_spec("my-qwen3-1.7b", "Org/My-Qwen3-1.7B",
                          revision="<commit hash>")
        registry.register_model(spec)

    `params` counts the weights every token multiplies: the attention
    and MLP projections and the norms, without the embedding table or
    the output head, which only reads its TRUE and FALSE rows.
    `w_mem_bytes` adds the bf16 embedding table, and the output head
    when it is not tied to the embeddings, to the projection bytes.

    Args:
        name: The `model=` value the spec is registered under.
        hf_name: The Hugging Face repo id of the checkpoint.
        revision: The pinned hub commit hash; "" tracks the default branch.
        config: The parsed config.json, or a path to it. None downloads
            `config_file` from `hf_name` at `revision`.
        config_file: The config's path inside the repo.
        **fields: ModelSpec fields to set after the derived ones, such
            as a measured `w_mem_bytes` or a `chunk_cap_tokens`.

    Returns:
        The ModelSpec.

    Raises:
        ValueError: The config is not a Qwen3 decoder, slides its
            attention window, stores a dtype other than bfloat16, or
            quantizes weights in a layout the executor does not read.
    """
    config = _read_config(hf_name, revision, config, config_file)
    if config.get("model_type") != "qwen3":
        raise ValueError(
            f"{hf_name}: model_type is {config.get('model_type')!r}, not 'qwen3'")
    if config.get("use_sliding_window"):
        raise ValueError(f"{hf_name}: sliding-window Qwen3 is not supported")
    weight_precision, w_bytes = _weight_precision(hf_name, config)
    layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    n_q = config["num_attention_heads"]
    n_kv = config["num_key_value_heads"]
    d_head = config.get("head_dim") or hidden // n_q
    intermediate = config["intermediate_size"]
    per_layer = (hidden * (n_q + 2 * n_kv) * d_head    # qkv_proj
                 + n_q * d_head * hidden                # o_proj
                 + 3 * hidden * intermediate            # gate_up, down
                 + 2 * hidden + 2 * d_head)             # four norms
    params = float(layers * per_layer + hidden)
    tied_head = bool(config.get("tie_word_embeddings", False))
    # vLLM keeps the embedding table and output head in bf16
    embedding_bytes = config["vocab_size"] * hidden * 2
    spec = ModelSpec(
        name=name,
        params=params,
        layers=layers,
        hidden=hidden,
        n_q=n_q,
        n_kv=n_kv,
        d_head=d_head,
        ffn_width=2 * intermediate,
        w_bytes=w_bytes,
        hf_name=hf_name,
        revision=revision,
        vocab=config["vocab_size"],
        tied_head=tied_head,
        w_mem_bytes=(params * w_bytes
                     + embedding_bytes * (1 if tied_head else 2)),
        weight_precision=weight_precision,
        arch="qwen3",
    )
    return replace(spec, **fields)


def _read_config(hf_name, revision, config, config_file) -> dict:
    if isinstance(config, dict):
        return config
    if config is None:
        from huggingface_hub import hf_hub_download

        config = hf_hub_download(hf_name, config_file, revision=revision or None)
    return json.loads(Path(config).read_text())


def _weight_precision(hf_name, config) -> tuple[str, float]:
    dtype = config.get("torch_dtype") or config.get("dtype") or "bfloat16"
    if dtype != "bfloat16":
        raise ValueError(
            f"{hf_name}: the executor runs bfloat16 activations; the "
            f"checkpoint stores {dtype}")
    quant = config.get("quantization_config")
    if quant is None:
        return "bf16", 2.0
    if (quant.get("quant_method") != "fp8"
            or quant.get("weight_block_size") != FP8_BLOCK):
        raise ValueError(
            f"{hf_name}: the executor reads bf16 weights or fp8 weights in "
            f"{FP8_BLOCK} blocks, not {quant.get('quant_method')!r} with "
            f"block size {quant.get('weight_block_size')}")
    return "fp8", 1.0
