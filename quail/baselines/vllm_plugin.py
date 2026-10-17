"""Quail extensions loaded in each vLLM process."""

from weakref import WeakKeyDictionary


def register() -> None:
    """Register Quail extensions and model cache ownership."""
    from vllm.model_executor.models import interfaces
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers import TokenizerRegistry

    from quail.baselines.vllm_decision import register as register_decision

    # vLLM 0.26's strong keys retain multimodal weights and KV after shutdown.
    cache = interfaces._language_model_by_module
    if not isinstance(cache, WeakKeyDictionary):
        interfaces._language_model_by_module = WeakKeyDictionary(cache)

    tokenizer = ("quail.baselines.vllm", "GigatokenVLLMTokenizer")
    if TokenizerRegistry.tokenizers.get("gigatoken") != tokenizer:
        TokenizerRegistry.register("gigatoken", *tokenizer)
    renderer = ("vllm.renderers.hf", "HfRenderer")
    if RENDERER_REGISTRY.renderers.get("gigatoken") != renderer:
        RENDERER_REGISTRY.register("gigatoken", *renderer)
    register_decision()
