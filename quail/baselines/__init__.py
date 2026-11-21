"""The comparison backends: stock vLLM, one request per prompt.

These backends exist to measure Quail against; a user runs the
``quail`` backend. ``register_baselines`` adds them to a registry.
"""



def register_baselines(registry):
    """Register the comparison backends and their node's runtime."""
    from quail.execution.runtimes import ModelNodeRuntime
    from quail.physical import RequestExecution

    from .request import RequestBackend
    from .vllm import (
        DefaultVLLMEngine,
        VLLMEngine,
        ngram_vllm_backend,
        stock_vllm_backend,
    )

    for backend in (
        stock_vllm_backend(),
        RequestBackend(
            name="pipelined_vllm", engine=VLLMEngine(),
            filter_submission="pipelined", answer_submission="pipelined"),
        RequestBackend(
            name="shuffled_vllm", engine=VLLMEngine(),
            filter_submission="operator-at-a-time",
            answer_submission="shuffled"),
        ngram_vllm_backend(),
        RequestBackend(
            name="dumb_vllm", engine=DefaultVLLMEngine(),
            filter_submission="operator-at-a-time"),
    ):
        registry.register_backend(backend)
    registry.register_runtime(
        ModelNodeRuntime(), key=RequestExecution.runtime_key)
    return registry
