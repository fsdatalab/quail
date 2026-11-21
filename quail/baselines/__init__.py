"""The comparison backends: stock vLLM, one request per prompt.

These backends exist to measure Quail against; a user runs the
``quail`` backend. ``register_baselines`` adds them to a registry.
"""



def register_baselines(registry):
    """Register the comparison backends and their node's runtime."""
    from quail.execution.runtimes import ModelNodeRuntime
    from quail.physical import RequestExecution

    from .vllm import (
        dumb_vllm_backend,
        ngram_vllm_backend,
        pipelined_vllm_backend,
        shuffled_vllm_backend,
        stock_vllm_backend,
    )

    for backend in (stock_vllm_backend(), pipelined_vllm_backend(),
                    shuffled_vllm_backend(), ngram_vllm_backend(),
                    dumb_vllm_backend()):
        registry.register_backend(backend)
    registry.register_runtime(
        ModelNodeRuntime(), key=RequestExecution.runtime_key)
    return registry
