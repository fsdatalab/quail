"""The comparison backends: stock vLLM, one request per prompt.

These backends exist to measure Quail against; a user runs the
``quail`` backend. ``register_baselines`` adds them to a registry.
"""



def register_baselines(registry):
    """Register the comparison backends and their node's runtime."""
    from .request import request_runtimes
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
    for runtime_key, runtime in request_runtimes().items():
        registry.register_runtime(runtime, key=runtime_key)
    return registry
