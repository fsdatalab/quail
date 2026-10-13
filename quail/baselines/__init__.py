"""The comparison backends: stock vLLM and SGLang, one request per prompt.

These backends exist to measure Quail against; a user runs the
``quail`` backend. ``register_baselines`` adds them to a registry.
"""

from .nodes import (
    RequestClassifySpec,
    RequestExecution,
    RequestExtractSpec,
    RequestFilterSpec,
    RequestJoinSpec,
)
from .request import RequestBackend, RequestModelExecution, request_runtimes
from .sglang import SGLangClient, SGLangEngine, pipelined_sglang_backend
from .vllm import (
    DefaultVLLMEngine,
    VLLMClient,
    VLLMEngine,
    dumb_vllm_backend,
    pipelined_vllm_backend,
    stock_vllm_backend,
)

__all__ = [
    "DefaultVLLMEngine",
    "RequestBackend",
    "RequestClassifySpec",
    "RequestExecution",
    "RequestExtractSpec",
    "RequestFilterSpec",
    "RequestJoinSpec",
    "RequestModelExecution",
    "SGLangClient",
    "SGLangEngine",
    "VLLMClient",
    "VLLMEngine",
    "dumb_vllm_backend",
    "pipelined_sglang_backend",
    "pipelined_vllm_backend",
    "register_baselines",
    "request_runtimes",
    "stock_vllm_backend",
]


def register_baselines(registry):
    """Register the comparison backends, their node, and their runtimes."""
    from quail.physical import NodeCodec

    for backend in (stock_vllm_backend(), pipelined_vllm_backend(),
                    dumb_vllm_backend(), pipelined_sglang_backend()):
        registry.register_backend(backend)
    registry.register_codec(NodeCodec(RequestExecution))
    for runtime_key, runtime in request_runtimes().items():
        registry.register_runtime(runtime, key=runtime_key)
    return registry
