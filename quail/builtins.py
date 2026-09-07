"""The registry of Quail's built in backends, models, and runtimes."""

from quail.backends import (
    QuailBackend,
    pipelined_sglang_backend,
    pipelined_vllm_backend,
    stock_vllm_backend,
)
from quail.backends.quail import quail_runtimes
from quail.backends.request import request_runtimes
from quail.catalog import built_in_source_readers
from quail.extensions import ExtensionRegistry
from quail.physical import built_in_codecs
from quail.runtime.runner import built_in_runtimes
from quail.specs import DEVICES, MODELS


def built_in_registry() -> ExtensionRegistry:
    """Create one registry with Quail built ins."""
    registry = ExtensionRegistry()
    for model in MODELS.values():
        registry.register_model(model)
    for device in DEVICES.values():
        registry.register_device(device)
    registry.register_backend(QuailBackend())
    registry.register_backend(stock_vllm_backend())
    registry.register_backend(pipelined_vllm_backend())
    registry.register_backend(pipelined_sglang_backend())
    for codec in built_in_codecs():
        registry.register_codec(codec)
    for runtimes in (built_in_runtimes(), quail_runtimes(), request_runtimes()):
        for runtime_key, runtime in runtimes.items():
            registry.register_runtime(runtime, key=runtime_key)
    for name, reader in built_in_source_readers().items():
        registry.register_source_reader(reader, source_type=name)
    return registry
