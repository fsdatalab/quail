"""The Modal image, volumes, and functions that run logical queries."""

from dataclasses import dataclass

import modal

from quail.builtins import registry_from_manifest
from quail.catalog import DocumentProvider
from quail.extensions import ExtensionManifest
from quail.planner.plan import EngineConfig
from quail.runtime.compute import QueryRequest
from quail.runtime.local import execute_query_request
from quail.runtime.volumes import hf_cache, kernel_cache, results_vol


def build_worker_image(
    local_python_sources=(),
    pip_packages=(),
    runtime_package="vllm==0.26.0",
):
    """Build the Modal image containing Quail and registered extensions."""
    packages = tuple(dict.fromkeys((
        "sqlglot>=27.0",
        "bpe-qwen>=0.1.5",
        "datasets>=5.0.1",
        *pip_packages,
    )))
    sources = tuple(dict.fromkeys(("quail", *local_python_sources)))
    return (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12"
        )
        .entrypoint([])
        .pip_install(
            runtime_package,
            "huggingface_hub",
            "numpy",
            "pyarrow",
        )
        .pip_install(*packages)
        .env({# vLLM caches its model-architecture inspection (a ~13 s
              # subprocess) under VLLM_CACHE_ROOT; the default location
              # is ephemeral, so it lives on the kernel-cache volume
              "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
              "VLLM_LOGGING_LEVEL": "WARNING",
              "VLLM_USE_FLASHINFER_SAMPLER": "0",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
              "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
              "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
              "TRITON_CACHE_DIR": "/root/.cache/kernels/triton"})
        .add_local_python_source(*sources)
    )


def _execute_logical_query(value, gpu_count: int):
    """Open the query sources and run the query in this container."""
    config_value = value["config"]
    if not isinstance(config_value, EngineConfig):
        raise TypeError("a worker query needs an EngineConfig")
    requested_gpus = int(config_value.gpus)
    if requested_gpus != gpu_count:
        raise ValueError(
            f"query needs {requested_gpus} GPUs but worker has {gpu_count}"
        )
    manifest = ExtensionManifest.from_value(value["extensions"])
    registry = registry_from_manifest(manifest)
    providers = {}
    for name, source in value["sources"].items():
        if "remote" in source:
            provider = registry.open_source(source["remote"])
        elif "table" in source:
            provider = DocumentProvider.from_table(
                source["table"], id_col=str(source["id_col"])
            )
        else:
            raise ValueError(f"query source {name!r} has no location")
        providers[name] = provider
    return execute_query_request(QueryRequest(
        logical_plan=value["logical_plan"],
        providers=providers,
        config=config_value,
        device=str(value["device"]),
        order=value["order"],
        extensions=manifest,
    ))


@dataclass(frozen=True)
class ModalWorker:
    """Modal app and query functions for one extension set."""

    app: object
    execute_1: object
    execute_2: object
    execute_4: object
    execute_8: object

    def function(self, gpu_count: int):
        """Return the function for one supported GPU count."""
        functions = {
            1: self.execute_1,
            2: self.execute_2,
            4: self.execute_4,
            8: self.execute_8,
        }
        try:
            return functions[gpu_count]
        except KeyError as error:
            raise ValueError("Modal supports 1, 2, 4, or 8 GPUs") from error


def modal_worker(
    local_python_sources=(),
    pip_packages=(),
    *,
    secrets=(),
    runtime_package="vllm==0.26.0",
) -> ModalWorker:
    """Return Modal Functions containing requested extensions."""
    local_python_sources = tuple(local_python_sources)
    pip_packages = tuple(pip_packages)
    secrets = tuple(secrets)

    # Every function stays in the existing app so it shares image and volume
    # caches with earlier Quail workers.
    worker_app = modal.App("quail-engine")
    worker_image = build_worker_image(
        local_python_sources,
        pip_packages,
        runtime_package,
    )
    volumes = {
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/kernels": kernel_cache,
        "/results": results_vol,
    }

    def define_function(name, gpu, gpu_count, memory):
        @worker_app.function(
            name=name,
            serialized=True,
            image=worker_image,
            gpu=gpu,
            memory=memory,
            secrets=secrets,
            max_containers=1,
            scaledown_window=300,
            startup_timeout=120,
            timeout=21600,
            volumes=volumes,
        )
        def execute(value):
            result = _execute_logical_query(value, gpu_count)
            return result.collect(), result.report

        return execute

    execute_1 = define_function("execute_1", "H100!", 1, 98304)
    execute_2 = define_function("execute_2", "H100!:2", 2, 131072)
    execute_4 = define_function("execute_4", "H100!:4", 4, 196608)
    execute_8 = define_function("execute_8", "H100!:8", 8, 262144)

    return ModalWorker(
        worker_app,
        execute_1,
        execute_2,
        execute_4,
        execute_8,
    )

