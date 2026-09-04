"""Run logical queries on Modal GPU workers."""

import json
import os
import time
from dataclasses import dataclass, field

import modal

from quail.backends import BackendExecutionContext
from quail.builtins import registry_from_modules
from quail.catalog import DocumentProvider
from quail.execution import PhysicalRequest, PhysicalResponse
from quail.physical import check_plan_envelope, decode_graph, DocumentInput
from quail.planner.plan import EngineConfig, Refusal
from quail.runtime.session import Query, RefusalError, Session
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


@dataclass
class _WorkerRuntime:
    """Per container state that backends keep between queries."""

    booted: dict = field(default_factory=dict)


_RUNTIME = _WorkerRuntime()


def _validate_physical_request(request):
    """Validate and decode a physical execution request."""

    if not isinstance(request, PhysicalRequest):
        raise TypeError("the worker needs a PhysicalRequest")
    envelope = request.plan
    check_plan_envelope(envelope)
    registry = registry_from_modules(tuple(envelope["extension_modules"]))
    backend = registry.backend(envelope["backend"])
    graph = decode_graph(envelope["graph"], registry.codecs)
    graph.validate(runtime_keys=set(registry.runtimes))
    graph.validate_backend(envelope["backend"])
    needed_inputs = {
        node.input_id for node in graph.nodes
        if isinstance(node, DocumentInput)
    }
    missing = needed_inputs - set(request.inputs)
    extra = set(request.inputs) - needed_inputs
    if missing or extra:
        raise ValueError(
            "execution request has wrong input bindings; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return request, registry, graph, backend


def _execute_physical(request):
    """Run the backend selected by a registered physical plan."""

    request, registry, graph, backend = _validate_physical_request(request)
    response = backend.execute_request(BackendExecutionContext(
        request=request,
        graph=graph,
        registry=registry,
        gpu_count=request.gpu_count,
        runtime_state=_RUNTIME.booted,
    ))
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a model backend must return PhysicalResponse")
    if (
        response.metrics.get("result_volume_path") is None
        and os.path.isdir("/results")
        and os.access("/results", os.W_OK)
    ):
        metrics = dict(response.metrics)
        os.makedirs("/results/runs", exist_ok=True)
        result_path = f"/results/runs/run_{time.time_ns()}.json"
        metrics["result_volume_path"] = result_path
        with open(result_path, "w") as output:
            json.dump({
                key: metrics.get(key)
                for key in (
                    "backend",
                    "wall_s",
                    "boot_s",
                    "boot_kind",
                    "boot",
                    "fresh_tokens",
                    "cached_tokens",
                    "regret_tokens",
                    "peak_gib",
                    "node_metrics",
                    "backend_metrics",
                )
            }, output)
        results_vol.commit()
        response = PhysicalResponse(response.outputs, metrics)
    return response


def execute_worker_query(query, physical_executor=None):
    """Execute one query inside its current worker process."""

    plan = query.plan()
    if isinstance(plan, Refusal):
        raise RefusalError(plan)
    plan.graph.validate(runtime_keys=set(query.session.registry.runtimes))
    plan.graph.validate_backend(plan.backend)
    if plan.workers > 8:
        raise NotImplementedError(
            "more than 8 GPUs means multiple containers; the "
            "multi-container coordinator is a later step"
        )
    request = query._prepare_physical()
    started = time.perf_counter()
    response = (physical_executor or _execute_physical)(request)
    if not isinstance(response, PhysicalResponse):
        raise TypeError("a physical executor must return PhysicalResponse")
    return query.finish(response, time.perf_counter() - started)


def _execute_logical_query(value, gpu_count: int):
    """Read query sources, plan the query, and execute it."""

    config_value = value["config"]
    if not isinstance(config_value, EngineConfig):
        raise TypeError("a worker query needs an EngineConfig")
    requested_gpus = int(config_value.gpus)
    if requested_gpus != gpu_count:
        raise ValueError(
            f"query needs {requested_gpus} GPUs but worker has {gpu_count}"
        )
    started = time.perf_counter()
    registry = registry_from_modules(tuple(value["extension_modules"]))
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
    session = Session(
        config_value,
        device=str(value["device"]),
        registry=registry,
    )
    for name, provider in providers.items():
        session.register(name, provider)
    query = Query(session, value["logical_plan"], order=value["order"])
    result = execute_worker_query(query)
    result.report["worker_total_s"] = round(
        time.perf_counter() - started, 4
    )
    return result


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

