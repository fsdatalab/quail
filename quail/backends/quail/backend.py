"""Capabilities and entry points for the Quail engine."""

from typing import Any

from quail.backends.base import GpuContext
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.runtime import QuailModelExecution
from quail.backends.quail.lowering import plan_region
from quail.backends.quail.worker import execute_quail_request, prepare_quail_request
from quail.planner.optimizer.candidates import (
    ModelRegion,
    PhysicalCandidate,
    SupportResult,
)
from quail.planner.optimizer.context import PlanningContext


class QuailBackend:
    """Plan and start Quail model execution."""

    name = "quail"
    runtime_package = "vllm==0.26.0"
    reads_document_text = False
    locates_answers_in_text = True

    def supports(self, model, device, gpu_count: int) -> SupportResult:
        if device.name not in {"h100-sxm", "rtx-pro-6000-blackwell-server"}:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if model.arch not in supported_archs():
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}: "
                f"no forward pass for architecture {model.arch!r}")
        if gpu_count not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
        return SupportResult.accept()

    def plan(self, region: ModelRegion,
             context: PlanningContext) -> tuple[PhysicalCandidate, ...]:
        return plan_region(region, context)

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)

    def prepare_request(self, context) -> None:
        """Boot the GPU for a request before its documents are ready."""
        prepare_quail_request(context)

    def execute_request(self, context) -> Any:
        """Run one Quail request inside a compute process."""
        return execute_quail_request(context)
