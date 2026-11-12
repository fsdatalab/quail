"""Capabilities and entry points for the Quail engine."""

from __future__ import annotations

import os

from quail.backends.base import GpuContext
from quail.backends.quail.executor.models import supported_archs
from quail.backends.quail.executor.runtime import QuailModelExecution
from quail.backends.quail.graph import region_report
from quail.backends.quail.worker import (
    boot_record,
    quail_workers,
    release_quail_request,
)
from quail.specs import DEVICES, SupportResult


class QuailBackend:
    """Start Quail model execution; the Rust planner plans its queries."""

    name = "quail"
    engine_name = "quail"
    runtime_package = "vllm==0.29.0"
    reads_document_text = False
    locates_answers_in_text = True

    def supports(self, model, device, gpus: int) -> SupportResult:
        if device.name not in DEVICES:
            return SupportResult.reject(
                f"Quail does not support device {device.name!r}")
        if model.arch not in supported_archs():
            return SupportResult.reject(
                f"Quail does not support model {model.name!r}: "
                f"no forward pass for architecture {model.arch!r}")
        if gpus not in {1, 2, 4, 8}:
            return SupportResult.reject(
                "Quail requires 1, 2, 4, or 8 GPUs")
        prefill = os.environ.get("QUAIL_GDN_PREFILL", "auto")
        if prefill not in ("auto", "triton", "flashinfer"):
            return SupportResult.reject(
                f"QUAIL_GDN_PREFILL={prefill!r} is not triton, flashinfer, "
                f"or auto")
        if (prefill == "flashinfer"
                and "gated_delta_net" in model.layer_kinds
                and device.name != "h100-sxm"):
            return SupportResult.reject(
                f"QUAIL_GDN_PREFILL=flashinfer needs a Hopper GPU, and "
                f"device {device.name!r} is not one; use triton or auto "
                f"for {model.name!r}")
        return SupportResult.accept()

    def start(self, context: GpuContext) -> QuailModelExecution:
        return QuailModelExecution(context)

    def workers(self, config):
        """Return the GPUs' workers for a session's configuration."""
        return quail_workers(config.model, config.device, config.backend,
                             config.gpus)

    def report_region(self, plan, ran: list, boots: list, seconds: float,
                      device) -> dict:
        """Report a region's run on workers that Rust drove.

        Args:
            plan: The region's physical plan.
            ran: Per worker, what it ran: its counters as JSON and its
                nodes' output tables.
            boots: Each worker's boot record.
            seconds: Seconds the region took on the workers.
            device: The GPUs' device spec.

        Returns:
            The run's report, with its outputs under "_outputs".
        """
        report = region_report(plan.graph, ran, seconds, device, gpus=plan.gpus,
                               gpu_timing=plan.settings.get("gpu_timing", False))
        boot = boot_record(boots)
        report.update(boot_s=boot["boot_s"], boot_kind=boot["kind"], boot=boot)
        return report

    def release_request(self, state) -> None:
        """Release the GPU processes and the model loaded for one configuration."""
        release_quail_request(state)
