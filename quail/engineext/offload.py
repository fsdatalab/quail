"""Quail offloading connector: one transfer per step, not per request.

Boot with

    KVTransferConfig(
        kv_connector="QuailOffloadingConnector",
        kv_connector_module_path="quail.engineext.offload",
        kv_role="kv_both", ...)

Identical to vLLM's OffloadingConnector except on the worker side:
the stock worker submits each request's load as its own transfer job,
and the handler's strict job ordering prices every job at ~1.7 ms of
overhead against ~0.6 ms of copying (measured,
results/engine/persist_xfer_trace.jsonl.gz). This subclass merges all
load jobs the scheduler created in one engine step into a single
batched transfer, then fans the finished transfer back out into
per-job results so scheduler-side bookkeeping and stats see exactly
the jobs it created.

Merging is only descriptor concatenation when the offload chunking is
trivial (one KV group, CPU chunk == GPU block); the decision lives in
offload_logic.mergeable. Jobs that do not qualify are submitted
unmerged, so this class degrades to stock behavior rather than
guessing at offset arithmetic.

Depends on vLLM 0.26.0 internals, pinned in the image for the same
reason the scheduler subclass pins it: OffloadingConnectorWorker and
its handlers are not public API.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, TransferResult

from .offload_logic import plan_merge, split_result, synthetic_ids


class QuailOffloadingConnectorWorker(OffloadingConnectorWorker):
    def __init__(self, spec, kv_cache_config):
        super().__init__(spec, kv_cache_config)
        # synthetic merged-job id -> list of (job_id, n_blocks)
        self._merged: dict[int, list[tuple[int, int]]] = {}
        self._synthetic = synthetic_ids()

    def _src_blocks_per_chunk(self) -> int:
        # the load handler's CPU-side chunking; 1 means CPU chunks and
        # GPU blocks are the same size and merges are offset-free
        return self.worker._load_handler.src_blocks_per_chunk

    def start_kv_transfers(self, metadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            assert self.worker.submit_store(job_id, src_spec, dst_spec)
        self._unsubmitted_store_jobs.clear()

        load_items = metadata.load_jobs
        if len(load_items) < 2:
            for job_id, entry in load_items.items():
                self._load_jobs[job_id] = entry.req_id
                assert self.worker.submit_load(
                    job_id, entry.src_spec, entry.dst_spec)
            return

        shapes = []
        for job_id, entry in load_items.items():
            dst = entry.dst_spec
            assert isinstance(dst, GPULoadStoreSpec)
            shapes.append((job_id, len(entry.src_spec.block_ids),
                           len(dst.block_ids), len(dst.group_sizes)))
        merged_ids, single_ids = plan_merge(
            shapes, self._src_blocks_per_chunk())

        for job_id in single_ids:
            entry = load_items[job_id]
            self._load_jobs[job_id] = entry.req_id
            assert self.worker.submit_load(
                job_id, entry.src_spec, entry.dst_spec)

        if not merged_ids:
            return
        src_blocks, dst_blocks, parts = [], [], []
        src_cls = type(load_items[merged_ids[0]].src_spec)
        for job_id in merged_ids:
            entry = load_items[job_id]
            self._load_jobs[job_id] = entry.req_id
            n = len(entry.src_spec.block_ids)
            src_blocks.extend(int(b) for b in entry.src_spec.block_ids)
            dst_blocks.extend(int(b) for b in entry.dst_spec.block_ids)
            parts.append((job_id, n))
        merged_id = next(self._synthetic)
        self._merged[merged_id] = parts
        # block_indices is only read for partial-first-chunk skips,
        # which mergeable() has ruled out; zero is the aligned value
        merged_dst = GPULoadStoreSpec(
            block_ids=dst_blocks,
            group_sizes=[len(dst_blocks)],
            block_indices=[0])
        assert self.worker.submit_load(
            merged_id, src_cls(src_blocks), merged_dst)

    def get_finished(self, finished_req_ids):
        assert self.worker is not None
        finished_recving: set[str] = set()
        for result in self._expand(self.worker.get_finished()):
            job_id = result.job_id
            assert result.success
            is_load = job_id in self._load_jobs
            if (result.transfer_time is not None
                    and result.transfer_size is not None):
                stats = (self._connector_worker_meta.transfer_stats.load
                         if is_load else
                         self._connector_worker_meta.transfer_stats.store)
                stats.record(result.transfer_size, result.transfer_time)
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)
        return set(), finished_recving

    def _expand(self, results):
        """Fan a merged transfer's result out into its constituents so
        every downstream consumer sees the scheduler's own job ids."""
        out = []
        for result in results:
            parts = self._merged.pop(result.job_id, None)
            if parts is None:
                out.append(result)
                continue
            total_blocks = sum(n for _, n in parts)
            times = split_result(result.transfer_time or 0.0,
                                 [n for _, n in parts])
            for (job_id, n), t in zip(parts, times):
                size = ((result.transfer_size or 0) * n // total_blocks
                        if total_blocks else 0)
                out.append(TransferResult(
                    job_id=job_id, success=result.success,
                    transfer_size=size, transfer_time=t))
        return out


class QuailOffloadingConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            # rebuild the worker as the coalescing subclass; __init__
            # only stores references, so the swap has no side effects
            self.connector_worker = QuailOffloadingConnectorWorker(
                self.connector_worker.spec, kv_cache_config)
