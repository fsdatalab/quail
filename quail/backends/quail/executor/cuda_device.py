"""The CUDA device implementation, over torch.

Chunk inputs go to the GPU through pinned memory without blocking, CUDA
events time each forward pass, and the KV pools are torch tensors. The
forward passes (executor.models), the kernels (executor.attention), and
the readouts (executor.readout) are the rest of the implementation.
"""

import numpy as np

from quail.backends.quail.executor.device import DeviceImplementation
from quail.progress import logger


class InputStaging:
    """Reuse CPU and GPU input buffers on the current CUDA stream."""

    def __init__(self, torch):
        self.torch = torch
        self.buffers = {}
        self.fixed_tokens = {}

    def host(self, name, count, dtype):
        torch = self.torch
        previous = self.buffers.get(name)
        if previous is not None:
            host, device, event = previous
            # The CPU must not overwrite a buffer while its transfer is pending.
            event.synchronize()
        if previous is None or host.numel() < count or host.dtype != dtype:
            host = torch.empty(count, dtype=dtype, pin_memory=True)
            device = torch.empty(count, dtype=dtype, device="cuda")
            event = torch.cuda.Event()
        self.buffers[name] = host, device, event
        return host[:count]

    def upload(self, name, count):
        host, device, event = self.buffers[name]
        # Reusing the device buffer is ordered after its previous readers.
        device[:count].copy_(host[:count], non_blocking=True)
        event.record()
        return device[:count]

    def copy(self, name, data, dtype):
        source = self.torch.as_tensor(data)
        host = self.host(name, source.numel(), dtype)
        host.copy_(source.reshape(-1))
        return self.upload(name, source.numel())

    def fixed(self, tokens):
        key = id(tokens)
        if key not in self.fixed_tokens:
            self.fixed_tokens[key] = tokens, self.torch.as_tensor(tokens)
        return self.fixed_tokens[key][1]


class TorchKVPools:
    """One K pool and one V pool of KV pages per layer, as torch tensors.

    A layer's pool has shape (pages * page tokens, KV heads, head dim).

    Args:
        torch: The torch module.
        dtype: Torch dtype of K and V; bf16 when omitted.
        device: Torch device the pools live on.
    """

    def __init__(self, torch, dtype=None, device="cuda"):
        self.torch = torch
        self.dtype = dtype or torch.bfloat16
        self.device = device
        self.page_tokens = None
        self.k = self.v = None

    def build(self, page_tokens: int, layers) -> None:
        """Allocate the pools; nothing of earlier pools survives.

        Args:
            page_tokens: Tokens per page.
            layers: Per layer, (pages, KV heads, head dim).
        """
        torch = self.torch
        # the earlier pools go before the new ones are allocated
        self.k = self.v = None
        self.page_tokens = page_tokens
        k, v = [], []
        for pages, heads, dim in layers:
            shape = (pages * page_tokens, heads, dim)
            k.append(torch.empty(shape, dtype=self.dtype, device=self.device))
            v.append(torch.empty(shape, dtype=self.dtype, device=self.device))
        self.k, self.v = k, v

    def layer_kv(self, layer: int):
        """Flat K and V pools of one layer, shape (rows, n_kv, d_head)."""
        return self.k[layer], self.v[layer]

    def paged_kv(self, layer: int):
        """Pools as (n_pages, page_tokens, n_kv, d_head) for paged attention."""
        n_kv, d = self.k[layer].shape[-2], self.k[layer].shape[-1]
        shape = (-1, self.page_tokens, n_kv, d)
        return self.k[layer].view(shape), self.v[layer].view(shape)


class CudaImplementation(DeviceImplementation):
    """Chunk staging, timing, pools, and readouts on a CUDA GPU."""

    name = "cuda"

    def __init__(self):
        import torch

        self.torch = torch

    def kv_pools(self, dtype=None) -> TorchKVPools:
        """Return unbuilt KV pools on the GPU."""
        return TorchKVPools(self.torch, dtype)

    # ---- chunk inputs

    def input_staging(self) -> InputStaging:
        return InputStaging(self.torch)

    def stage(self, values, dtype, name=None, staging=None):
        data = self.torch.as_tensor(np.asarray(values, dtype=dtype))
        if staging is not None and name is not None:
            return staging.copy(name, data, data.dtype).view(data.shape)
        # pageable copies block the CPU behind the running stream
        return data.pin_memory().to("cuda", non_blocking=True)

    def stage_tokens(self, ids, staging=None):
        torch = self.torch
        total = len(ids)
        if staging is None:
            host = torch.empty(total, dtype=torch.int64, pin_memory=True)
            host.copy_(torch.from_numpy(ids))
            return host.to("cuda", non_blocking=True)
        host = staging.host("tokens", total, torch.int64)
        host.copy_(torch.from_numpy(ids))
        return staging.upload("tokens", total)

    def select_rows(self, rows, index):
        torch = self.torch
        if not torch.is_tensor(rows):
            return super().select_rows(rows, index)
        if not torch.is_tensor(index):
            index = torch.from_numpy(np.asarray(index, dtype=np.int64)).to(
                rows.device, non_blocking=True)
        return rows.index_select(0, index)

    # ---- timing and memory

    def record_event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def elapsed_ms(self, start, end) -> float:
        return start.elapsed_time(end)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def inference_mode(self):
        return self.torch.inference_mode()

    def peak_memory_bytes(self) -> int:
        return self.torch.cuda.max_memory_allocated()

    # ---- readouts

    def label_readout(self, model, targets, *, rows: int, normalize: bool):
        from quail.backends.quail.executor.model import full_output_head
        from quail.backends.quail.executor.readout import AsyncLabelLogprobs

        torch = self.torch
        head = full_output_head(model)
        readout = AsyncLabelLogprobs(
            torch, torch.nn.functional, head, targets, rows=rows,
            normalize=normalize)
        logger.info("label readout: head %s x %s in %s, %s targets, "
                    "%s rows per request, %s", *head.shape,
                    str(head.dtype).replace("torch.", ""), len(targets),
                    rows, "normalized" if normalize else "targets' logits")
        return readout

    def decision_choices(self, head, offsets):
        from quail.backends.quail.executor.readout import AsyncDecisionChoices

        return AsyncDecisionChoices(self.torch, head, offsets)

    def scores(self, answer_rows):
        from quail.backends.quail.executor.readout import AsyncScores

        return AsyncScores(self.torch, answer_rows)
