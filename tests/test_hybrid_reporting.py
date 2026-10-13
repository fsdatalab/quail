"""What a hybrid run reports: plan lines, planning refusals, and memory."""

from types import SimpleNamespace

import pytest

from quail.backends.quail.backend import QuailBackend
from quail.backends.quail.worker import memory_breakdown, pool_bytes
from quail.explain import architecture_lines
from quail.specs import DEVICES, MODELS


class _Tensor:
    def __init__(self, elements, element_size):
        self._elements, self._size = elements, element_size

    def numel(self):
        return self._elements

    def element_size(self):
        return self._size


def test_hybrid_model_lines_name_layers_precision_and_kernels(monkeypatch):
    monkeypatch.setenv("QUAIL_GDN_PREFILL", "flashinfer")
    architecture, kernels = architecture_lines("qwen3.5-4b-bf16")
    assert architecture == ("architecture=qwen3_5 (24 gated_delta_net, "
                            "8 full_attention), weights=bf16")
    assert "bf16 GEMM (cuBLAS)" in kernels
    assert "recurrence=flashinfer (QUAIL_GDN_PREFILL)" in kernels
    assert "FlashAttention over paged KV" in kernels


def test_dense_fp8_model_lines_have_no_recurrence_kernel():
    architecture, kernels = architecture_lines("qwen3-4b-fp8")
    assert architecture.startswith("architecture=qwen3 (")
    assert architecture.endswith("weights=fp8")
    assert "fp8 GEMM (DeepGEMM)" in kernels and "recurrence" not in kernels


def test_unknown_model_has_no_architecture_lines():
    assert architecture_lines("no-such-model") == []


@pytest.mark.parametrize("device, accepted", [
    ("h100-sxm", True), ("rtx-pro-6000-blackwell-server", False)])
def test_flashinfer_prefill_is_refused_off_hopper(monkeypatch, device,
                                                  accepted):
    monkeypatch.setenv("QUAIL_GDN_PREFILL", "flashinfer")
    result = QuailBackend().supports(
        MODELS["qwen3.5-4b-bf16"], DEVICES[device], 1)
    assert result.supported is accepted
    if not accepted:
        assert "flashinfer needs a Hopper GPU" in result.reason


def test_triton_prefill_is_accepted_off_hopper(monkeypatch):
    monkeypatch.delenv("QUAIL_GDN_PREFILL", raising=False)
    device = DEVICES["rtx-pro-6000-blackwell-server"]
    assert QuailBackend().supports(
        MODELS["qwen3.5-4b-bf16"], device, 1).supported


def test_pool_bytes_and_breakdown_add_up():
    arena = SimpleNamespace(
        k=[_Tensor(10, 2)] * 2, v=[_Tensor(10, 2)] * 2,
        gdn_state=_Tensor(5, 4), conv_state=_Tensor(7, 2))
    pools = pool_bytes(arena)
    assert pools == {"kv": 80, "state": 34}
    parts = memory_breakdown(1_000, pools, 1_500)
    assert parts == {"weights": 1_000, "kv": 80, "state": 34,
                     "other": 386, "peak": 1_500}


def test_pool_bytes_without_checkpoint_state():
    arena = SimpleNamespace(k=[_Tensor(4, 2)], v=[_Tensor(4, 2)],
                            gdn_state=None, conv_state=None)
    assert pool_bytes(arena) == {"kv": 16, "state": 0}
