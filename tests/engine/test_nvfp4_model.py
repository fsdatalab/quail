"""Qwen3.8 27B in NVFP4: its spec and the GPUs that can run it."""

from dataclasses import fields

from quail.backends.quail.backend import QuailBackend
from quail.specs import B200, H100_SXM, QWEN3_8_27B_FP8, QWEN3_8_27B_NVFP4


def test_qwen3_8_nvfp4_is_the_fp8_spec_with_fp4_mlp_weights():
    differ = {field.name for field in fields(QWEN3_8_27B_FP8)
              if getattr(QWEN3_8_27B_FP8, field.name)
              != getattr(QWEN3_8_27B_NVFP4, field.name)}

    assert differ == {"name", "hf_name", "revision", "mlp_weight_precision",
                      "w_mem_bytes"}
    assert QWEN3_8_27B_NVFP4.mlp_weight_precision == "fp4"
    assert QWEN3_8_27B_NVFP4.w_mem_bytes < QWEN3_8_27B_FP8.w_mem_bytes


def test_qwen3_8_nvfp4_is_refused_on_an_h100_and_supported_on_a_b200():
    backend = QuailBackend()

    refused = backend.supports(QWEN3_8_27B_NVFP4, H100_SXM, 1)

    assert not refused.supported
    assert "no FP4 tensor cores" in refused.reason
    assert backend.supports(QWEN3_8_27B_NVFP4, B200, 1).supported
    assert backend.supports(QWEN3_8_27B_FP8, H100_SXM, 1).supported
