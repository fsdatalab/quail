from dataclasses import replace

from .qwen3_5_27b import QWEN3_5_27B_FP8

QWEN3_8_27B_FP8 = replace(
    QWEN3_5_27B_FP8,
    name="qwen3.8-27b-fp8",
    hf_name="Qwen/Qwen3.8-27B-FP8",
    revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
)
