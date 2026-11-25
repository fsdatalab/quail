"""qwen3_spec against the hand-written Qwen3 specs."""

import json
from pathlib import Path

import pytest

from quail.specs import MODELS, qwen3_spec

CONFIGS = Path(__file__).parents[1] / "fixtures" / "qwen3_configs"
ARCHITECTURE = ("layers", "hidden", "num_q_heads", "num_kv_heads", "d_head",
                "ffn_width",
                "w_bytes", "vocab", "tied_head", "weight_precision", "arch",
                "kappa", "widest_projection")


@pytest.mark.parametrize("name", ["qwen3-4b-fp8"])
def test_qwen3_spec_matches_built_in(name):
    built_in = MODELS[name]
    config = json.loads((CONFIGS / f"{name}.json").read_text())
    if built_in.role == "decision":
        # the float32 backbone, which vLLM loads as bfloat16
        config.update(torch_dtype="bfloat16", dtype="bfloat16")
    spec = qwen3_spec(name, built_in.hf_name, built_in.revision,
                      config=config, role=built_in.role)
    for field in ARCHITECTURE:
        assert getattr(spec, field) == getattr(built_in, field), field
    assert spec.role == built_in.role
    assert spec.turn == built_in.turn
    if built_in.role != "reranker":
        # the reranker specs count the embedding table in params
        assert spec.params == pytest.approx(built_in.params, rel=0.01)
        # the measured footprint also holds fp8 block scales and buffers
        assert spec.w_mem_bytes == pytest.approx(built_in.w_mem_bytes, rel=0.03)






