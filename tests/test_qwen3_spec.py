"""qwen3_spec against the hand-written Qwen3 specs."""

import json
from pathlib import Path

import pytest

from quail.specs import MODELS, qwen3_spec

CONFIGS = Path(__file__).parent / "fixtures" / "qwen3_configs"
ARCHITECTURE = ("layers", "hidden", "n_q", "n_kv", "d_head", "ffn_width",
                "w_bytes", "vocab", "tied_head", "weight_precision", "arch",
                "kappa", "widest_projection")


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIGS.glob("*.json")))
def test_qwen3_spec_matches_built_in(name):
    built_in = MODELS[name]
    config = json.loads((CONFIGS / f"{name}.json").read_text())
    if built_in.role == "decision":
        # the float32 backbone as convert_decision2 rewrites it
        config.update(torch_dtype="bfloat16", dtype="bfloat16")
    spec = qwen3_spec(name, built_in.hf_name, built_in.revision,
                      config=config, role=built_in.role)
    for field in ARCHITECTURE:
        assert getattr(spec, field) == getattr(built_in, field), field
    assert spec.role == built_in.role
    if built_in.role != "reranker":
        # the reranker specs count the embedding table in params
        assert spec.params == pytest.approx(built_in.params, rel=0.01)


def _config(**changes):
    config = json.loads((CONFIGS / "qwen3-4b-fp8.json").read_text())
    config.update(changes)
    return config


def test_qwen3_spec_fields_override_derived_values():
    spec = qwen3_spec("m", "Org/M", config=_config(), w_mem_bytes=4.5e9)
    assert spec.w_mem_bytes == 4.5e9
    assert spec.W_mem == 4.5e9


@pytest.mark.parametrize("changes, message", [
    ({"model_type": "llama"}, "not 'qwen3'"),
    ({"use_sliding_window": True}, "sliding-window"),
    ({"torch_dtype": "float32"}, "float32"),
    ({"quantization_config": {"quant_method": "fp8"}}, "block size None"),
    ({"quantization_config": {"quant_method": "awq"}}, "'awq'"),
])
def test_qwen3_spec_rejects_layouts_the_executor_cannot_run(changes, message):
    with pytest.raises(ValueError, match=message):
        qwen3_spec("m", "Org/M", config=_config(**changes))
