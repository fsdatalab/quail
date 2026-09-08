"""RTX PRO 6000 planning and attention dispatch without a GPU."""

import sys
from types import ModuleType

import pyarrow as pa
import pytest

import quail
from quail.executor.attention import Pipeline, flash_attention_version
from quail.planner import budgets
from quail.specs import H100_SXM, MODELS, RTX_PRO_6000_BLACKWELL_SERVER


@pytest.mark.parametrize("model_name,gpus", [("qwen3-4b-fp8", 1),
                                             ("qwen3-32b-fp8", 8)])
def test_rtx_plan_uses_its_memory_budget(model_name, gpus):
    device = RTX_PRO_6000_BLACKWELL_SERVER
    config = quail.EngineConfig(model=model_name, device=device.name, gpus=gpus)
    with quail.Session(config, tokenizer=lambda text: list(text.encode())) as session:
        session.register("docs", quail.DocumentProvider.from_table(
            pa.table({"id": list(range(16)), "body": ["a document"] * 16}),
            id_col="id",
        ))
        plan = session.sql(
            "SELECT d.id FROM docs d WHERE "
            "AI_FILTER(PROMPT('Is this relevant? {0}', d.body))"
        ).plan()

    model = MODELS[model_name]
    assert plan.device == device.name
    assert plan.workers == gpus
    assert plan.settings["admission_tokens"] == budgets.arena_tokens(model, device)
    assert plan.settings["admission_tokens"] > budgets.arena_tokens(model, H100_SXM)
    assert plan.estimated_seconds > 0


@pytest.mark.parametrize("capability,version", [((9, 0), 3), ((12, 0), 2)])
def test_attention_dispatch_preserves_paged_arguments(monkeypatch, capability, version):
    recorded = {}
    expected = object()

    def fake_attention(*args, **kwargs):
        recorded.update(kwargs)
        return expected

    module = ModuleType("vllm.vllm_flash_attn")
    module.flash_attn_varlen_func = fake_attention
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.fa_version = flash_attention_version(capability)
    table, lengths = object(), object()
    result = pipeline._fa(
        None, None, None, None, None, 16, 32, True,
        block_table=table, seqused_k=lengths,
    )
    assert result is expected
    assert recorded["fa_version"] == version
    assert recorded["block_table"] is table
    assert recorded["seqused_k"] is lengths
    assert recorded["return_softmax_lse"] is True
    assert recorded["causal"] is True
