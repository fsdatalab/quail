"""RTX PRO 6000 attention dispatch without a GPU."""

import sys
from types import ModuleType

from quail.backends.quail.executor.attention import Engine, flash_attention_version


def test_attention_dispatch_preserves_paged_arguments(monkeypatch):
    for capability, version in [((9, 0), 3), ((12, 0), 2)]:
        recorded = {}
        expected = object()

        def fake_attention(*args, **kwargs):
            recorded.update(kwargs)
            return expected

        module = ModuleType("vllm.vllm_flash_attn")
        module.flash_attn_varlen_func = fake_attention
        monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
        pipeline = Engine.__new__(Engine)
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
