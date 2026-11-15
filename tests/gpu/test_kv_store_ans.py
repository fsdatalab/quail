"""The host KV store on a GPU: nvCOMP's ANS coder restores KV bit for bit."""

import pytest

from quail.backends.quail.executor.kv_store import PIECE_TOKENS, KVStore

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA GPU")

TENSORS, HEADS, DIM = 72, 8, 128


def _pool(rows):
    generator = torch.Generator().manual_seed(0)
    values = torch.randn((TENSORS, rows, HEADS, DIM), generator=generator)
    return values.to(torch.bfloat16).cuda()


def test_the_ans_coder_restores_prefixes_bit_for_bit_in_fewer_bytes():
    pool = _pool(8192)
    store = KVStore(torch, (TENSORS, HEADS, DIM), torch.bfloat16, 4 << 30)
    try:
        # scattered pages of three prefixes: shorter than a piece, longer
        # than one, and longer than a group of pieces
        order = torch.randperm(4096, generator=torch.Generator().manual_seed(1))
        lengths = {b"short": 300, b"long": PIECE_TOKENS + 77,
                   b"longer": 2 * PIECE_TOKENS + 500}
        rows, start = {}, 0
        for name, tokens in lengths.items():
            rows[name] = order[start:start + tokens]
            start += tokens
            store.save(name, rows[name])
        store.flush(pool)
        stats = store.stats()
        assert stats["coder"] == "ans"
        assert stats["saved_tokens"] == sum(lengths.values())
        # random bf16 values still have a skewed exponent byte
        assert stats["held_bytes"] < 0.9 * stats["held_tokens"] * store.record_bytes
        target = 4096
        for name, tokens in lengths.items():
            store.restore(name, torch.arange(target, target + tokens))
            target += tokens
        store.flush(pool)
        torch.cuda.synchronize()
        target = 4096
        for name, tokens in lengths.items():
            assert torch.equal(pool[:, rows[name].cuda()],
                               pool[:, target:target + tokens]), name
            target += tokens
    finally:
        store.close()
