"""Tests for model specs and budget calculations on Qwen3-4B / H100."""

from dataclasses import replace

import pytest

from quail.common import budgets
from quail.specs import (
    B200,
    DIFFUSION_GEMMA_26B_FP8,
    H100_SXM,
    L40S,
    QWEN3_4B_FP8,
    QWEN3_5_4B_BF16,
    QWEN3_32B_FP8,
    RTX_PRO_6000_BLACKWELL_SERVER,
)


def test_model_weights_and_kv_memory_budgets():
    # Tied output weights must remain available for input embeddings.
    assert QWEN3_4B_FP8.head_mem_bytes == 0.0
    assert QWEN3_4B_FP8.W_resident == QWEN3_4B_FP8.W_mem
    # The separate 32B output head stays resident for AI.CLASSIFY: its
    # 1,555,824,640 bytes are 5,936 KV tokens (262,144 bytes each, in
    # whole 16-token pages) the arena does not get.
    head = QWEN3_32B_FP8.head_mem_bytes
    assert head == 151_936 * 5_120 * 2
    assert QWEN3_32B_FP8.W_resident == QWEN3_32B_FP8.W_mem
    assert budgets.arena_tokens(QWEN3_32B_FP8, H100_SXM) == 96_128
    # the chunk budget does not move: the 32B chunk is capped by the
    # int32 kernel index, not by memory
    kept = replace(QWEN3_32B_FP8, tied_head=True)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == \
        budgets.chunk_budget(kept, H100_SXM)

    tokens = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    assert tokens == 353_136
    assert tokens // 400 == 882
    fp8 = budgets.arena_tokens(replace(QWEN3_4B_FP8, kv_bytes=1.0), H100_SXM)
    assert fp8 == pytest.approx(2 * tokens, rel=0.01)
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, RTX_PRO_6000_BLACKWELL_SERVER) \
            > budgets.arena_tokens(model, H100_SXM)


def test_copy_verification_shares_readout_workspace_without_reducing_kv():
    for model, chunk in ((QWEN3_4B_FP8, 1), (DIFFUSION_GEMMA_26B_FP8, 1_000_000)):
        workspace = budgets.copy_verification_workspace_bytes(model, chunk)
        readout = 2 * budgets.READOUT_BLOCK_ROWS * model.vocab * 4
        assert isinstance(workspace, int)
        assert workspace == int(min(chunk * model.act_per_token, readout)), model.name
        activations = budgets.ACT_RESERVE_CHUNKS * chunk * model.act_per_token
        assert activations - workspace >= workspace, model.name
        assert budgets.arena_bytes(model, H100_SXM, chunk) == (
            H100_SXM.mem_bytes * budgets.POOL_FRACTION - model.W_resident
            - activations - budgets.graph_memory_bytes(model, H100_SXM)), model.name
        assert budgets.chunk_memory_bytes(model, chunk) == (
            model.W_resident + activations
            + budgets.checkpoint_memory_bytes(model, chunk)), model.name


def test_device_memory_budgets():
    for model in (QWEN3_4B_FP8, QWEN3_32B_FP8):
        assert budgets.arena_tokens(model, B200) > budgets.arena_tokens(model, H100_SXM)
        assert budgets.arena_tokens(model, L40S) < budgets.arena_tokens(model, H100_SXM)
    assert budgets.minimum_weight_gpus(QWEN3_32B_FP8, L40S) == 1
    # The int32 kernel index still caps 32B on H100 and B200. On L40S
    # the memory bound would spend the leftover on two activation
    # chunks, so the chunk shrinks until the arena holds that chunk.
    cap = budgets.kernel_index_cap(QWEN3_32B_FP8)
    assert budgets.chunk_budget(QWEN3_32B_FP8, H100_SXM) == cap
    assert budgets.chunk_budget(QWEN3_32B_FP8, B200) == cap
    l40s_chunk = budgets.chunk_budget(QWEN3_32B_FP8, L40S)
    assert l40s_chunk < cap
    full, sliding = budgets.arena_pages(QWEN3_32B_FP8, L40S, l40s_chunk)
    assert sliding == 0
    assert full * budgets.PAGE_TOKENS >= l40s_chunk


@pytest.mark.parametrize("tokens,expected_slots", [(200, 587)])
def test_hybrid_budget_reserves_checkpoints_and_two_execution_chunks(
        tokens, expected_slots):
    from quail.backends.quail.worker import checkpoint_arena_args
    from quail.common.retention import retention_pages

    model = replace(QWEN3_5_4B_BF16, w_mem_bytes=8_411_152_384)
    chunk = budgets.chunk_budget(model, H100_SXM)
    assert chunk == 116_508
    assert model.checkpoint_bytes == 51_511_296
    assert budgets.checkpoint_pass_bytes(model, 1_024, chunk) == 4_941 * 2**20
    pool = budgets.hybrid_pool_pages(model, H100_SXM, chunk)
    # the 1.34 GB CUDA graph reserve takes 2,560 of 68,060 pages
    assert budgets.graph_memory_bytes(model, H100_SXM) == 2_560 * 16 * model.kappa
    assert pool == 65_500
    kv_pages, sliding = budgets.arena_pages(model, H100_SXM, chunk, tokens)
    assert sliding == 0
    slots = budgets.checkpoint_slots(model, H100_SXM, chunk, kv_pages)
    assert slots == expected_slots
    cost = -(-tokens // 16) + budgets.checkpoint_pages(model)
    # the recurrent state's unused row 0 takes one slot's room
    assert (slots + 1) * cost <= pool < (slots + 2) * cost
    retained = retention_pages(kv_pages * 16, chunk, 16)
    assert retained == kv_pages - budgets.execution_pages(chunk)
    assert retained + slots * budgets.checkpoint_pages(model) <= pool
    assert checkpoint_arena_args(model, H100_SXM, chunk, kv_pages) == {
        "num_state_slots": slots,
        "gdn_state_shape": (24, 32, 128, 128),
        "conv_state_shape": (24, 3, 8_192),
    }
    with pytest.raises(ValueError, match="holds no checkpoint"):
        budgets.hybrid_arena_pages(model, H100_SXM, chunk, 1e7)


def test_a_plan_without_an_answer_writer_gives_the_graph_memory_to_kv():
    for model in (QWEN3_4B_FP8, QWEN3_5_4B_BF16):
        chunk = budgets.chunk_budget(model, H100_SXM)
        graphs = budgets.graph_memory_bytes(model, H100_SXM)
        assert graphs > 0, model.name
        assert (budgets.arena_bytes(model, H100_SXM, chunk, writes_answers=False)
                - budgets.arena_bytes(model, H100_SXM, chunk)) == graphs, model.name
    # Qwen3-4B: the 1,280 MiB reserve is 569 more pages of 16 tokens
    page_bytes = QWEN3_4B_FP8.kappa * budgets.PAGE_TOKENS
    kept = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM)
    given = budgets.arena_tokens(QWEN3_4B_FP8, H100_SXM, writes_answers=False)
    assert kept == 353_136
    assert given - kept == 569 * budgets.PAGE_TOKENS
    reserve = budgets.graph_memory_bytes(QWEN3_4B_FP8, H100_SXM)
    assert reserve == 1_280 * 2**20
    assert 568 < reserve / page_bytes < 569

    # a hybrid model's pool grows by the reserve's 2,560 pages, and the
    # worker counts its checkpoint slots from the same pool
    model = replace(QWEN3_5_4B_BF16, w_mem_bytes=8_411_152_384)
    chunk = budgets.chunk_budget(model, H100_SXM)
    assert budgets.hybrid_pool_pages(model, H100_SXM, chunk) == 65_500
    assert budgets.hybrid_pool_pages(
        model, H100_SXM, chunk, writes_answers=False) == 68_060
    kv_pages, _ = budgets.arena_pages(model, H100_SXM, chunk, 200,
                                      writes_answers=False)
    slots = budgets.checkpoint_slots(model, H100_SXM, chunk, kv_pages,
                                     writes_answers=False)
    assert slots > budgets.checkpoint_slots(model, H100_SXM, chunk, kv_pages)
    retained = kv_pages - budgets.execution_pages(chunk)
    assert retained + (slots + 1) * budgets.checkpoint_pages(model) <= 68_060

    # a model without CUDA graphs has no reserve to give
    assert budgets.graph_memory_bytes(DIFFUSION_GEMMA_26B_FP8, H100_SXM) == 0
    chunk = budgets.chunk_budget(DIFFUSION_GEMMA_26B_FP8, H100_SXM)
    assert (budgets.arena_bytes(DIFFUSION_GEMMA_26B_FP8, H100_SXM, chunk,
                                writes_answers=False)
            == budgets.arena_bytes(DIFFUSION_GEMMA_26B_FP8, H100_SXM, chunk))


def test_only_a_plan_with_a_map_or_an_extraction_keeps_the_graph_memory():
    import pyarrow as pa
    from helpers.sessions import CONFIG, TOKENIZER

    import quail

    reviews = pa.table({"id": ["r0", "r1"], "body": ["kind words", "rude words"]})
    queries = {
        "score": "SELECT r.id, AI.SCORE(PROMPT('kind: {0}', r.body)) AS s "
                 "FROM reviews r",
        "filter": "SELECT r.id FROM reviews r "
                  "WHERE AI_FILTER(PROMPT('kind: {0}', r.body))",
        "map": "SELECT r.id, AI.MAP(r.body, 'Is it kind?') AS v FROM reviews r",
        "extract": "SELECT r.id, AI.EXTRACT(r.body, {'w': 'Which word?'}) AS e "
                   "FROM reviews r",
    }
    with quail.Session(CONFIG, tokenizer=TOKENIZER) as session:
        session.register(
            "reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
        settings = {name: session.sql(sql).plan().settings
                    for name, sql in queries.items()}
    kept = budgets.arena_pages(QWEN3_4B_FP8, H100_SXM)
    given = budgets.arena_pages(QWEN3_4B_FP8, H100_SXM, writes_answers=False)
    assert given[0] - kept[0] == 569
    for name in ("map", "extract"):
        assert settings[name]["writes_answers"] is True, name
        assert tuple(settings[name]["arena_pages"]) == kept, name
        assert settings[name]["admission_tokens"] == 353_136, name
    for name in ("score", "filter"):
        assert settings[name]["writes_answers"] is False, name
        assert tuple(settings[name]["arena_pages"]) == given, name
        assert settings[name]["admission_tokens"] == 353_136 + 569 * 16, name
