"""Attention work and prices against explicit causal masks."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quail
from quail.backends.quail.retention import policy
from quail.cost.dense_decoder_cost import attention_pair_flops, dense_decoder_components
from quail.cost.retention import coefficients
from quail.cost.sol import prefix_recompute_seconds
from quail.cost.work import Work, ask, scan, stream
from quail.planner.joins import stage_work, summarize_alias
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8


def mask_pairs(prefix, suffix, window):
    """Count allowed keys for every new query position."""
    return sum(1 for q in range(prefix, prefix + suffix) for k in range(q + 1)
               if not window or q - k < window)


@pytest.mark.parametrize("window", [0, 1, 4])
@pytest.mark.parametrize("prefix,suffix", [(0, 0), (0, 3), (2, 4), (4, 1), (9, 3)])
def test_work_matches_attention_masks(window, prefix, suffix):
    first = scan(prefix, suffix, window=window)
    continuation = ask(prefix, suffix, window=window)
    assert first.pairs == mask_pairs(0, prefix + suffix, 0)
    assert continuation.pairs == mask_pairs(prefix, suffix, 0)
    assert first.sliding_pairs == (
        mask_pairs(0, prefix + suffix, window) if window else 0)
    assert continuation.sliding_pairs == (
        mask_pairs(prefix, suffix, window) if window else 0)
    assert continuation.sliding_kv_read == (
        min(prefix, window - 1) if window else 0)
    branches = stream(prefix, [suffix, suffix + 1], window=window)
    assert branches.sliding_pairs == (
        sum(mask_pairs(prefix, s, window) for s in (suffix, suffix + 1))
        if window else 0)
    assert branches.sliding_kv_read == continuation.sliding_kv_read


def test_work_arithmetic_and_dominance_include_sliding_counts():
    small = Work(10, 30, 10, 5, 20, 3)
    assert small + small == small * 2
    assert small.dominates(small * 2)
    assert not small.dominates(replace(small, sliding_pairs=19))
    assert not small.dominates(replace(small, sliding_kv_read=2))


@pytest.mark.parametrize("model", [QWEN3_4B_FP8, DIFFUSION_GEMMA_26B_FP8])
def test_attention_prices_each_layer_and_retention_uses_the_same_cost(model):
    work = ask(4096, 16, window=model.sliding_window)
    attention = dense_decoder_components(work, model, 1)[2]
    flops = moved = 0
    for layer, (kv_heads, head_size) in enumerate(model.kv_shapes):
        sliding = layer in model.sliding_layer_set
        pairs = work.sliding_pairs if sliding else work.pairs
        read = work.sliding_kv_read if sliding else work.kv_read
        flops += 4 * model.n_q * head_size * pairs
        moved += 2 * kv_heads * head_size * model.kv_bytes * (work.kv_written + read)
    assert attention.flops == flops
    assert attention.bytes_moved == moved
    retained = policy(coefficients(model, H100_SXM), {"a": (0.4, 2)})
    for length in (1, 1023, 1024, 1025, 24000):
        assert retained.priority(("a", 0), length, 1)[0] == pytest.approx(
            0.4 * prefix_recompute_seconds(length, model, H100_SXM))
    if not model.sliding_window:
        assert attention.flops == (
            4 * model.n_q * model.d_head * model.layers * work.pairs)


def test_hybrid_head_sizes_and_layers_without_a_window():
    model = DIFFUSION_GEMMA_26B_FP8
    assert attention_pair_flops(model) == (4 * 16 * 5 * 512, 4 * 16 * 25 * 256)
    assert attention_pair_flops(replace(model, sliding_window=0)) == (
        4 * 16 * (5 * 512 + 25 * 256), 0)


@pytest.mark.parametrize("resident", [False, True])
def test_windowed_join_summary_matches_per_document_work(resident):
    window = 16
    lengths = {"a": [0, 3, 11, 12, 15, 16, 17, 1000], "b": [1, 5]}
    stats = {alias: summarize_alias(values, window)
             for alias, values in lengths.items()}
    spec = dict(aliases=["a", "b"], tail_tokens=1, label_tokens={"b": 1},
                frame_tokens={"a": 3}, pair_fraction=0.5)
    live = {"a": 4.0, "b": 1.5}
    actual = stage_work(spec, "a", live, stats, 2, resident=resident, window=window)
    expected = Work()
    for length in lengths["a"]:
        prefix = 2 + length
        start = (ask(prefix, 3, window=window) if resident
                 else scan(prefix, 3, window=window))
        suffix = ask(prefix + 3, 5, window=window)
        suffix = replace(suffix, tokens=suffix.tokens * 0.75,
                         pairs=suffix.pairs * 0.75,
                         sliding_pairs=suffix.sliding_pairs * 0.75,
                         kv_written=suffix.kv_written * 0.75)
        expected += (start + suffix) * 0.5
    assert actual == expected
    assert len(stats["a"].short_counts) <= window


def test_planner_and_estimate_price_hybrid_queries_and_prefix_reuse(tmp_path):
    pq.write_table(pa.table({
        "id": ["r0", "r1"], "body": ["shared " * 1030 + s for s in ("a", "b")],
    }), tmp_path / "reviews.parquet")
    pq.write_table(pa.table({
        "id": ["q0", "q1"], "body": ["one two", "three four"],
    }), tmp_path / "questions.parquet")
    with quail.Session(config=quail.EngineConfig(
            model="diffusion-gemma-26b-a4b-fp8", device="h100-sxm"),
            tokenizer=str.split) as session:
        for name in ("reviews", "questions"):
            session.register(name, quail.DocumentProvider.from_parquet(
                str(tmp_path / f"{name}.parquet"), id_col="id"))
        query = (session.docs("reviews").alias("r")
                 .ai_filter(quail.prompt("First: {0}", quail.col("r.body")),
                            selectivity=1.0)
                 .ai_filter(quail.prompt("Second: {0}", quail.col("r.body")),
                            selectivity=1.0)
                 .ai_join(session.docs("questions").alias("q"),
                          quail.prompt("Compare {0} and {1}", quail.col("r.body"),
                                       quail.col("q.body")), selectivity=1.0)
                 .select("r.id", "q.id"))
        exact = quail.speed_of_light_estimate(
            query, lambda *_: True, credit_shared_prefixes=False)
        shared = quail.speed_of_light_estimate(query, lambda *_: True)
        planned = query.plan()
        pre = query.logical.operators().filters["r"][0].prompt.preamble_tokens
    assert exact.join_stages[0]["anchor"] == "r"
    assert exact.seconds == pytest.approx(planned.estimated_seconds)
    assert exact.work.sliding_pairs < exact.work.pairs
    assert exact.work.sliding_kv_read < exact.work.kv_read
    assert exact.work.pairs - shared.work.pairs == mask_pairs(0, pre + 1030, 0)
    assert exact.work.sliding_pairs - shared.work.sliding_pairs == mask_pairs(
        0, pre + 1030, 1024)
    assert shared.as_dict()["sliding_pairs"] == shared.work.sliding_pairs
