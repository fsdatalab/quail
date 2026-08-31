import itertools
from types import SimpleNamespace

import pytest

from baselines.stock import build_join_grouped_inputs, run_filter_chain
from baselines.stock_vllm.run import (
    GPU_MEMORY_UTILIZATION,
    _baseline_configuration,
    _baseline_schedule,
    _filter_chain_inputs,
    _filter_prompts,
    _join_records_by_written_position,
    _join_regret,
    _lcp,
    _paired_baseline_order,
    _paired_ground_truth_workload,
    _query_set_name,
    _run_join,
    _run_pipelined_filter_chain,
    _run_stage_major_filter_chain,
    _select_join_anchor,
    _split_query_sets,
    _vllm_failure_entry,
    _vllm_filter_capacity,
    define_all_queries,
)
from quail.bench.quailb import AGENT_IMPLEMENTED_FIX, QUERY_ORDER
from quail.logical import (ColumnRef, bind_join_prompt, bind_prompt,
                           render_filter_prompt_ids,
                           render_join_prompt_ids)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def _tok(tokenizer):
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def test_stock_filter_prompts_match_quail_token_for_token():
    tokenizer = CharTokenizer()
    tok = _tok(tokenizer)
    template = ("Judge the review.\n\n{0}\n\nInstruction: answer TRUE "
                "when it passes, FALSE otherwise.")
    texts = ["first review", "second review"]
    ref = ColumnRef("r", "reviews", "body")
    bound = bind_prompt(template, (ref,), tok)

    stock = _filter_prompts(template, texts, tokenizer)
    expected = [
        {"prompt_token_ids": render_filter_prompt_ids(
            bound, tok(document), tok)}
        for document in texts
    ]

    assert stock == expected


def test_stock_catalog_matches_current_queries():
    definitions = define_all_queries()

    assert set(definitions) == set(QUERY_ORDER)
    assert sorted(
        query_id for query_id in definitions
        if query_id.startswith("BIO-")) == [
            "BIO-1", "BIO-2", "BIO-3"]
    assert sorted(
        query_id for query_id in definitions
        if query_id.startswith("AGENT-")) == ["AGENT-1", "AGENT-2"]
    assert definitions["AGENT-2"]["steps"] == [
        ("filter", "t", [AGENT_IMPLEMENTED_FIX])]


def test_pipelined_filter_parts_match_complete_filter_prompts():
    tokenizer = CharTokenizer()
    templates = [
        "First question about {0}?",
        "A longer second question about {0}?",
    ]
    texts = ["short", "a longer document"]

    bodies, tails = _filter_chain_inputs(templates, texts, tokenizer)

    for document, body in zip(texts, bodies):
        for template, tail in zip(templates, tails):
            expected = _filter_prompts(
                template, [document], tokenizer)[0]["prompt_token_ids"]
            assert body + tail == expected


def test_baseline_names_select_only_the_filter_submission():
    assert GPU_MEMORY_UTILIZATION == 0.91
    assert _baseline_configuration("stock_vllm") == "stage-major"
    assert _baseline_configuration("pipelined_vllm") == "pipelined"
    with pytest.raises(ValueError, match="unknown baseline"):
        _baseline_configuration("other")


def test_vllm_failure_entry_marks_cuda_oom():
    entry = _vllm_failure_entry(
        "BIO-2", RuntimeError("CUDA out of memory"))

    assert entry["query"] == "BIO-2"
    assert entry["status"] == "oom"
    assert entry["error"] == "oom"
    assert entry["total_wall_s"] is None
    assert entry["regret_tokens"] is None


def test_vllm_failure_entry_marks_dead_engine_as_oom():
    class EngineDeadError(RuntimeError):
        pass

    entry = _vllm_failure_entry(
        "BIO-3", EngineDeadError("engine stopped"))

    assert entry["status"] == "oom"


def test_vllm_failure_entry_preserves_other_errors():
    entry = _vllm_failure_entry("BIO-1", ValueError("bad input"))

    assert entry["status"] == "error"
    assert entry["error"] == "ValueError: bad input"


def test_join_anchor_uses_larger_mean_surviving_document_length():
    anchor, means = _select_join_anchor((
        [[1] * 10, [1] * 20],
        [[1] * 40, [1] * 50],
    ))
    assert anchor == 1
    assert means == [15, 45]

    tie, _ = _select_join_anchor(([[1] * 10], [[1] * 10]))
    assert tie == 0


def test_join_execution_uses_the_selected_anchor(monkeypatch):
    tokenizer = CharTokenizer()

    def fake_join(_llm, _sp, prefixes, suffixes, _true_set,
                  submission="anchor-major", tile_budget_tokens=None):
        assert len(prefixes) == 2
        assert len(suffixes) == 2
        assert submission == "anchor-major"
        assert tile_budget_tokens is None
        return dict(
            answers=[1, 0, 0, 1],
            fresh_tokens=100,
            prompt_tokens=120,
            cached_tokens=20,
        )

    monkeypatch.setattr("baselines.stock.run_join_grouped", fake_join)
    result = _run_join(
        object(), object(), {1},
        "Does DOCUMENT {0} match DOCUMENT {1}?",
        ["a", "bb"],
        ["right side one", "right side two"],
        tokenizer,
    )

    assert result[8] == 1
    assert result[5] == [(0, 0), (1, 1)]
    assert len(result[10]) == 2     # one prefix per anchor document


def test_lcp_counts_shared_leading_tokens():
    assert _lcp([1, 2, 3], [1, 2, 4]) == 2
    assert _lcp([1, 2], [1, 2, 3]) == 2
    assert _lcp([], [1]) == 0


def test_join_regret_buckets_pair0_and_rest():
    # Two anchors, two suffixes each, 16-token blocks. Anchor 0 was
    # computed before (40 seen tokens): pair 0 misses everything
    # (regret 32, the block floor of 40) and pair 1 hits fully.
    # Anchor 1 is new: pair 0 owes nothing even though vLLM served 16
    # tokens, and pair 1 recomputes half its 40-token prefix
    # (regret 32 - 16 = 16).
    prefixes = [[7] * 40, [9] * 40]
    cached = [0, 32, 16, 16]

    assert _join_regret(prefixes, 2, cached, [40, 0], 16) == 48


def test_join_regret_is_zero_when_cache_serves_every_would_hit():
    prefixes = [[7] * 32]
    assert _join_regret(prefixes, 2, [32, 32], [32], 16) == 0


class _FakeFilterLLM:
    """Returns scripted TRUE/FALSE verdicts, one list per generate."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    def generate(self, prompts, _sp, use_tqdm=False):
        verdicts = self.verdicts.pop(0)
        assert len(verdicts) == len(prompts)
        return [SimpleNamespace(
            prompt_token_ids=prompt["prompt_token_ids"],
            num_cached_tokens=0,
            outputs=[SimpleNamespace(
                token_ids=[1 if verdict else 0], text="")])
            for prompt, verdict in zip(prompts, verdicts)]


def test_stage_major_chain_records_passed_prompts_for_regret():
    tokenizer = CharTokenizer()
    templates = ["First about {0}?", "Second about {0}?"]
    texts = ["doc zero", "doc one", "doc two"]
    llm = _FakeFilterLLM([[True, False, True], [True, False]])

    result = _run_stage_major_filter_chain(
        llm, object(), {1}, templates, texts, tokenizer)

    def prompt(template, text):
        return _filter_prompts(
            template, [text], tokenizer)[0]["prompt_token_ids"]

    assert result["survivors"] == [0]
    # documents 0 and 2 passed stage 1; only document 0 passed stage
    # 2; document 1 passed nothing and computed no reusable KV
    assert result["prior_prompts"] == {
        0: [prompt(templates[0], texts[0]),
            prompt(templates[1], texts[0])],
        2: [prompt(templates[0], texts[2])],
    }


def test_filter_capacity_comes_from_started_vllm_config():
    cache = SimpleNamespace(
        kv_cache_size_tokens=123_456,
        num_gpu_blocks=7_716,
        block_size=16,
        cache_dtype="auto",
    )
    config = SimpleNamespace(
        cache_config=cache,
        scheduler_config=SimpleNamespace(max_num_seqs=4096),
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=config))

    assert _vllm_filter_capacity(llm) == {
        "kv_cache_size_tokens": 123_456,
        "num_gpu_blocks": 7_716,
        "block_size": 16,
        "max_num_seqs": 4096,
        "kv_cache_dtype": "auto",
    }


class _FakeFilterEngine:
    def __init__(self):
        self.pending = []
        self.events = []

    def add_request(self, request_id, prompt, _sampling_params):
        self.pending.append((request_id, prompt["prompt_token_ids"]))
        self.events.append(("add", request_id))

    def step(self):
        request_id, prompt_token_ids = self.pending.pop(0)
        self.events.append(("finish", request_id))
        return [SimpleNamespace(
            finished=True,
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            num_cached_tokens=0,
            outputs=[SimpleNamespace(token_ids=[1], text="TRUE")],
        )]


def test_filter_chain_submits_next_stage_before_prior_stage_finishes():
    engine = _FakeFilterEngine()
    result = run_filter_chain(
        engine,
        sampling_params=object(),
        body_ids=[[1] * 5, [2] * 17],
        q_ids=[[3] * 3, [4] * 3],
        budget_tokens=100,
        tag="test",
        true_ids={1},
        block_size=16,
        max_num_seqs=2,
    )

    assert result["doc_cap"] == 2
    assert engine.events.index(("add", "test-0-1")) < \
        engine.events.index(("finish", "test-1-0"))
    assert result["survivors"] == [0, 1]


def test_pipelined_chain_records_passed_prompts_for_regret():
    engine = _FakeFilterEngine()    # answers TRUE for every request
    llm = SimpleNamespace(llm_engine=engine)
    tokenizer = CharTokenizer()
    templates = ["First about {0}?", "Second about {0}?"]
    texts = ["short", "a longer document"]
    capacity = dict(kv_cache_size_tokens=10_000, block_size=16,
                    max_num_seqs=8)

    result = _run_pipelined_filter_chain(
        llm, object(), {1}, templates, texts, tokenizer,
        capacity, tag="t")

    bodies, tails = _filter_chain_inputs(templates, texts, tokenizer)
    assert result["prior_prompts"] == {
        0: [bodies[0] + tails[0], bodies[0] + tails[1]],
        1: [bodies[1] + tails[0], bodies[1] + tails[1]],
    }
    assert result["prior_prompts"][0][0] == _filter_prompts(
        templates[0], [texts[0]], tokenizer)[0]["prompt_token_ids"]


def test_stock_query_set_split_uses_one_chunk_per_set():
    ids = ["IMDB-1", "IMDB-5", "BIO-1", "FEV-9", "LEP-3"]
    assert _split_query_sets(ids) == [
        ["IMDB-1", "IMDB-5"],
        ["BIO-1"],
        ["FEV-9"],
        ["LEP-3"],
    ]


def test_paired_order_alternates_by_query_and_rep():
    assert _paired_baseline_order(0, 0) == (
        "stock_vllm", "pipelined_vllm")
    assert _paired_baseline_order(0, 1) == (
        "pipelined_vllm", "stock_vllm")
    assert _paired_baseline_order(1, 0) == (
        "pipelined_vllm", "stock_vllm")
    assert _paired_baseline_order(1, 1) == (
        "stock_vllm", "pipelined_vllm")


def test_method_major_schedule_finishes_each_baseline_first():
    assert _baseline_schedule(
        ["IMDB-1", "IMDB-2"],
        ("stock_vllm", "pipelined_vllm"),
        rep=0,
        method_order="method-major",
    ) == [
        ("stock_vllm", "IMDB-1"),
        ("stock_vllm", "IMDB-2"),
        ("pipelined_vllm", "IMDB-1"),
        ("pipelined_vllm", "IMDB-2"),
    ]


def test_repeated_join_predicate_keeps_each_alias_pair():
    ground_truth = SimpleNamespace(
        key_for_template=lambda template: template)
    evaluator = SimpleNamespace(ground_truth=ground_truth)

    def join(template, left_alias, right_alias):
        predicate = SimpleNamespace(
            template=template,
            args=(SimpleNamespace(alias=left_alias),
                  SimpleNamespace(alias=right_alias)),
        )
        return SimpleNamespace(predicate=predicate)

    logical_joins = [
        join("same predicate", "r1", "a1"),
        join("same predicate", "r2", "a1"),
        join("other predicate", "r2", "a2"),
    ]
    records = [
        ("same predicate", "r1", "a1", "r1-a1 answers"),
        ("same predicate", "r2", "a1", "r2-a1 answers"),
        ("other predicate", "r2", "a2", "r2-a2 answers"),
    ]

    assert _join_records_by_written_position(
        evaluator, logical_joins, records) == {
            0: "r1-a1 answers",
            1: "r2-a1 answers",
            2: "r2-a2 answers",
        }


@pytest.mark.parametrize(("query_ids", "workload"), [
    (["IMDB-1", "IMDB-10"], "imdb"),
    (["BIO-1"], "biodex"),
    (["FEV-9"], "fever"),
    (["LEP-2", "LEP-8"], "lepard"),
])
def test_paired_auto_ground_truth_follows_query_set(query_ids, workload):
    assert _query_set_name(query_ids) == workload
    assert _paired_ground_truth_workload("auto", query_ids) == workload


def test_query_set_name_rejects_mixed_query_sets():
    with pytest.raises(ValueError, match="expected one query family"):
        _query_set_name(["IMDB-1", "BIO-1"])


def test_three_way_stock_join_prompts_match_quail():
    tokenizer = CharTokenizer()
    tok = _tok(tokenizer)
    template = "Do DOCUMENT {0}, DOCUMENT {1}, and DOCUMENT {2} agree?"
    args = tuple(ColumnRef(f"d{i}", f"d{i}", "document")
                 for i in range(3))
    bound = bind_join_prompt(template, args, tok)
    documents = (
        [tok("a0"), tok("a1")],
        [tok("b0")],
        [tok("c0"), tok("c1")],
    )

    prefixes, suffixes, members = build_join_grouped_inputs(
        bound, documents, anchor=1, tokenizer=tok)
    partner_slots = [0, 2]
    for anchor_idx, prefix in enumerate(prefixes):
        for suffix, member in zip(suffixes, members):
            indices = [None, anchor_idx, None]
            for slot, document_idx in zip(partner_slots, member):
                indices[slot] = document_idx
            tuple_documents = tuple(
                documents[slot][indices[slot]] for slot in range(3))
            assert prefix + suffix == render_join_prompt_ids(
                bound, tuple_documents, anchor=1, tokenizer=tok)

    assert members == list(itertools.product(range(2), range(2)))


def _tiled_order_tiles(order, n_suffixes):
    """Split a suffix-major-tiled order back into its anchor tiles."""
    tiles = []
    position = 0
    while position < len(order):
        tile = []
        while (position + len(tile) < len(order)
               and order[position + len(tile)][1] == 0):
            tile.append(order[position + len(tile)][0])
        assert tile, "each tile must start with suffix 0"
        expected = [
            (anchor_index, suffix_index)
            for suffix_index in range(n_suffixes)
            for anchor_index in tile
        ]
        assert order[position:position + len(expected)] == expected
        tiles.append(tile)
        position += len(expected)
    return tiles


def test_suffix_major_tiles_cover_all_pairs_within_budget():
    from baselines.stock import suffix_major_tiled_order

    prefixes = [[0] * length for length in (30, 30, 30, 50, 10, 90)]
    suffixes = [[0] * 5, [0] * 3]
    budget = 100

    order = suffix_major_tiled_order(prefixes, suffixes, budget)
    assert sorted(order) == sorted(
        (i, j) for i in range(len(prefixes))
        for j in range(len(suffixes)))

    tiles = _tiled_order_tiles(order, len(suffixes))
    assert [anchor for tile in tiles for anchor in tile] == list(
        range(len(prefixes)))
    for tile in tiles:
        cost = sum(len(prefixes[anchor]) + 5 for anchor in tile)
        assert cost <= budget or len(tile) == 1


def test_tiled_join_answers_match_anchor_major_order():
    from baselines.stock import run_join_grouped

    prefixes = [[100 + i] * (4 + i) for i in range(5)]
    suffixes = [[200 + j] * 3 for j in range(4)]

    class ParityLLM:
        def generate(self, prompts, _sp, use_tqdm=False):
            outs = []
            for prompt in prompts:
                ids = prompt["prompt_token_ids"]
                bit = 1 if (ids[0] + ids[-1]) % 2 == 0 else 0
                outs.append(SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[bit])],
                    prompt_token_ids=ids,
                    num_cached_tokens=0))
            return outs

    baseline = run_join_grouped(
        ParityLLM(), object(), prefixes, suffixes, {1})
    tiled = run_join_grouped(
        ParityLLM(), object(), prefixes, suffixes, {1},
        submission="suffix-major-tiled", tile_budget_tokens=20)

    assert baseline["answers"] == tiled["answers"]
    assert baseline["prompt_tokens"] == tiled["prompt_tokens"]
    assert tiled["submission"] == "suffix-major-tiled"

    with pytest.raises(ValueError):
        run_join_grouped(ParityLLM(), object(), prefixes, suffixes,
                         {1}, submission="suffix-major-tiled")
