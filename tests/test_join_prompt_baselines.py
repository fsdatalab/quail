import itertools
from types import SimpleNamespace

import pytest

from baselines.stock import build_join_grouped_inputs, run_filter_chain
from baselines.old_stock.operators import Join
from baselines.stock_vllm.run import (
    _baseline_configuration,
    _filter_chain_inputs,
    _filter_prompts,
    _run_join,
    _select_join_anchor,
    _split_query_sets,
    _vllm_filter_capacity,
)
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
    assert _baseline_configuration("stock_vllm") == "stage-major"
    assert _baseline_configuration("pipelined_vllm") == "pipelined"
    with pytest.raises(ValueError, match="unknown baseline"):
        _baseline_configuration("other")


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

    def fake_join(_llm, _sp, prefixes, suffixes, _true_set):
        assert len(prefixes) == 2
        assert len(suffixes) == 2
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


def test_stock_query_set_split_uses_one_chunk_per_set():
    ids = ["IMDB-1", "IMDB-5", "BIO-1", "FEV-9", "LEP-3"]
    assert _split_query_sets(ids) == [
        ["IMDB-1", "IMDB-5"],
        ["BIO-1"],
        ["FEV-9"],
        ["LEP-3"],
    ]


@pytest.mark.parametrize("anchor", [0, 1])
def test_stock_and_naive_vllm_join_prompts_match_quail(anchor):
    tokenizer = CharTokenizer()
    tok = _tok(tokenizer)
    template = ("Does DOCUMENT {0} support the statement in "
                "DOCUMENT {1}?")
    left = ["left a", "left b"]
    right = ["right a", "right b", "right c"]
    args = (ColumnRef("left", "left", "document"),
            ColumnRef("right", "right", "document"))
    bound = bind_join_prompt(template, args, tok)
    documents = ([tok(text) for text in left],
                 [tok(text) for text in right])

    prefixes, suffixes, _ = build_join_grouped_inputs(
        bound, documents, anchor, tok)
    stock_prompts = [prefix + suffix
                     for prefix in prefixes for suffix in suffixes]
    naive_prompts, pairs = Join(
        "test", template, anchor=anchor).build_prompts(
            left, right, tokenizer)

    assert naive_prompts == stock_prompts
    assert len(pairs) == len(stock_prompts)
    for prompt_ids, (left_idx, right_idx) in zip(naive_prompts, pairs):
        expected = render_join_prompt_ids(
            bound,
            (documents[0][left_idx], documents[1][right_idx]),
            anchor,
            tok,
        )
        assert prompt_ids == expected


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
