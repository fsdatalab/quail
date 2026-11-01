"""Liquid's d1 decision models: prompt layout, answer ids, spec, and support."""

from types import SimpleNamespace

import pytest

from quail.baselines.request import RequestBackend
from quail.logical import ColumnRef, bind_prompt
from quail.logical.prompts import bind_join_prompt, render_join_prompt_text
from quail.planner.prompt_lowering import answer_ids
from quail.specs import D1_2B_BF16, H100_SXM

REVIEW = ColumnRef(alias="r", provider="reviews", column="body")
MOVIE = ColumnRef(alias="m", provider="movies", column="plot")
# d1-2B's prompt.render for a text state and a yes/no question, with the
# chat template's empty thinking block as the lead
D1_PROMPT = ("<|startoftext|><|im_start|>user\n{state}\n\n\nQUESTION:\n"
             "{question}\n\nReply with yes or no only.<|im_end|>\n"
             "<|im_start|>assistant\n<think></think>")
# text_config.layer_types of LiquidAI/d1-2B at the spec's revision
CHECKPOINT_LAYER_TYPES = (
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention",
    "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv")


def test_a_d1_filter_prompt_is_the_text_d1_was_trained_on():
    prompt = bind_prompt("Does the review praise the acting? {0}", (REVIEW,),
                         turn=D1_2B_BF16.turn, layout=D1_2B_BF16.prompt_layout)
    document = "The lead is superb."

    text = prompt.preamble + document + prompt.tail[len("{0}"):]

    assert text == D1_PROMPT.format(
        state=document, question="Does the review praise the acting?")


def test_a_d1_join_asks_its_question_after_both_documents():
    prompt = bind_join_prompt("Is {0} a review of the movie in {1}?",
                              (REVIEW, MOVIE), turn=D1_2B_BF16.turn,
                              layout=D1_2B_BF16.prompt_layout)

    text = render_join_prompt_text(prompt, ["Great film.", "A heist."], 0)

    assert prompt.frame == ""
    assert text == D1_PROMPT.format(
        state=("Great film.\n\n(The document above is DOCUMENT {0}.)"
               "\n\nDOCUMENT {1}:\nA heist."),
        question="Is {0} a review of the movie in {1}?")


def test_d1_answers_compare_its_one_token_yes_spellings_with_its_no_spellings():
    vocabulary = {"yes": [1], "Yes": [2], "YES": [3, 4], "no": [5],
                  "No": [6], "NO": [7]}

    true_ids, false_ids = answer_ids(D1_2B_BF16, vocabulary.__getitem__)

    assert (true_ids, false_ids) == ([1, 2], [5, 6, 7])


def test_the_d1_2b_spec_has_the_checkpoints_layer_kinds():
    expected = tuple("short_conv" if kind == "conv" else kind
                     for kind in CHECKPOINT_LAYER_TYPES)

    assert D1_2B_BF16.layer_kinds == expected
    assert D1_2B_BF16.sliding_layer_set == {4, 8, 12, 16, 20}


def test_the_vllm_baseline_refuses_d1_2b():
    engine = SimpleNamespace(label="stock vLLM", kind="vllm")
    backend = RequestBackend(name="stock", engine=engine,
                             filter_submission="operator")

    support = backend.supports(D1_2B_BF16, H100_SXM, 1)

    assert not support.supported
    assert "lfm3" in support.reason


def test_d1_2b_refuses_classify_extract_and_map_with_a_reason():
    from quail.planner.validation import (
        classification_refusal,
        extraction_refusal,
    )

    context = SimpleNamespace(model=D1_2B_BF16, tokenizer=lambda text: [1])

    classify = classification_refusal(context)
    extract = extraction_refusal(context)
    mapped = extraction_refusal(context, "AI.MAP")

    assert "option-code prompt" in classify.reasons[0]
    assert "document positions" in extract.reasons[0]
    assert "writes text" in mapped.reasons[0]


def test_a_set_attention_model_keeps_no_arena_or_checkpoints():
    from quail.cost import budgets

    chunk = budgets.chunk_budget(D1_2B_BF16, H100_SXM)

    assert budgets.arena_pages(D1_2B_BF16, H100_SXM) == (1, 1)
    assert budgets.checkpoint_memory_bytes(D1_2B_BF16, chunk) == 0
    assert chunk > 0


def test_a_lone_question_reads_its_whole_pass_both_ways():
    from quail.cost.work import packed, triangle

    work = packed(300, (50,), window=256)
    two = packed(300, (50, 40))

    assert work.tokens == 350
    assert work.pairs == 350 * 350
    assert work.sliding_pairs == triangle(350, 256)
    assert (work.state_written, work.state_read) == (0, 0)
    # the state reads all 390 tokens; each question, the state and itself
    assert two.pairs == 300 * 390 + 50 * 350 + 40 * 340


def test_a_set_attention_filter_costs_the_same_first_or_later():
    from quail.cost.filters import filter_cost

    first, later = (filter_cost(50, 300.0, D1_2B_BF16, H100_SXM, 8192,
                                first=flag) for flag in (True, False))

    assert first == later > 0


def test_set_attention_refuses_joins_and_a_score_beside_filters():
    from quail.planner.validation import set_attention_refusal

    joined = SimpleNamespace(joins=[object()], filters={})
    two = SimpleNamespace(joins=[], filters={"r": [object(), object()]})

    assert "joins" in set_attention_refusal(joined, False, D1_2B_BF16).reasons[0]
    assert "or a score" in set_attention_refusal(two, True, D1_2B_BF16).reasons[0]
    assert set_attention_refusal(two, False, D1_2B_BF16) is None


def _stages(selectivities, questions=50, state=300.0, longest=300):
    from quail.cost.packing import choose_stages

    count = len(selectivities)
    return choose_stages([questions] * count, selectivities, 5000.0, state,
                         longest, D1_2B_BF16, H100_SXM, 8192, set_size=16)


def test_a_filter_that_keeps_few_rows_gets_its_own_stage():
    assert _stages([0.02, 0.5, 0.5]) == (0, 1, 1)


def test_filters_that_keep_most_rows_share_one_pass():
    assert _stages([0.8, 0.8, 0.8]) == (0, 0, 0)


def test_a_pass_holds_at_most_the_trained_set_size():
    stages = _stages([1.0] * 20)

    assert max(stages) >= 1
    assert all(stages.count(stage) <= 16 for stage in set(stages))


def test_a_pass_suffix_opens_each_later_question_with_its_header():
    from quail.backends.quail.executor.operators.filter import set_pass_stage

    first = SimpleNamespace(state_tail_tokens=2, question_token_ids=(7, 8, 1, 2),
                            question_header_token_ids=(8,))
    second = SimpleNamespace(state_tail_tokens=2, question_token_ids=(7, 8, 3),
                             question_header_token_ids=(8,))

    stage = set_pass_stage([first, second], readout=None)

    assert stage.suffixes == [[1, 2, 8, 3]]
    assert stage.questions == (2, 2)
    assert stage.read_mask.tolist() == [False, True, False, True]


class _Pieces:
    """A tokenizer of fixed pieces, longest first: a mark joins a blank line."""

    pieces = {1: "a", 2: ".", 3: "\n\n", 4: ".\n\n", 5: "Q"}

    def __init__(self):
        self.calls = 0

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)

    def __call__(self, text):
        self.calls += 1
        ids = []
        while text:
            i, piece = max(((i, p) for i, p in self.pieces.items()
                            if text.startswith(p)), key=lambda e: len(e[1]))
            ids.append(i)
            text = text[len(piece):]
        return ids


def test_a_documents_last_mark_joins_the_blank_line_after_it_once_per_ending():
    from quail.backends.quail.executor.operators.filter import StateTails

    encoder = _Pieces()
    tails = StateTails(encoder)

    joined = [tails.prefix([1, 2], [3, 5]) for _ in range(3)]
    plain = tails.prefix([2, 1], [3, 5])

    assert joined == [[1, 4, 5]] * 3
    assert plain == [2, 1, 3, 5]
    assert encoder.calls == 2


def test_two_ai_filters_on_d1_2b_share_one_pass(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiFilter
    from quail.planner.plan import EngineConfig

    # documents much longer than a question, so one pass costs less
    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": [1, 2, 3], "body": [body] * 3}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT('Is it kind? {0}', r.body)) "
        "AND AI_FILTER(PROMPT('Is it short? {0}', r.body))").plan()

    filters = [node for node in plan.nodes if isinstance(node, AiFilter)]
    assert [node.stage for node in filters] == [0, 0]
    assert all(node.state_tail_tokens == 1 for node in filters)
    assert all(node.question_header_token_ids == ("QUESTION:",) for node in filters)
    session.close()


def _varlen_attention(torch):
    """FlashAttention's varlen call, in plain torch, returning (out, lse)."""
    def fa(q, k, v, cu_q, cu_k, max_q, max_k, causal, window=None):
        outs, lses = [], []
        groups = q.shape[1] // k.shape[1]
        for i in range(len(cu_q) - 1):
            qi = q[cu_q[i]:cu_q[i + 1]].float()
            ki = k[cu_k[i]:cu_k[i + 1]].float().repeat_interleave(groups, 1)
            vi = v[cu_k[i]:cu_k[i + 1]].float().repeat_interleave(groups, 1)
            scores = torch.einsum("thd,shd->hts", qi, ki) / qi.shape[-1] ** 0.5
            outs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), vi))
            lses.append(scores.logsumexp(-1))
        return torch.cat(outs), torch.cat(lses, dim=1)
    return fa


def test_set_attention_lets_the_state_read_every_question_and_each_question_itself():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.models.lfm3 import Lfm3Pipeline

    torch.manual_seed(0)
    # pass 0: rows 0-4 state, questions of 2, 3, and 1 rows; pass 1: 4
    # rows, one question of 2 rows
    passes = ((0, 5, (2, 3, 1)), (11, 13, (2,)))
    group = [-1] * 5 + [1, 1, 2, 2, 2, 3] + [-1] * 2 + [1, 1]
    sequence = [0] * 11 + [1] * 4
    rows, heads, kv_heads, dim = 15, 4, 2, 8
    q = torch.randn(rows, heads, dim)
    k, v = torch.randn(rows, kv_heads, dim), torch.randn(rows, kv_heads, dim)
    pipeline = Lfm3Pipeline.__new__(Lfm3Pipeline)
    pipeline.engine = SimpleNamespace(torch=torch, _fa=_varlen_attention(torch))
    meta = dict(passes=passes, cu_a=torch.tensor([0, 11, 15]), max_a=11)
    chunk = SimpleNamespace(input_ids=torch.zeros(rows, dtype=torch.long))
    meta["set_rows"] = pipeline._plan_set_rows(chunk, meta)

    out = pipeline._set_attention(q, k, v, meta)

    sees = torch.tensor([[sequence[i] == sequence[j] and (
        group[i] == -1 or group[j] == -1 or group[i] == group[j])
        for j in range(rows)] for i in range(rows)])
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, 1)) / dim ** 0.5
    weights = scores.masked_fill(~sees, float("-inf")).softmax(-1)
    expected = torch.einsum("hts,shd->thd", weights, v.repeat_interleave(2, 1))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
