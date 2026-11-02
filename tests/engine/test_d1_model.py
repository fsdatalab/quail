"""Liquid's d1 decision models: prompt layout, answer ids, spec, and support."""

import math
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


def test_a_d1_join_puts_the_anchor_in_the_state_and_its_partner_in_the_question():
    prompt = bind_join_prompt(
        "Is the review in DOCUMENT {0} about the movie in {1}?",
        (REVIEW, MOVIE), turn=D1_2B_BF16.turn, layout=D1_2B_BF16.prompt_layout)

    text = render_join_prompt_text(prompt, ["Great film.", "A heist."], 0)

    assert text == D1_PROMPT.format(
        state="reviews record:\nGreat film.",
        question=("movies record:\nA heist.\n\nIs the review in the reviews "
                  "record about the movie in the movies record?"))
    assert prompt.tail_segments[0] == "\n\n"


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


def test_d1_2b_refuses_map_with_a_reason():
    from quail.planner.validation import extraction_refusal

    context = SimpleNamespace(model=D1_2B_BF16, tokenizer=lambda text: [1])

    assert extraction_refusal(context) is None
    assert "writes text" in extraction_refusal(context, "AI.MAP").reasons[0]


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


def test_set_attention_refuses_a_score_beside_filters():
    from quail.planner.validation import set_attention_refusal

    two = SimpleNamespace(joins=[], filters={"r": [object(), object()]},
                          labels=SimpleNamespace(calls=[]))

    assert "or a score" in set_attention_refusal(two, True, D1_2B_BF16).reasons[0]
    assert set_attention_refusal(two, False, D1_2B_BF16) is None


def test_a_set_join_costs_each_anchors_passes_with_packed():
    from quail.cost.joins import PricedJoin, packed_join_work
    from quail.cost.work import packed
    from quail.planner.statistics.tables import AliasStats

    join = PricedJoin(written_position=0, aliases=("r", "a"), anchor="r",
                      anchor_free=False, semantics="full", selectivity=None,
                      frame_tokens={"r": 3, "a": 3},
                      label_tokens={"r": 2, "a": 2}, tail_tokens=20)
    lengths = {"r": AliasStats(count=10, total=3000, squared=900_000,
                               maximum=300),
               "a": AliasStats(count=20, total=200, squared=2000, maximum=10)}

    # 20 partners per anchor: two passes of 10 questions of 32 tokens
    work = packed_join_work(join, "r", 10, 200, lengths, 5, set_size=16)
    one = packed(310, (32,) * 10)

    assert work.tokens == pytest.approx(20 * one.tokens)
    assert work.pairs == pytest.approx(20 * one.pairs)
    assert work.kv_read == pytest.approx(20 * one.kv_read)


def test_a_join_pass_holds_at_most_the_set_size_and_fits_one_chunk():
    from quail.backends.quail.executor.set_passes import split_passes

    lengths = [10] * 20

    assert split_passes(100, range(20), lengths, 1, 16, 10_000) == [
        list(range(16)), list(range(16, 20))]
    # 100 + 10 + 3 * 11 = 143 fits; a fifth question does not
    assert split_passes(100, range(5), lengths, 1, 16, 150) == [
        [0, 1, 2, 3], [4]]
    with pytest.raises(ValueError, match="does not fit"):
        split_passes(100, [0], lengths, 1, 16, 105)


def test_a_join_question_labels_each_partner_and_joins_its_end_to_the_tail():
    from quail.backends.quail.executor.operators.join import set_join_questions
    from quail.backends.quail.executor.set_passes import StateTails

    node = SimpleNamespace(partners=("a",), label_token_ids=(("a", (9,)),),
                           tail_token_ids=(3, 5), separator_tokens=1)

    questions = set_join_questions(node, {"a": [[1, 2], [2, 1]]},
                                   StateTails(_Pieces()))

    assert questions == [[9, 1, 4, 5], [9, 2, 1, 3, 5]]


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
    assert stage.questions == ((2, 2),)


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
    from quail.backends.quail.executor.set_passes import StateTails

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


def test_a_d1_join_makes_the_longer_table_the_state(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiJoin
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    pq.write_table(pa.table({"id": list(range(12)), "aspect": ["acting"] * 12}),
                   tmp_path / "aspects.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    for name in ("reviews", "aspects"):
        session.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, a.id FROM aspects a JOIN reviews r ON AI_FILTER(PROMPT("
        "'Does {1} discuss {0}?', a.aspect, r.body))").plan()

    (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
    assert join.anchor == "r"
    assert join.document_head[-2:] == ("reviews", "record:")
    assert join.label_token_ids == (("a", ("aspects", "record:")),)
    assert join.question_header_token_ids == ("QUESTION:",)
    assert not join.keep_kv
    session.close()


def test_a_d1_classification_lists_its_options_by_code_as_d1_does():
    from quail.logical.prompts import bind_classify_prompt

    prompt = bind_classify_prompt(
        "Which sentiment does the review express?\n\n{0}", (REVIEW,),
        ["positive", "negative"], ("", "the review is negative"),
        turn=D1_2B_BF16.turn, layout=D1_2B_BF16.prompt_layout)

    text = prompt.preamble + "Great film." + prompt.tail[len("{0}"):]

    assert text == D1_PROMPT.format(
        state="Great film.",
        question=("Which sentiment does the review express?\n\nOptions:\n"
                  "A positive\nB the review is negative"),
    ).replace("Reply with yes or no only.", "Reply with the option code only.")


def test_an_option_whose_code_is_two_tokens_takes_d1s_first_free_fallback():
    from quail.logical.prompts import option_codes
    from quail.planner.prompt_lowering import option_aliases

    def tokenizer(text):
        return list(text) if text.isupper() else [text]

    codes = option_codes([f"label {index}" for index in range(27)])

    assert codes[-1] == "AA"
    assert option_aliases(codes, tokenizer)[-1] == "00"


def test_an_option_scores_its_best_spelling_over_the_temperature():
    from quail.backends.quail.executor.operators.set_classify import (
        option_probabilities,
    )

    logits = [2.0, 4.0, 1.0]  # A, " A", B
    probabilities = option_probabilities(logits, {7: 0, 8: 1, 9: 2},
                                         ((7, 8), (9,)), 2.0)

    expected = math.exp(4.0 / 2) / (math.exp(4.0 / 2) + math.exp(1.0 / 2))
    assert probabilities[0] == pytest.approx(expected)


def test_two_d1_classifications_plan_option_code_questions(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": [1, 2, 3], "body": [body] * 3}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), "
        "ARRAY['happy', 'sad']) AS mood, AI.CLASSIFY(PROMPT('Which length? "
        "{0}', r.body), ARRAY['short', 'long']) AS size FROM reviews r").plan()

    nodes = [node for node in plan.nodes if isinstance(node, AiClassify)]
    assert [node.spec.scoring for node in nodes] == ["set_choice"] * 2
    assert all(node.spec.label_token_ids == (("A",), ("B",)) for node in nodes)
    assert all(node.spec.state_tail_tokens == 1 for node in nodes)
    assert all(node.spec.estimated_seconds > 0 for node in nodes)
    session.close()


def test_a_d1_extraction_asks_for_a_read_with_start_and_end_slots():
    from quail.logical.prompts import bind_extract_prompt

    prompt = bind_extract_prompt((REVIEW,), "The film's director",
                                 D1_2B_BF16.turn, D1_2B_BF16.prompt_layout)

    text = prompt.preamble + "Great film." + prompt.tail[len("{0}"):]

    assert text == (
        "<|startoftext|><|im_start|>user\nGreat film.\n\n\nQUESTION:\n"
        "The film's director\n\nRead the text from the state, or nothing if it "
        "is absent.<|im_end|>\n<|im_start|>assistant\n<think></think>"
        "start:\nend:")


def test_a_read_slot_is_the_token_holding_its_colon():
    from quail.planner.prompt_lowering import read_slot_offsets

    pieces = ("x", "start", ":\n", ":", "end", "\n")

    def tokenizer(text):
        ids = []
        while text:
            piece = max((p for p in pieces if text.startswith(p)), key=len)
            ids.append(pieces.index(piece))
            text = text[len(piece):]
        return ids

    # the whole segment joins the start colon with the line break
    assert read_slot_offsets("xstart:\nend:", ("start", "end"),
                             tokenizer) == (-2, 0)


def test_a_found_probability_follows_the_calibration_knots():
    from quail.backends.quail.executor.operators.set_extract import calibrated

    knots = ((0.1, 0.0), (0.5, 0.4), (0.9, 1.0))

    assert calibrated(knots, 0.05) == 0.0
    assert calibrated(knots, 0.3) == pytest.approx(0.2)
    assert calibrated(knots, 0.95) == 1.0
    assert calibrated((), 0.3) == 0.3


def test_a_read_returns_its_tokens_characters_without_edge_spaces():
    from quail.backends.quail.executor.operators.set_extract import token_span

    text = "Paid to Zoë Ltd. now"
    vocabulary = {1: b"Paid", 2: b" to", 3: b" Zo\xc3\xab", 4: b" Ltd.", 5: b" now"}

    assert token_span(text, [1, 2, 3, 4, 5], vocabulary, 2, 3) == (
        "Zoë Ltd.", (8, 16))


def test_a_pointer_read_takes_the_likeliest_span_or_nothing():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.readout import AsyncSlotPointers

    readout = AsyncSlotPointers(torch, torch.nn.functional, 2, 512)
    width = 4
    state = torch.eye(5, width) * 0.0
    state[2, 0] = state[3, 1] = 1.0   # rows 2 and 3 answer the first read
    state[0, 2] = 1.0                 # row 0 stands for nothing
    slots = torch.zeros(4, width)
    slots[0, 0] = slots[1, 1] = 40.0  # read 1: start at row 2, end at row 3
    slots[2, 2] = slots[3, 2] = 40.0  # read 2: both slots on row 0

    spans = readout._spans(torch.cat((state, slots)))

    assert spans[0, :2].tolist() == [2.0, 3.0] and spans[0, 2] > 0.99
    assert spans[1, :2].tolist() == [0.0, 0.0] and spans[1, 2] < 0.01


def test_a_d1_extraction_plans_a_read(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiExtract
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": [1, 2, 3], "text": [body] * 3}),
                   tmp_path / "contracts.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("contracts", DocumentProvider.from_parquet(
        str(tmp_path / "contracts.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT c.id, AI.EXTRACT(c.text, {'signed': 'The date of the contract'}) "
        "AS terms FROM contracts c").plan()

    (node,) = [node for node in plan.nodes if isinstance(node, AiExtract)]
    # str.split keeps "<think></think>start:" whole, one token before "end:"
    assert node.spec.read_slots == (-1, 0)
    assert node.spec.state_tail_tokens == 1
    assert node.spec.max_answer_tokens == 512
    assert node.spec.estimated_seconds > 0
    session.close()


def test_a_d1_plan_prices_and_explains_the_passes_its_operators_share(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    session.register("reviews", DocumentProvider.from_parquet(
        str(tmp_path / "reviews.parquet"), id_col="id"))
    query = session.sql(
        "SELECT r.id, AI.CLASSIFY(PROMPT('Which mood? {0}', r.body), "
        "ARRAY['happy', 'sad']) AS mood, AI.CLASSIFY(PROMPT('Which size? {0}', "
        "r.body), ARRAY['short', 'long']) AS size FROM reviews r")

    plan = query.plan()
    first, second = [node.node_id for node in plan.nodes
                     if isinstance(node, AiClassify)]
    text = query.explain()

    assert plan.estimates[first]["pass_members"] == (first, second)
    assert plan.estimates[first]["seconds"] > 0
    assert "seconds" not in plan.estimates[second]
    assert f"Passes on r: [{first}, {second}]" in text
    assert "paged KV" not in text and "admission budget" not in text
    session.close()


def test_a_joined_d1_classification_asks_about_the_partner_record():
    from quail.logical.prompts import bind_classify_prompt

    aspect = ColumnRef(alias="a", provider="aspects", column="aspect")
    prompt = bind_classify_prompt(
        "What does the review in DOCUMENT {0} say about DOCUMENT {1}?",
        (REVIEW, aspect), ["praise", "complaint"], (),
        turn=D1_2B_BF16.turn, layout=D1_2B_BF16.prompt_layout)

    text = render_join_prompt_text(prompt, ["Great film.", "the acting"], 0)

    assert text == D1_PROMPT.format(
        state="reviews record:\nGreat film.",
        question=("aspects record:\nthe acting\n\nWhat does the review in "
                  "the reviews record say about the aspects record?\n\n"
                  "Options:\nA praise\nB complaint"),
    ).replace("Reply with yes or no only.", "Reply with the option code only.")


def test_a_d1_classification_of_joined_rows_follows_its_join(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import quail
    from quail.catalog import DocumentProvider
    from quail.physical import AiClassify, AiJoin
    from quail.physical.pipelines import build_pipelines
    from quail.planner.plan import EngineConfig

    body = " ".join(["word"] * 300)
    pq.write_table(pa.table({"id": list(range(40)), "body": [body] * 40}),
                   tmp_path / "reviews.parquet")
    pq.write_table(pa.table({"id": list(range(12)), "aspect": ["acting"] * 12}),
                   tmp_path / "aspects.parquet")
    session = quail.Session(EngineConfig(model="d1-2b-bf16", device="h100-sxm"),
                            tokenizer=str.split)
    for name in ("reviews", "aspects"):
        session.register(name, DocumentProvider.from_parquet(
            str(tmp_path / f"{name}.parquet"), id_col="id"))

    plan = session.sql(
        "SELECT r.id, a.id, AI.CLASSIFY(PROMPT('What does DOCUMENT {0} say "
        "about DOCUMENT {1}?', r.body, a.aspect), ARRAY['praise', 'complaint']) "
        "AS stance FROM reviews r JOIN aspects a ON AI_FILTER(PROMPT("
        "'Does {0} mention {1}?', r.body, a.aspect))").plan()

    (join,) = [node for node in plan.nodes if isinstance(node, AiJoin)]
    (label,) = [node for node in plan.nodes if isinstance(node, AiClassify)]
    spec = label.spec
    assert spec.scoring == "set_choice" and spec.aliases == ("r", "a")
    assert spec.join_layout == (("QUESTION:",), ("aspects", "record:"))
    assert spec.prompt_token_parts[0][-2:] == ("reviews", "record:")
    assert build_pipelines(plan.graph)[join.node_id] == (join, label)
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
