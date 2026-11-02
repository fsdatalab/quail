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


def test_d1_2b_fp8_is_d1_2b_with_fp8_weights():
    from dataclasses import fields

    from quail.specs import D1_2B_FP8

    differ = {field.name for field in fields(D1_2B_BF16)
              if getattr(D1_2B_BF16, field.name) != getattr(D1_2B_FP8, field.name)}

    assert differ == {"name", "w_bytes", "weight_precision", "w_mem_bytes"}
    assert D1_2B_FP8.weight_precision == "fp8"


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


def test_set_attention_refuses_a_score_beside_filters():
    from quail.planner.validation import set_attention_refusal

    two = SimpleNamespace(joins=[], filters={"r": [object(), object()]},
                          labels=SimpleNamespace(calls=[]))

    assert "or a score" in set_attention_refusal(two, True, D1_2B_BF16).reasons[0]
    assert set_attention_refusal(two, False, D1_2B_BF16) is None


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
    # priced by its own estimate over each anchor's kept pairs, not as a
    # pass group of one question for each pair
    assert label.pass_group == -1
    assert plan.estimates[label.node_id]["seconds"] == pytest.approx(
        spec.estimated_seconds)
    session.close()
