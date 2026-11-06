"""Quail's d1-2B forward pass against Liquid's own code, on a GPU.

Liquid's code for d1-2B needs transformers 5.19 or later, newer than
the project pins, so it runs in a child process with that version on
its path, from a folder the GPU test image installs it into. The child
renders each review and question with d1's own prompt code, so the
test checks Quail's prompt text against it. It then runs the
checkpoint's own model with d1's set attention over Quail's token ids,
through d1's own slot readout, and records the logits of the yes and
no spellings at the answer slot. With one question, set attention
reads the whole prompt in both directions.

Quail runs the same prompts through its stage executor, one question
per pass: each review and question is one unpaged sequence, with no KV
kept. The two sides run different attention and convolution kernels
in bf16, so the logits differ slightly; the test checks that the
yes-against-no margins agree in sign and closely in value.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword d1`.
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

MODEL = "d1-2b-bf16"
BUDGET = 8192
# transformers 5.19, installed by experiments/run_gpu_tests.py's image
REFERENCE_PACKAGES = "/opt/d1-reference"
QUESTIONS = ["Does the review praise the acting? {0}",
             "Does the reviewer recommend the film? {0}"]
REVIEWS = [
    "The acting was superb, every scene carried by the lead's quiet "
    "intensity. The script drags in the middle but the final act "
    "lands with real weight.",
    "Two hours I will never get back. Wooden performances, a plot "
    "that makes no sense, and a soundtrack that never stops.",
    "A charming little film. The cast has chemistry, the jokes are "
    "gentle, and the ending is earned. Nothing more, nothing less.",
    "Visually stunning but hollow. The leads recite their lines as if "
    "reading a manual; only the cinematography deserves praise.",
    "I laughed, I cried, and I bought the ticket twice. The ensemble "
    "cast is the best I have seen this year, especially the villain.",
    "Loud, long, and lifeless. The stunts are impressive; the "
    "performances are not. Skip it unless explosions are enough.",
]
# one pass of 16 questions, d1-2B's trained set size
ASPECTS = ["the acting", "the plot", "the music", "the ending", "the script",
           "the cast", "the villain", "the pacing", "the camera work",
           "the dialogue", "the stunts", "the soundtrack", "the director",
           "the length", "the jokes", "the price of the ticket"]
PACKED = [f"Does the review mention {aspect}? {{0}}" for aspect in ASPECTS]
# a join of reviews and aspects: 18 aspects make two passes per review
JOIN = ("Does the review in DOCUMENT {0} discuss the movie aspect in "
        "DOCUMENT {1}?")
JOIN_ASPECTS = [*ASPECTS, "the sets", "the costumes"]
# two reads packed in one pass: some reviews hold the answer, some do not
READS = ["The part of the cast the reviewer singles out",
         "The number of times the reviewer bought a ticket"]
# three classifications packed in one pass; the second describes its labels
CHOICES = [
    ("Judge strictly from the review above which category best describes "
     "the overall sentiment it expresses about the movie.\n\n{0}",
     ["positive", "negative", "neutral", "mixed"], ()),
    ("Judge strictly from the review above how it rates the acting.\n\n{0}",
     ["strong", "weak", "not_mentioned"],
     ("the acting is praised", "the acting is criticized", "")),
    ("Judge strictly from the review above whether the reviewer would watch "
     "the film again.\n\n{0}", ["yes", "no", "unclear"], ()),
]
# the largest difference between the two sides' yes-against-no logit
# margins this test accepts, in logits
MARGIN_TOLERANCE = 0.5
# the largest difference between Quail's P(yes) and d1's own, which
# tokenizes the prompt whole: Quail tokenizes the document apart from
# the question, so a document's last character can be its own token
PROBABILITY_TOLERANCE = 0.05


def _spec():
    from quail.specs import MODELS

    return MODELS[MODEL]


def _tokenizer():
    from quail.common.tokens import Tokenizer
    from quail.specs.files import resolve_model_path

    spec = _spec()
    return Tokenizer(resolve_model_path(spec.hf_name, spec.revision))


def _prompts():
    """Per question, the lowered prompt and the text and ids of each review."""
    from quail.api.prompts import bind_prompt
    from quail.physical.lowering import (
        lower_prompt,
        render_filter_prompt_ids,
    )

    spec, ids = _spec(), _tokenizer()
    out = []
    for question in QUESTIONS:
        prompt = bind_prompt(question, ("body",), turn=spec.turn,
                             layout=spec.prompt_layout)
        lowered = lower_prompt(prompt, ids)
        out.append((prompt, lowered, [
            (prompt.preamble + review + prompt.tail[len("{0}"):],
             render_filter_prompt_ids(lowered, ids(review)))
            for review in REVIEWS]))
    return out


def reference_main(request_path, out_path):
    """Run d1-2B's own code over the prompts in request_path.

    Writes, per prompt, d1's rendered text and token ids, the logits of
    every yes and no spelling at the slot of Quail's ids under set
    attention, and d1's own P(yes) from its own ids.
    """
    from transformers import AutoModel, AutoTokenizer

    request = json.loads(open(request_path).read())
    model = AutoModel.from_pretrained(
        request["hf_name"], revision=request["revision"],
        trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    package = type(model).__module__.rsplit(".", 1)[0]
    runner = sys.modules[package + ".runner"]
    prompt = sys.modules[package + ".prompt"]
    slots = sys.modules[package + ".slots"]
    tokenizer = AutoTokenizer.from_pretrained(
        request["hf_name"], revision=request["revision"])
    engine = runner.SystemOne(model=model, tokenizer=tokenizer)
    if engine.attention != "set":
        raise ValueError(f"d1-2B serves {engine.attention!r} attention, not set")
    ids = request["answer_ids"]
    rows = []
    with torch.inference_mode():
        for item in request["prompts"]:
            question = prompt.Noul(item["question"])
            text = engine.render(item["review"], question)
            quail_ids = torch.tensor([item["ids"]], device="cuda")
            n = quail_ids.shape[1]
            # one question: the state is the whole prompt, read both ways
            codes = slots.slot_logits(
                model, quail_ids, torch.ones_like(quail_ids),
                torch.tensor([n - 1]), torch.tensor([n]), None).codes
            rows.append(dict(
                text=text,
                ids=tokenizer.encode(text, add_special_tokens=False),
                logits=codes[0].float()[ids].tolist(),
                p_yes=engine.probabilities(item["review"], [question])[0][0]))
        packs = []
        for item in request["packs"]:
            questions = [prompt.Noul(text) for text in item["questions"]]
            pack_ids = torch.tensor([item["ids"]], device="cuda")
            n = pack_ids.shape[1]
            codes = slots.slot_logits(
                model, pack_ids, torch.ones_like(pack_ids),
                torch.tensor(item["slots"]), torch.tensor([item["state"]]),
                torch.tensor([item["groups"]]),
                torch.zeros(len(item["slots"]), dtype=torch.long)).codes
            (sequence,) = engine._sequences([(item["review"], questions, ())])
            packs.append(dict(
                rows=n, d1_ids=list(sequence[2]),
                logits=codes.float()[:, ids].tolist(),
                p_yes=[p[0] for p in engine.probabilities(
                    item["review"], questions)]))
        # the served runner, with the checkpoint's calibrated temperature
        served = model.engine
        if served.calibration is None:
            raise ValueError("d1-2B serves no calibration.json")
        choices = []
        for item in request["choices"]:
            questions = [prompt.Choice(question["instructions"],
                                       question["criteria"])
                         for question in item["questions"]]
            choices.append(dict(
                text=engine.render(item["review"], questions[0]),
                probabilities=served.probabilities(item["review"], questions)))
        reads = []
        for review in request["reads"]:
            questions = [prompt.Read(question) for question in READS]
            reads.append(served.probabilities(review, questions))
        mixed = []
        for review in request["reads"]:
            question, labels, descriptions = CHOICES[0]
            questions = [
                prompt.Noul(QUESTIONS[0].removesuffix(" {0}")),
                prompt.Choice(question.split("\n\n{0}")[0], {
                    label: (descriptions[index] if descriptions else "") or None
                    for index, label in enumerate(labels)}),
                prompt.Read(READS[0])]
            # yes/no from the uncalibrated runner, as the packed tests;
            # the choice and the read from the served one
            p_yes = engine.probabilities(review, questions)[0][0]
            _, choice, read = served.probabilities(review, questions)
            mixed.append(dict(p_yes=p_yes, choice=choice, read=read))
    with open(out_path, "w") as f:
        json.dump(dict(prompts=rows, packs=packs, choices=choices,
                       reads=reads, mixed=mixed), f)


@pytest.fixture(scope="module")
def prompts():
    return _prompts()


@pytest.fixture(scope="module")
def answer_ids():
    from quail.physical.lowering import answer_ids as ids

    true_ids, false_ids = ids(_spec(), _tokenizer())
    return true_ids, false_ids


def _encoder():
    from quail.common.tokens import Tokenizer
    from quail.specs.files import resolve_model_path

    spec = _spec()
    return Tokenizer(resolve_model_path(spec.hf_name, spec.revision))


def _ends(head, tail, documents):
    """The plan's joins of these documents' endings with a tail."""
    from quail.common.document_ends import document_ends

    last = {int(document[-1]) for document in documents} | set(head[-1:])
    return document_ends(sorted(last), tail, _encoder())


def _yes_no_questions(templates):
    """AI filter templates as the SetQuestions of one pass group."""
    from quail.api.prompts import bind_prompt
    from quail.backends.quail.executor.pass_groups import SetQuestion
    from quail.physical.lowering import lower_prompt, set_question_parts

    spec, ids = _spec(), _tokenizer()
    questions = []
    for template in templates:
        lowered = lower_prompt(bind_prompt(template, ("body",), turn=spec.turn,
                                           layout=spec.prompt_layout), ids)
        parts = set_question_parts(lowered, spec, ids)
        tail, cut = tuple(lowered.tail_token_ids), parts["state_tail_tokens"]
        questions.append(SetQuestion(
            "yes_no", tuple(lowered.preamble_token_ids), tail[:cut], tail[cut:],
            parts["question_header_token_ids"]))
    return questions


def _packs():
    """Per review, one pass of the 16 PACKED questions as Quail lays it out.

    Returns the questions, and per review the pass's token ids, its state
    length, each token's question (-1 in the state, then 1 to 16), and
    each question's slot.
    """
    from quail.backends.quail.executor.set_passes import group_passes
    from quail.backends.quail.executor.stages import question_ends

    ids = _tokenizer()
    questions = _yes_no_questions(PACKED)
    from quail.common.document_ends import joined_state

    first = questions[0]
    ends = dict(_ends(first.head, first.state_tail, [ids(r) for r in REVIEWS]))
    out = []
    for review in REVIEWS:
        prefix = list(joined_state(list(first.head) + ids(review),
                                   first.state_tail, ends))
        ((_, suffix, shape),) = group_passes(
            [prefix], [q.tokens for q in questions], first.header, 16, BUDGET)
        state = len(prefix)
        groups = [-1] * state + [number for number, length in enumerate(shape, 1)
                                 for _ in range(length)]
        slots = [state + int(index) for index in question_ends(shape).nonzero()[0]]
        out.append(dict(review=review, ids=prefix + suffix, state=state,
                        groups=groups, slots=slots,
                        questions=[q.removesuffix(" {0}") for q in PACKED],
                        prefix=prefix))
    return questions, out


@pytest.fixture(scope="module")
def packs():
    return _packs()


def _join_passes():
    """Per review, its join passes over JOIN_ASPECTS as Quail lays them out.

    Returns the passes' Stage, each pass's state ids, and per pass the
    entry the reference reads: d1's state and question texts, and the
    pass's ids, state length, token questions, and slots.
    """
    from types import SimpleNamespace

    from quail.api.prompts import bind_join_prompt
    from quail.backends.quail.executor.operators.join import join_pass_stage
    from quail.backends.quail.executor.stages import question_ends
    from quail.backends.quail.lowering import set_join_parts
    from quail.common.expressions import ColumnRef
    from quail.physical.lowering import lower_prompt

    spec, ids = _spec(), _tokenizer()
    prompt = bind_join_prompt(JOIN, (
        ColumnRef(alias="r", provider="reviews", column="body"),
        ColumnRef(alias="a", provider="aspects", column="aspect")),
        turn=spec.turn, layout=spec.prompt_layout)
    lowered = lower_prompt(prompt, ids)
    (_, review_label, frame), (_, aspect_label, _) = lowered.label_token_ids
    parts = set_join_parts(lowered, spec, ids)
    separator = tuple(lowered.tail_token_ids[:parts["separator_tokens"]])
    head = list(lowered.preamble_token_ids) + list(review_label)
    aspects = [ids(aspect) for aspect in JOIN_ASPECTS]
    node = SimpleNamespace(
        partners=("a",), label_token_ids=(("a", aspect_label),),
        frame_token_ids=frame, tail_token_ids=lowered.tail_token_ids,
        anchor="r", document_ends=(
            ("r", tuple(frame), _ends(head, frame, [ids(r) for r in REVIEWS])),
            ("a", separator, _ends((), separator, aspects))), **parts)
    prefixes = [head + ids(review) for review in REVIEWS]
    stage, states, owners = join_pass_stage(
        node, prefixes, range(len(REVIEWS)), lambda anchor: None,
        {"a": aspects}, None, spec.set_size, BUDGET)
    states = [list(state) for state in states]
    question = prompt.tail[len("\n\n"):].split("\n\nReply")[0]
    entries, first = [], 0
    for index, (state, anchor) in enumerate(zip(states, owners)):
        shape = stage.questions[index]
        if index and owners[index - 1] != anchor:
            first = 0
        aspects = JOIN_ASPECTS[first:first + len(shape)]
        first += len(shape)
        entries.append(dict(
            review="reviews record:\n" + REVIEWS[anchor],
            ids=state + stage.suffixes[index], state=len(state),
            groups=[-1] * len(state) + [number for number, length
                                        in enumerate(shape, 1)
                                        for _ in range(length)],
            slots=[len(state) + int(row)
                   for row in question_ends(shape).nonzero()[0]],
            questions=[f"aspects record:\n{aspect}\n\n{question}"
                       for aspect in aspects]))
    return stage, states, entries


@pytest.fixture(scope="module")
def join_passes():
    return _join_passes()


@pytest.fixture(scope="module")
def reference(prompts, packs, join_passes, answer_ids, tmp_path_factory):
    """Per prompt, question by question, what d1-2B's own code returns."""
    spec = _spec()
    if not os.path.isdir(REFERENCE_PACKAGES):
        pytest.fail(f"d1's reference needs transformers 5.19 in "
                    f"{REFERENCE_PACKAGES}; run through run_gpu_tests.py")
    folder = tmp_path_factory.mktemp("d1")
    request = folder / "request.json"
    request.write_text(json.dumps(dict(
        hf_name=spec.hf_name, revision=spec.revision,
        answer_ids=list(answer_ids[0]) + list(answer_ids[1]),
        prompts=[dict(review=review, question=question.removesuffix(" {0}"),
                      ids=ids)
                 for question, (_, _, items) in zip(QUESTIONS, prompts)
                 for review, (_, ids) in zip(REVIEWS, items)],
        packs=[{key: pack[key] for key in
                ("review", "ids", "state", "groups", "slots", "questions")}
               for pack in [*packs[1], *join_passes[2]]],
        choices=[dict(review=review, questions=[
            dict(instructions=question.split("\n\n{0}")[0],
                 criteria={label: (descriptions[index] if descriptions else "")
                           or None for index, label in enumerate(labels)})
            for question, labels, descriptions in CHOICES])
            for review in REVIEWS],
        reads=REVIEWS)))
    out = folder / "reference.json"
    environment = dict(os.environ, PYTHONPATH=REFERENCE_PACKAGES)
    subprocess.run([sys.executable, __file__, str(request), str(out)],
                   check=True, env=environment)
    return json.loads(out.read_text())


def _load(answer_ids):
    """Quail's d1-2B model and pipeline, with the yes and no rows kept."""
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.common import budgets
    from quail.specs import H100_SXM

    spec, device = _spec(), H100_SXM
    model = load_model(spec.hf_name, revision=spec.revision,
                       answer_token_ids=list(answer_ids[0]) + list(answer_ids[1]))
    full, sliding = budgets.arena_pages(spec, device, BUDGET)
    arena = KVArena(num_layers=spec.layers, num_pages=full,
                    page_tokens=device.page_tokens,
                    num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                    dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    num_sliding_pages=sliding)
    return model, arena, build_pipeline(spec, model, arena)


@pytest.fixture(scope="module")
def loaded(answer_ids, reference):
    """Built after the reference so the two model copies never share the GPU."""
    return _load(answer_ids)


@pytest.fixture(scope="module")
def quail(prompts, answer_ids, loaded):
    """Per prompt, Quail's logits of the yes and no spellings, one question a pass."""
    import numpy as np

    from quail.backends.quail.executor.model_loading import full_output_head
    from quail.backends.quail.executor.readout import AsyncLabelLogprobs
    from quail.backends.quail.executor.stages import Stage, run_stages

    model, arena, pipeline = loaded
    targets = list(answer_ids[0]) + list(answer_ids[1])
    budget = BUDGET
    readout = AsyncLabelLogprobs(torch, torch.nn.functional,
                                 full_output_head(model), targets,
                                 normalize=False)
    ids = _tokenizer()
    rows = []
    for _, lowered, _ in prompts:
        # one pass per review: the preamble and review, then the question
        documents = [list(lowered.preamble_token_ids) + ids(review)
                     for review in REVIEWS]
        stage = Stage(suffixes=[list(lowered.tail_token_ids)], readout=readout)
        answers, _, _ = run_stages(torch, arena, pipeline, [stage], documents,
                                   budget, paged=False)
        rows.extend(np.asarray(answers[0][document])[0].tolist()
                    for document in range(len(REVIEWS)))
    return rows


def _execution_state(loaded, answer_ids):
    """The execution state run_pass_group reads, over the loaded model."""
    from types import SimpleNamespace

    from quail.backends.quail.executor.readout import AnswerRows

    model, arena, pipeline = loaded
    return SimpleNamespace(
        torch=torch, chunk_tokens=BUDGET, gpu_timing=False,
        answer_rows=AnswerRows(torch, torch.nn.functional, model, answer_ids[0],
                               answer_ids[1]),
        loaded_model=SimpleNamespace(model=model, arena=arena, pipeline=pipeline,
                                     model_spec=_spec(), input_staging=None))


def _ask(loaded, answer_ids, questions):
    """Each question's answers for each review, all in one pass a review."""
    from dataclasses import replace

    from quail.backends.quail.executor.pass_groups import run_pass_group

    ids = _tokenizer()
    first = questions[0]
    ends = _ends(first.head, first.state_tail, [ids(r) for r in REVIEWS])
    group = run_pass_group(
        _execution_state(loaded, answer_ids),
        [replace(question, ends=ends) for question in questions],
        [list(first.head) + ids(review) for review in REVIEWS])
    assert group.positions.tolist() == list(range(len(REVIEWS)))
    return group.answers


@pytest.fixture(scope="module")
def quail_packs(packs, answer_ids, loaded):
    """Per review, Quail's yes-minus-no margin at each of the 16 slots."""
    import numpy as np

    return np.stack(_ask(loaded, answer_ids, packs[0]), -1).tolist()


@pytest.fixture(scope="module")
def quail_joins(join_passes, answer_ids, loaded):
    """Per join pass, Quail's yes-minus-no margin at each question's slot."""
    import numpy as np

    from quail.backends.quail.executor.readout import AnswerRows, AsyncSlotMargins
    from quail.backends.quail.executor.stages import run_stages

    model, arena, pipeline = loaded
    stage, states, _ = join_passes
    stage.readout = AsyncSlotMargins(torch, AnswerRows(
        torch, torch.nn.functional, model, answer_ids[0], answer_ids[1]))
    answers, _, _ = run_stages(torch, arena, pipeline, [stage], states, BUDGET,
                               paged=False)
    return [np.asarray(answers[0][index][0], dtype=float).tolist()
            for index in range(len(states))]


def _choice_questions():
    """The CHOICES as SetQuestions, and each one's lone prompt."""
    from quail.api.prompts import bind_classify_prompt
    from quail.backends.quail.executor.pass_groups import SetQuestion
    from quail.common.expressions import ColumnRef
    from quail.physical.lowering import lower_prompt, option_spellings

    spec, ids = _spec(), _tokenizer()
    questions, prompts = [], []
    for question, labels, descriptions in CHOICES:
        lowered = lower_prompt(bind_classify_prompt(
            question, (ColumnRef(alias="r", provider="reviews", column="body"),),
            labels, descriptions, turn=spec.turn, layout=spec.prompt_layout), ids)
        tail = tuple(lowered.tail_token_ids)
        cut = len(ids(lowered.prompt.tail_segments[0]))
        questions.append(SetQuestion(
            "choice", tuple(lowered.preamble_token_ids), tail[:cut], tail[cut:],
            tuple(ids("QUESTION:\n")),
            spellings=option_spellings(lowered.prompt.letters, ids)))
        prompts.append(lowered.prompt)
    return questions, prompts


def _read_questions():
    """The READS as SetQuestions."""
    from quail.api.prompts import bind_extract_prompt
    from quail.backends.quail.executor.pass_groups import SetQuestion
    from quail.common.expressions import ColumnRef
    from quail.common.prompts import PROMPT_LAYOUTS
    from quail.physical.lowering import lower_prompt, read_slot_offsets

    spec, ids = _spec(), _tokenizer()
    layout = PROMPT_LAYOUTS[spec.prompt_layout]
    questions = []
    for question in READS:
        lowered = lower_prompt(bind_extract_prompt(
            (ColumnRef(alias="r", provider="reviews", column="body"),),
            question, spec.turn, spec.prompt_layout), ids)
        segments = lowered.prompt.tail_segments
        tail, cut = tuple(lowered.tail_token_ids), len(ids(segments[0]))
        questions.append(SetQuestion(
            "read", tuple(lowered.preamble_token_ids), tail[:cut], tail[cut:],
            tuple(ids("QUESTION:\n")),
            slots=read_slot_offsets(segments[-1], layout.read_slots, ids)))
    return questions


def _read_answers(spans):
    """Per review, a read's text and calibrated P(found) from its spans."""
    from quail.backends.quail.executor.operators.set_extract import (
        found_reads,
        read_texts,
    )

    spec, ids = _spec(), _tokenizer()
    found, keep = found_reads(spans, spec.read_found)
    head = len(_read_questions()[0].head)
    texts, _ = read_texts(spans, keep, [[0] * head + ids(review) for review in REVIEWS],
                          REVIEWS, head, _encoder().token_bytes())
    return list(zip(texts, found.tolist()))


@pytest.fixture(scope="module")
def quail_choices(loaded, answer_ids):
    """Per review and classification, Quail's option probabilities, one pass."""
    answers = _ask(loaded, answer_ids, _choice_questions()[0])
    return [[answer[review].tolist() for answer in answers]
            for review in range(len(REVIEWS))]


@pytest.fixture(scope="module")
def quail_reads(loaded, answer_ids):
    """Per review and read, Quail's text and calibrated P(found), one pass."""
    every = [_read_answers(spans)
             for spans in _ask(loaded, answer_ids, _read_questions())]
    return [[reads[review] for reads in every] for review in range(len(REVIEWS))]


@pytest.fixture(scope="module")
def quail_mixed(loaded, answer_ids):
    """Per review, one pass of a yes/no, a choice, and a read, as Quail asks it."""
    margins, probabilities, spans = _ask(loaded, answer_ids, [
        _yes_no_questions([QUESTIONS[0]])[0], _choice_questions()[0][0],
        _read_questions()[0]])
    return list(zip(margins.tolist(), probabilities.tolist(), _read_answers(spans)))


def _margin(logits, count):
    """The best yes spelling's logit minus the best no spelling's."""
    return max(logits[:count]) - max(logits[count:])


def test_quail_writes_the_prompts_d1_renders(prompts, reference):
    texts = [text for _, _, items in prompts for text, _ in items]
    assert texts == [row["text"] for row in reference["prompts"]]
    first = _choice_questions()[1][0]
    for review, theirs in zip(REVIEWS, reference["choices"]):
        assert first.preamble + review + first.tail[len("{0}"):] == theirs["text"]


def test_one_question_answers_as_d1s_own_model(answer_ids, reference, quail):
    import math

    count = len(answer_ids[0])
    rows = [(_margin(row["logits"], count), _margin(ours, count), row["p_yes"])
            for row, ours in zip(reference["prompts"], quail)]
    print("yes minus no, d1 code vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b, _ in rows])
    assert all((a > 0) == (b > 0) for a, b, _ in rows)
    assert max(abs(a - b) for a, b, _ in rows) < MARGIN_TOLERANCE
    assert max(abs(p - 1 / (1 + math.exp(-b)))
               for _, b, p in rows) < PROBABILITY_TOLERANCE


def test_a_packed_pass_answers_as_d1_packing_the_same_questions(
        answer_ids, packs, reference, quail_packs):
    import math

    count = len(answer_ids[0])
    theirs = reference["packs"][:len(packs[1])]
    assert [pack["ids"] for pack in packs[1]] == [t["d1_ids"] for t in theirs]
    rows = [(_margin(logits, count), ours, p_yes)
            for pack, margins in zip(theirs, quail_packs)
            for logits, ours, p_yes in zip(pack["logits"], margins,
                                           pack["p_yes"])]
    print("packed yes minus no, d1 code vs Quail (first pass):",
          [(round(a, 2), round(b, 2)) for a, b, _ in rows[:16]])
    assert all((a > 0) == (b > 0) for a, b, _ in rows)
    assert max(abs(a - b) for a, b, _ in rows) < MARGIN_TOLERANCE
    assert max(abs(p - 1 / (1 + math.exp(-b)))
               for _, b, p in rows) < PROBABILITY_TOLERANCE


def test_a_join_answers_as_d1_packing_each_reviews_partners(
        answer_ids, packs, join_passes, reference, quail_joins):
    import math

    stage, _, items = join_passes
    assert [len(shape) for shape in stage.questions] == [16, 2] * len(REVIEWS)
    count = len(answer_ids[0])
    theirs = reference["packs"][len(packs[1]):]
    assert [item["ids"] for item in items] == [t["d1_ids"] for t in theirs]
    rows = [(_margin(logits, count), ours, p_yes)
            for pack, margins in zip(theirs, quail_joins)
            for logits, ours, p_yes in zip(pack["logits"], margins,
                                           pack["p_yes"])]
    print("join yes minus no, d1 code vs Quail (first review):",
          [(round(a, 2), round(b, 2)) for a, b, _ in rows[:18]])
    assert all((a > 0) == (b > 0) for a, b, _ in rows if abs(a) > 0.1)
    assert max(abs(a - b) for a, b, _ in rows) < MARGIN_TOLERANCE
    assert max(abs(p - 1 / (1 + math.exp(-b)))
               for _, b, p in rows) < PROBABILITY_TOLERANCE


def test_the_fused_kernels_match_a_plain_pytorch_forward(packs, answer_ids,
                                                         loaded, quail_packs):
    """Quail's packed passes against tests/gpu/lfm3_reference.py, two reviews."""
    import sys
    from pathlib import Path

    from quail.backends.quail.executor.model_loading import full_output_head

    sys.path.insert(0, str(Path(__file__).parent))
    from lfm3_reference import reference_hidden

    model, _, _ = loaded
    head = full_output_head(model).float()
    yes, no = list(answer_ids[0]), list(answer_ids[1])
    pairs = []
    for item, ours in zip(packs[1][:2], quail_packs[:2]):
        hidden = reference_hidden(model, _spec(), item["ids"], item["groups"])
        logits = hidden[item["slots"]].float() @ head.T
        theirs = (logits[:, yes].amax(-1) - logits[:, no].amax(-1)).tolist()
        pairs.extend(zip(theirs, ours))
    print("packed yes minus no, plain PyTorch vs Quail's kernels:",
          [(round(a, 2), round(b, 2)) for a, b in pairs[:16]])
    assert max(abs(a - b) for a, b in pairs) < MARGIN_TOLERANCE


def test_packed_choices_match_d1s_option_probabilities(reference, quail_choices):
    pairs = [(a, b) for theirs, ours in zip(reference["choices"], quail_choices)
             for question_a, question_b in zip(theirs["probabilities"], ours)
             for a, b in zip(question_a, question_b)]
    print("option probabilities, Quail vs d1 code (first review):",
          [[round(p, 3) for p in question] for question in quail_choices[0]],
          [[round(p, 3) for p in question]
           for question in reference["choices"][0]["probabilities"]])
    assert max(abs(a - b) for a, b in pairs) < PROBABILITY_TOLERANCE


def test_packed_reads_match_d1s_text_and_found(reference, quail_reads):
    from quail.backends.quail.executor.operators.set_extract import calibrated

    knots = _spec().read_found
    pairs = []
    for theirs, ours in zip(reference["reads"], quail_reads):
        for read, (text, found) in zip(theirs, ours):
            present = calibrated(knots, read["found"])
            pairs.append(((read["text"] if present >= 0.5 else None, present),
                          (text, found)))
    print("reads, d1 code vs Quail:", pairs)
    # a found near one half may land on either side
    assert all(a[0] == b[0] for a, b in pairs if abs(a[1] - 0.5) > 0.05)
    assert max(abs(a[1] - b[1]) for a, b in pairs) < PROBABILITY_TOLERANCE


def test_one_mixed_pass_matches_d1_asking_the_same_three_questions(reference,
                                                                   quail_mixed):
    import math

    from quail.backends.quail.executor.operators.set_extract import calibrated

    knots = _spec().read_found
    for theirs, (margin, choice, (text, found)) in zip(reference["mixed"],
                                                      quail_mixed):
        print("mixed pass, d1 code vs Quail:", theirs,
              (1 / (1 + math.exp(-margin)), choice, text, found))
        assert abs(theirs["p_yes"] - 1 / (1 + math.exp(-margin))) \
            < PROBABILITY_TOLERANCE
        assert max(abs(a - b) for a, b in zip(theirs["choice"], choice)) \
            < PROBABILITY_TOLERANCE
        present = calibrated(knots, theirs["read"]["found"])
        assert abs(present - found) < PROBABILITY_TOLERANCE
        if abs(present - 0.5) > 0.05:
            assert (theirs["read"]["text"] if present >= 0.5 else None) == text


if __name__ == "__main__":
    reference_main(sys.argv[1], sys.argv[2])
