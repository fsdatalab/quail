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
    from gigatoken import Tokenizer

    from quail.specs.files import resolve_model_path

    spec = _spec()
    hf = Tokenizer(resolve_model_path(spec.hf_name, spec.revision)).as_hf()
    return lambda text: hf(text, add_special_tokens=False)["input_ids"]


def _prompts():
    """Per question, the lowered prompt and the text and ids of each review."""
    from quail.logical import bind_prompt
    from quail.planner.prompt_lowering import (
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
    with open(out_path, "w") as f:
        json.dump(dict(prompts=rows, packs=packs), f)


@pytest.fixture(scope="module")
def prompts():
    return _prompts()


@pytest.fixture(scope="module")
def answer_ids():
    from quail.planner.prompt_lowering import answer_ids as ids

    true_ids, false_ids = ids(_spec(), _tokenizer())
    return true_ids, false_ids


def _packs():
    """Per review, one pass of the 16 PACKED questions as Quail lays it out.

    Returns the pass's token ids, its state length, each token's
    question (-1 in the state, then 1 to 16), and each question's slot.
    """
    from types import SimpleNamespace

    from quail.backends.quail.executor.operators.filter import (
        StateTails,
        set_pass_stage,
    )
    from quail.backends.quail.lowering import set_question_parts
    from quail.catalog.tokens import GigatokenEncoder
    from quail.logical import bind_prompt
    from quail.planner.prompt_lowering import lower_prompt
    from quail.specs.files import resolve_model_path

    spec, ids = _spec(), _tokenizer()
    nodes = []
    for question in PACKED:
        lowered = lower_prompt(bind_prompt(question, ("body",), turn=spec.turn,
                                           layout=spec.prompt_layout), ids)
        nodes.append(SimpleNamespace(
            question_token_ids=tuple(lowered.tail_token_ids),
            **set_question_parts(lowered, spec, ids)))
    stage = set_pass_stage(nodes, readout=None, decide=lambda _, row: True)
    tails = StateTails(GigatokenEncoder(resolve_model_path(spec.hf_name,
                                                           spec.revision)))
    tail = list(nodes[0].question_token_ids[:nodes[0].state_tail_tokens])
    out = []
    for review in REVIEWS:
        prefix = tails.prefix(list(lowered.preamble_token_ids) + ids(review),
                              tail)
        state = len(prefix)
        groups = [-1] * state + [
            number for number, length in enumerate(stage.questions, 1)
            for _ in range(length)]
        slots = [len(prefix) + int(index)
                 for index in stage.read_mask.nonzero()[0]]
        out.append(dict(review=review, ids=prefix + stage.suffixes[0],
                        state=state, groups=groups, slots=slots,
                        questions=[q.removesuffix(" {0}") for q in PACKED],
                        prefix=prefix))
    return stage, out


@pytest.fixture(scope="module")
def packs():
    return _packs()


@pytest.fixture(scope="module")
def reference(prompts, packs, answer_ids, tmp_path_factory):
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
               for pack in packs[1]])))
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
    from quail.cost import budgets
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


@pytest.fixture(scope="module")
def quail_packs(packs, answer_ids, loaded):
    """Per review, Quail's yes-minus-no margin at each of the 16 slots."""
    import numpy as np

    from quail.backends.quail.executor.readout import AnswerRows, AsyncSlotMargins
    from quail.backends.quail.executor.stages import run_stages

    model, arena, pipeline = loaded
    stage, items = packs
    stage.readout = AsyncSlotMargins(torch, AnswerRows(
        torch, torch.nn.functional, model, answer_ids[0], answer_ids[1]))
    answers, _, _ = run_stages(torch, arena, pipeline, [stage],
                               [item["prefix"] for item in items], BUDGET,
                               paged=False)
    return [np.asarray(answers[0][index][0], dtype=float).tolist()
            for index in range(len(items))]


def _margin(logits, count):
    """The best yes spelling's logit minus the best no spelling's."""
    return max(logits[:count]) - max(logits[count:])


def test_quail_writes_the_prompt_d1_renders(prompts, reference):
    texts = [text for _, _, items in prompts for text, _ in items]
    assert texts == [row["text"] for row in reference["prompts"]]
    ids = [quail_ids for _, _, items in prompts for _, quail_ids in items]
    differ = sum(ours != row["ids"]
                 for ours, row in zip(ids, reference["prompts"]))
    print(f"prompts whose segment-wise ids differ from d1's whole-text "
          f"ids: {differ} of {len(ids)}")


def test_answers_agree_with_d1s_own_model(answer_ids, reference, quail):
    count = len(answer_ids[0])
    margins = [(_margin(row["logits"], count), _margin(ours, count))
               for row, ours in zip(reference["prompts"], quail)]
    print("yes minus no logit at the slot, d1 code vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b in margins])
    assert all((a > 0) == (b > 0) for a, b in margins)
    assert max(abs(a - b) for a, b in margins) < MARGIN_TOLERANCE


def test_quails_p_yes_matches_d1_reading_its_own_tokens(answer_ids,
                                                        reference, quail):
    import math

    count = len(answer_ids[0])
    ours = [1 / (1 + math.exp(-_margin(logits, count))) for logits in quail]
    theirs = [row["p_yes"] for row in reference["prompts"]]
    print("P(yes), d1 on its own tokens vs Quail:",
          [(round(a, 3), round(b, 3)) for a, b in zip(theirs, ours)])
    assert max(abs(a - b) for a, b in zip(theirs, ours)) < PROBABILITY_TOLERANCE


def test_a_packed_pass_is_the_sequence_d1_packs(packs, reference):
    for pack, theirs in zip(packs[1], reference["packs"]):
        assert pack["ids"] == theirs["d1_ids"]


def test_each_packed_slot_agrees_with_d1s_own_model(answer_ids, reference,
                                                    quail_packs):
    count = len(answer_ids[0])
    pairs = [(_margin(logits, count), ours)
             for pack, margins in zip(reference["packs"], quail_packs)
             for logits, ours in zip(pack["logits"], margins)]
    print("packed yes minus no, d1 code vs Quail (first pass):",
          [(round(a, 2), round(b, 2)) for a, b in pairs[:16]])
    assert all((a > 0) == (b > 0) for a, b in pairs)
    assert max(abs(a - b) for a, b in pairs) < MARGIN_TOLERANCE


def test_packed_p_yes_matches_d1_reading_its_own_pass(reference, quail_packs):
    import math

    pairs = [(theirs, 1 / (1 + math.exp(-ours)))
             for pack, margins in zip(reference["packs"], quail_packs)
             for theirs, ours in zip(pack["p_yes"], margins)]
    print("packed P(yes), d1 own pass vs Quail (first pass):",
          [(round(a, 3), round(b, 3)) for a, b in pairs[:16]])
    assert max(abs(a - b) for a, b in pairs) < PROBABILITY_TOLERANCE


if __name__ == "__main__":
    reference_main(sys.argv[1], sys.argv[2])
