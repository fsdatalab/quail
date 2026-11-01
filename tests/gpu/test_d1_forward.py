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
    with open(out_path, "w") as f:
        json.dump(rows, f)


@pytest.fixture(scope="module")
def prompts():
    return _prompts()


@pytest.fixture(scope="module")
def answer_ids():
    from quail.planner.prompt_lowering import answer_ids as ids

    true_ids, false_ids = ids(_spec(), _tokenizer())
    return true_ids, false_ids


@pytest.fixture(scope="module")
def reference(prompts, answer_ids, tmp_path_factory):
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
                 for review, (_, ids) in zip(REVIEWS, items)])))
    out = folder / "reference.json"
    environment = dict(os.environ, PYTHONPATH=REFERENCE_PACKAGES)
    subprocess.run([sys.executable, __file__, str(request), str(out)],
                   check=True, env=environment)
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def quail(prompts, answer_ids, reference):
    """Per prompt, Quail's logits of the yes and no spellings.

    Built after the reference so the two model copies never share the GPU.
    """
    import numpy as np

    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.model_loading import (
        full_output_head,
        load_model,
    )
    from quail.backends.quail.executor.models import build_pipeline
    from quail.backends.quail.executor.readout import AsyncLabelLogprobs
    from quail.backends.quail.executor.stages import Stage, run_stages
    from quail.cost import budgets
    from quail.specs import H100_SXM

    spec, device, budget = _spec(), H100_SXM, 8192
    targets = list(answer_ids[0]) + list(answer_ids[1])
    model = load_model(spec.hf_name, revision=spec.revision,
                       answer_token_ids=targets)
    full, sliding = budgets.arena_pages(spec, device, budget)
    arena = KVArena(num_layers=spec.layers, num_pages=full,
                    page_tokens=device.page_tokens,
                    num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                    dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    num_sliding_pages=sliding)
    pipeline = build_pipeline(spec, model, arena)
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


def _margin(logits, count):
    """The best yes spelling's logit minus the best no spelling's."""
    return max(logits[:count]) - max(logits[count:])


def test_quail_writes_the_prompt_d1_renders(prompts, reference):
    texts = [text for _, _, items in prompts for text, _ in items]
    assert texts == [row["text"] for row in reference]
    ids = [quail_ids for _, _, items in prompts for _, quail_ids in items]
    differ = sum(ours != row["ids"] for ours, row in zip(ids, reference))
    print(f"prompts whose segment-wise ids differ from d1's whole-text "
          f"ids: {differ} of {len(ids)}")


def test_answers_agree_with_d1s_own_model(answer_ids, reference, quail):
    count = len(answer_ids[0])
    margins = [(_margin(row["logits"], count), _margin(ours, count))
               for row, ours in zip(reference, quail)]
    print("yes minus no logit at the slot, d1 code vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b in margins])
    assert all((a > 0) == (b > 0) for a, b in margins)
    assert max(abs(a - b) for a, b in margins) < MARGIN_TOLERANCE


def test_quails_p_yes_matches_d1_reading_its_own_tokens(answer_ids,
                                                        reference, quail):
    import math

    count = len(answer_ids[0])
    ours = [1 / (1 + math.exp(-_margin(logits, count))) for logits in quail]
    theirs = [row["p_yes"] for row in reference]
    print("P(yes), d1 on its own tokens vs Quail:",
          [(round(a, 3), round(b, 3)) for a, b in zip(theirs, ours)])
    assert max(abs(a - b) for a, b in zip(theirs, ours)) < PROBABILITY_TOLERANCE


if __name__ == "__main__":
    reference_main(sys.argv[1], sys.argv[2])
