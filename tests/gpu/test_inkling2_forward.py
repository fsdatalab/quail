"""Quail's Inkling 2 Mini forward pass against the transformers reference.

The reference is transformers' Inkling2Loop model at the commit
experiments/run_gpu_tests.py installs, run in a child process with that
version on its path. It dequantizes the experts to bf16 and keeps bf16
activations; Quail runs the experts in NVFP4 and MXFP8 with quantized
activations, as the checkpoint declares, and different attention and
convolution kernels, so the logits differ slightly.

Quail packs each review as a document and asks one question in a first
chunk, then asks a second question in a later chunk from the document's
saved KV and convolution checkpoint. The later chunk replays CUDA graphs
of the work between attention calls. The reference runs each whole
sequence. The test checks that the next token and the TRUE-against-FALSE
margin agree, and that an answer step replayed as one graph, attention
included, matches the same step replayed piecewise with eager attention.

Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword inkling2_forward
--device b200`.
"""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

MODEL = "inkling-2-mini"
# transformers at the Inkling 2 Mini commit, installed by
# experiments/run_gpu_tests.py's image
REFERENCE_PACKAGES = "/opt/inkling-reference"
QUESTIONS = ["Does the review praise the acting? Answer TRUE or FALSE.",
             "Does the reviewer recommend the film? Answer TRUE or FALSE."]
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
# the largest mean difference between the two sides' TRUE-against-FALSE
# margins this test accepts, in logits, over the sequences whose answer the
# reference is not sure of (a margin under CERTAIN_MARGIN logits). NVFP4
# activations against bf16 ones move such a margin by about 1 logit on
# average; one near a tie can move by 3 or more as NVFP4 roundings flip.
MARGIN_TOLERANCE = 2.0
CERTAIN_MARGIN = 8.0
# the KL divergence of Quail's next-token distribution from the
# reference's, in nats: the median over sequences, and the most any one
# sequence near a tie may reach
MEDIAN_KL_TOLERANCE = 0.05
KL_TOLERANCE = 1.5


def _spec():
    from quail.specs import MODELS

    return MODELS[MODEL]


def _path():
    from quail.specs.files import resolve_model_path

    spec = _spec()
    return resolve_model_path(spec.hf_name, spec.revision)


def _sequences():
    """Per review, the document ids and each question's suffix ids."""
    from quail.common.tokens import Tokenizer

    spec = _spec()
    ids = Tokenizer(_path(), spec.turn_suffix)
    out = []
    for review in REVIEWS:
        document = ids(spec.turn_prefix + "Review: " + review + "\n\n")
        suffixes = [ids(question + spec.turn_suffix) for question in QUESTIONS]
        out.append((document, suffixes))
    return out


def _answer_ids():
    from quail.common.tokens import Tokenizer

    ids = Tokenizer(_path(), _spec().turn_suffix)
    return ids("TRUE")[0], ids("FALSE")[0]


def reference_main(out_path):
    """Write the reference's last-row logits of every sequence."""
    sys.path.insert(0, REFERENCE_PACKAGES)
    from transformers import Inkling2LoopForConditionalGeneration

    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        _path(), dtype=torch.bfloat16, device_map="cuda").eval()
    logits = []
    with torch.inference_mode():
        for document, suffixes in _sequences():
            for suffix in suffixes:
                ids = torch.tensor([document + suffix], device="cuda")
                logits.append(model(input_ids=ids).logits[0, -1].float().cpu())
    torch.save(torch.stack(logits), out_path)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    path = tmp_path_factory.mktemp("reference") / "logits.pt"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, __file__, str(path)], check=True, env=env)
    return torch.load(path)


@pytest.fixture(scope="module")
def quail(reference):
    """Quail's last-row logits of the same sequences, in the same order.

    Built after the reference so the two model copies never share the GPU.
    """
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.chunk import pack_chunk
    from quail.backends.quail.executor.cuda_graphs import PiecewiseGraphs
    from quail.backends.quail.executor.model_loading import (
        cap_logits,
        full_output_head,
        load_model,
        logit_softcap,
    )
    from quail.backends.quail.executor.models import build_pipeline
    from quail.specs import B200

    spec = _spec()
    model = load_model(spec.hf_name, spec.revision,
                       answer_token_ids=list(_answer_ids()))
    arena = KVArena(num_layers=spec.layers, num_pages=1024,
                    page_tokens=B200.page_tokens,
                    num_kv_heads=spec.num_kv_heads, d_head=spec.d_head,
                    dtype=torch.bfloat16, layer_kv=spec.kv_shapes,
                    num_state_slots=32,
                    gdn_state_shape=(spec.layers, 0, 0, 0),
                    conv_state_shape=(spec.layers, spec.conv_width - 1,
                                      spec.conv_state_channels))
    pipeline = build_pipeline(spec, model, arena)
    graphs = PiecewiseGraphs.for_specs(pipeline, spec, B200)
    with torch.inference_mode():
        graphs.prepare()
    head, cap = full_output_head(model), logit_softcap(model)
    sequences = _sequences()
    logits = [[None, None] for _ in sequences]
    for question in range(len(QUESTIONS)):
        groups = []
        for index, (document, suffixes) in enumerate(sequences):
            key = ("review", index)
            suffix = suffixes[question]
            arena.activate(key, len(document),
                           capacity_tokens=len(document) + len(suffix) + 8)
            # a resident document's group has no prefix
            prefix = document if question == 0 else None
            groups.append(dict(key=key, prefix=prefix, f=len(document),
                               suffixes=[suffix]))
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified")
        with torch.inference_mode():
            final = graphs.forward(chunk)
            rows = cap_logits(torch.nn.functional.linear(final, head).float(),
                              cap).cpu()
        for index in range(len(sequences)):
            logits[index][question] = rows[index]
        for key in chunk.temporary_keys:
            arena.free_key(key)
    # the documents' chunk exceeds the largest bucket and runs eagerly;
    # the resident documents' chunk replays
    assert graphs.replays == 1
    # one answer token per resident document: a whole-step graph, then the
    # same step replayed piecewise in the same bucket, attention eager
    def piecewise(chunk):
        pipeline.whole_step = lambda chunk, num_tokens: False
        try:
            return graphs.forward(chunk)
        finally:
            del pipeline.whole_step

    steps = []
    for run in (graphs.forward, piecewise):
        groups = []
        for index, (document, suffixes) in enumerate(sequences):
            key = ("review", index)
            arena.activate(key, len(document), capacity_tokens=len(document) + 8)
            groups.append(dict(key=key, prefix=None, f=len(document),
                               suffixes=[suffixes[0][:1]]))
        chunk = pack_chunk(torch, arena, groups, attention_mode="unified")
        with torch.inference_mode():
            steps.append(run(chunk).float())
        for key in chunk.temporary_keys:
            arena.free_key(key)
    assert any(captured.whole is not None for captured in graphs.graphs)
    return (torch.stack([row for pair in logits for row in pair]),
            (steps[0] - steps[1]).abs().max().item())


def test_next_tokens_and_answer_margins_agree_with_transformers(reference,
                                                                quail):
    quail = quail[0]
    true_id, false_id = _answer_ids()
    margins = [((r[true_id] - r[false_id]).item(),
                (q[true_id] - q[false_id]).item())
               for r, q in zip(reference, quail)]
    print("TRUE minus FALSE logit, reference vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b in margins])
    top = [(int(r.argmax()), int(q.argmax())) for r, q in zip(reference,
                                                               quail)]
    print("next token, reference vs Quail:", top)
    log_r = reference.log_softmax(-1)
    log_q = quail.log_softmax(-1)
    kl = (log_r.exp() * (log_r - log_q)).sum(-1)
    print("KL(reference || Quail) per sequence:",
          [round(v, 4) for v in kl.tolist()])
    assert all(a == b for a, b in top)
    assert all((a > 0) == (b > 0) for a, b in margins)
    unsure = [abs(a - b) for a, b in margins if abs(a) < CERTAIN_MARGIN]
    assert sum(unsure) / len(unsure) < MARGIN_TOLERANCE
    assert kl.median().item() < MEDIAN_KL_TOLERANCE
    assert kl.max().item() < KL_TOLERANCE


def test_an_answer_step_replayed_whole_matches_its_piecewise_replay(quail):
    # both pad to the same bucket, so only the attention calls differ
    assert quail[1] < 1e-2


if __name__ == "__main__":
    reference_main(sys.argv[1])
