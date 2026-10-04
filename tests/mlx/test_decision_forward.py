"""Quail's MLX forward pass of Decision-2.0-Kai-0.6B against mlx-lm's.

The shared stage scheduler runs two questions over each review: the
first packs the review with its question, the second reads the
review's retained KV. mlx-lm runs every whole prompt on the same
weights. Both run bf16, so each differs from mlx-lm in float32 by
rounding; the test checks that Quail's option scores are as close to
float32 as mlx-lm's bf16 scores are, and that the answers agree.

Runs on Apple silicon when the checkpoint is in the Hugging Face cache;
skipped otherwise, so it never downloads the weights.
"""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")
pytest.importorskip("mlx_lm")

from huggingface_hub import try_to_load_from_cache  # noqa: E402
from mlx_lm.models.qwen3 import Model, ModelArgs  # noqa: E402

from quail.backends.quail.executor.arena import KVArena  # noqa: E402
from quail.backends.quail.executor.mlx_device.implementation import (  # noqa: E402
    MlxImplementation,
)
from quail.backends.quail.executor.mlx_device.loader import (  # noqa: E402
    load_decision_backbone,
    load_decision_head,
)
from quail.backends.quail.executor.models import build_pipeline  # noqa: E402
from quail.backends.quail.executor.stages import Stage, run_stages  # noqa: E402
from quail.backends.quail.worker import decision_offsets  # noqa: E402
from quail.logical import bind_prompt, render_filter_prompt_ids  # noqa: E402
from quail.specs import DECISION_2_KAI_0_6B_BF16 as SPEC  # noqa: E402

WEIGHTS = try_to_load_from_cache(
    SPEC.hf_name, "backbone/model.safetensors", revision=SPEC.revision)
pytestmark = pytest.mark.skipif(
    not isinstance(WEIGHTS, str),
    reason=f"{SPEC.hf_name} is not in the Hugging Face cache")

QUESTIONS = ["Does the review in {0} praise the acting?",
             "Is the review in {0} positive overall?"]
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
    "The director clearly loves the genre, but the film never finds "
    "its footing. Uneven acting and a rushed third act sink it.",
    "Loud, long, and lifeless. The stunts are impressive; the "
    "performances are not. Skip it unless explosions are enough.",
    "An honest, small-scale drama with two terrific central "
    "performances and a script that trusts its audience.",
]
# option score differences below this are ties in bf16
TIE = 0.25


def reference_scores(path, dtype, head, offsets, prompts):
    """Option scores of whole prompts from mlx-lm's Qwen3 on the same weights."""
    backbone = Path(path) / "backbone"
    config = json.loads((backbone / "config.json").read_text())
    config["rope_theta"] = float(config["rope_parameters"]["rope_theta"])
    model = Model(ModelArgs.from_dict(config))
    tensors = mx.load(str(backbone / "model.safetensors"))
    model.load_weights([("model." + name, array.astype(dtype))
                        for name, array in tensors.items()])
    mx.eval(model.parameters())
    scores = []
    for tokens in prompts:
        hidden = model.model(mx.array([tokens]))[0].astype(mx.float32)
        last = hidden.shape[0] - 1
        rows = hidden[mx.array([last - offset for offset in offsets])]
        scores.append(np.array(head.scores(rows[None, :-1], rows[None, -1])[0]))
    return np.stack(scores)


def test_decision_scores_match_mlx_lm():
    from gigatoken import Tokenizer

    path = str(Path(WEIGHTS).parents[1])
    tokenizer = Tokenizer(path).as_hf()

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompts = [bind_prompt(question, ("body",), ids, turn=SPEC.turn,
                           layout=SPEC.prompt_layout) for question in QUESTIONS]
    docs = [list(prompts[0].preamble_token_ids) + ids(review) for review in REVIEWS]
    tails = [list(prompt.tail_token_ids) for prompt in prompts]
    assert docs[0] + tails[1] == render_filter_prompt_ids(
        prompts[1], ids(REVIEWS[0]), ids)
    offsets = decision_offsets(SPEC, path)

    weights = load_decision_backbone(path, mx.bfloat16)
    head = load_decision_head(path)
    assert abs(weights.nbytes / SPEC.w_mem_bytes - 1) < 0.02
    implementation = MlxImplementation()
    arena = KVArena(n_layers=SPEC.layers, n_pages=256, page_tokens=16,
                    n_kv=SPEC.n_kv, d_head=SPEC.d_head,
                    pools=implementation.kv_pools())
    readout = implementation.decision_choices(head, offsets)
    stages = [Stage(suffixes=[tail], readout=readout, single=True,
                    decide=lambda a, row: True) for tail in tails]
    answers, _, tokens = run_stages(
        implementation, arena,
        build_pipeline(SPEC, weights, arena, implementation="mlx"), stages,
        docs, 2048, attention_mode="unified", default_attention="unified")
    # the second question reads each review's retained KV
    assert tokens == sum(map(len, docs)) + len(docs) * sum(map(len, tails))
    got = np.stack([answers[stage][d][0] for d in range(len(docs))
                    for stage in range(len(tails))])

    whole = [doc + tail for doc in docs for tail in tails]
    exact = reference_scores(path, mx.float32, head, offsets, whole)
    half = reference_scores(path, mx.bfloat16, head, offsets, whole)
    ours = np.abs(got - exact).max()
    theirs = np.abs(half - exact).max()
    assert ours < 2 * theirs + 0.02, (ours, theirs)
    margin = exact[:, 1] - exact[:, 0]
    decided = np.abs(margin) > TIE
    assert decided.sum() >= len(whole) // 2
    assert ((got[:, 1] > got[:, 0]) == (margin > 0))[decided].all()
