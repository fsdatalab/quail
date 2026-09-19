"""Quail's DiffusionGemma forward pass against stock vLLM's, on a GPU.

Stock vLLM runs the same checkpoint in a child process and records
the hidden state of every prompt row after the last layer. Quail
packs the same prompts into one chunk and reads the same rows. The
prompt rows are computed the same way in both: causally, with the
canvas rows after them. Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py`.
"""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

MODEL = "diffusion-gemma-26b-a4b-fp8"
QUESTION = "Does the review in {0} praise the acting?"
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
    "Not the masterpiece critics claim. Competent performances, a "
    "serviceable story, and a runtime that outstays its welcome.",
    "The child actor steals the show. Against seasoned veterans she "
    "holds the screen with ease, and the film knows it.",
    "Loud, long, and lifeless. The stunts are impressive; the "
    "performances are not. Skip it unless explosions are enough.",
    "A slow burn that rewards patience. The lead gives a career-best "
    "turn, all restraint and small gestures, and the score is lovely.",
    "Bad. Just bad. The dialogue is embarrassing and the actors seem "
    "to know it, sleepwalking through every scene.",
    "An honest, small-scale drama with two terrific central "
    "performances and a script that trusts its audience.",
]
STOCK_ROWS = "stock_rows.pt"


def _prompt_ids():
    from transformers import AutoTokenizer

    from quail.logical import bind_prompt, render_filter_prompt_ids
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompt = bind_prompt(QUESTION, ("body",), ids, turn=spec.turn)
    return [render_filter_prompt_ids(prompt, ids(review), ids)
            for review in REVIEWS]


def stock_rows_main(out_path, every_layer=False):
    """Record stock vLLM's hidden state of every prompt row.

    After the last layer, or with every_layer after each layer: a
    list per prompt of one (rows, hidden) tensor per layer.
    """
    import os

    # keep the engine in this process so the hook sees the forward
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    from quail.specs import MODELS

    prompts = _prompt_ids()
    spec = MODELS[MODEL]
    llm = LLM(model=spec.hf_name, gpu_memory_utilization=0.9,
              enforce_eager=True, enable_prefix_caching=False,
              disable_log_stats=True)
    captured = {}

    def recorder(index):
        def record(module, args, output):
            captured.setdefault(index, []).append(output[0].detach().clone())
        return record

    def attach(model):
        layers = model.model.layers
        picked = enumerate(layers) if every_layer else [(len(layers) - 1,
                                                          layers[-1])]
        return [layer.register_forward_hook(recorder(i)) for i, layer in picked]

    llm.apply_model(attach)
    rows = []
    for ids in prompts:
        captured.clear()
        # the prompt prefills in the first forward, ahead of the canvas
        llm.generate([dict(prompt_token_ids=ids)],
                     SamplingParams(max_tokens=1), use_tqdm=False)
        per_layer = [torch.cat(captured[i])[:len(ids)].float().cpu()
                     for i in sorted(captured)]
        rows.append(per_layer if every_layer else per_layer[-1])
    torch.save(rows, out_path)


@pytest.fixture(scope="module")
def prompts():
    return _prompt_ids()


@pytest.fixture(scope="module")
def stock_rows(tmp_path_factory):
    path = tmp_path_factory.mktemp("stock") / STOCK_ROWS
    subprocess.run([sys.executable, __file__, str(path)], check=True)
    return torch.load(path)


@pytest.fixture(scope="module")
def quail(prompts, stock_rows):
    """Quail's rows for the same prompts, plus the loaded model.

    Built after the stock rows so the two model copies never share
    the GPU.
    """
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.loop import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models import build_pipeline
    from quail.cost.budgets import PAGE_TOKENS
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    model = load_model(spec.hf_name, max_batched_tokens=8192,
                       moe_backend=spec.moe_backend)
    arena = KVArena(n_layers=spec.layers, n_pages=2048,
                    page_tokens=PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes,
                    sliding_layers=spec.sliding_layer_set,
                    sliding_window=spec.sliding_window,
                    n_sliding_pages=2048)
    pipeline = build_pipeline(spec, model, arena)
    groups = []
    for index, ids in enumerate(prompts):
        key = ("review", index)
        # the last eight tokens go in as the suffix, the rest as the
        # kept document prefix
        split = len(ids) - 8
        arena.activate(key, len(ids), capacity_tokens=len(ids) + 64,
                       base_tokens=split)
        groups.append(dict(key=key, prefix=ids[:split], f=split,
                           suffixes=[ids[split:]]))
    chunk = pack_chunk(torch, arena, groups, attention_mode="unified",
                       canvas=pipeline.canvas_ids,
                       answer_row=pipeline.canvas_answer_row)
    with torch.inference_mode():
        hidden = pipeline.backbone_rows(chunk)
    rows, offset = [], 0
    for ids in prompts:
        rows.append(hidden[offset:offset + len(ids)].float().cpu())
        offset += len(ids) + len(pipeline.canvas_ids)
    return rows, model


def test_prompt_rows_match_stock_vllm(stock_rows, quail):
    rows, _ = quail
    errors = []
    for stock, ours in zip(stock_rows, rows):
        assert ours.shape == stock.shape
        errors.append((ours - stock).norm(dim=-1)
                      / stock.norm(dim=-1).clamp_min(1e-6))
    rel = torch.cat(errors)
    stats = dict(median=rel.median().item(),
                 p99=rel.quantile(0.99).item(), max=rel.max().item())
    print("relative L2 error per prompt row:", json.dumps(stats))
    # fp8 activations are quantized per kernel on both sides; the
    # experts and the attention run different kernels
    assert stats["median"] < 0.05
    assert stats["p99"] < 0.2


def test_answers_agree_with_stock_vllm(stock_rows, quail):
    from transformers import AutoTokenizer

    from quail.logical import true_false_ids
    from quail.specs import MODELS

    rows, model = quail
    spec = MODELS[MODEL]
    true_ids, false_ids = true_false_ids(
        AutoTokenizer.from_pretrained(spec.hf_name))
    columns = list(model.quail_answer_token_ids)
    weights = model.quail_answer_weights.float().cpu()
    norm = model.model.norm
    weight = norm.weight.float().cpu()

    def margin(hidden):
        last = hidden[-1]
        normed = last * torch.rsqrt(last.pow(2).mean() + norm.variance_epsilon)
        logits = (normed * weight) @ weights.T
        true = max(logits[columns.index(i)] for i in true_ids if i in columns)
        false = max(logits[columns.index(i)] for i in false_ids if i in columns)
        return (true - false).item()

    margins = [(margin(stock), margin(ours))
               for stock, ours in zip(stock_rows, rows)]
    print("TRUE minus FALSE logit at the last prompt row, stock vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b in margins])
    agree = sum((a > 0) == (b > 0) for a, b in margins)
    assert agree == len(margins)


if __name__ == "__main__":
    stock_rows_main(sys.argv[1], every_layer=len(sys.argv) > 2)
