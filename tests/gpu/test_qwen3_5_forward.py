"""Quail's Qwen3.8 27B forward pass against stock vLLM's, on a GPU.

Stock vLLM runs the same checkpoint in a child process and records
each layer's output for every prompt row of its prefill pass. Quail
packs the same prompts into one chunk, each as a document prefix
with the last eight tokens as its suffix, runs its stack cut after
each layer, and reads the same rows.

Both sides run vLLM's fp8 projections and norms; they differ in the
attention kernel and in the linear layers' segment bookkeeping, so
the rows agree to rounding after the first layer and the difference
compounds through the stack. The test checks that no layer adds a
jump of its own and that the TRUE/FALSE answers at the last row
agree. Skipped without a CUDA device; runs through
`uv run modal run experiments/run_gpu_tests.py --keyword qwen3_5`.
"""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA device")

MODEL = "qwen3.8-27b-fp8"
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
SUFFIX_TOKENS = 8


def _prompt_ids():
    from gigatoken import Tokenizer

    from quail.logical import bind_prompt, render_filter_prompt_ids
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    tokenizer = Tokenizer(spec.hf_name).as_hf()

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompt = bind_prompt(QUESTION, ("body",), ids, turn=spec.turn)
    return [render_filter_prompt_ids(prompt, ids(review), ids)
            for review in REVIEWS]


def stock_rows_main(out_path):
    """Record stock vLLM's output of every layer for every prompt row.

    A layer returns its feedforward output and the residual it was
    added to; their sum is the layer's output.
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
              language_model_only=True, max_model_len=4096, max_num_seqs=8,
              disable_log_stats=True)
    captured = {}

    def recorder(index):
        def record(module, args, output):
            hidden, residual = output
            captured.setdefault(index, []).append(
                (hidden + residual).detach().clone())
        return record

    def attach(model):
        layers = model.language_model.model.layers
        return [layer.register_forward_hook(recorder(i))
                for i, layer in enumerate(layers)]

    llm.apply_model(attach)
    rows = []
    for ids in prompts:
        captured.clear()
        llm.generate([dict(prompt_token_ids=ids)],
                     SamplingParams(max_tokens=1), use_tqdm=False)
        # the prompt prefills in the first forward
        rows.append([captured[i][0][:len(ids)].float().cpu()
                     for i in sorted(captured)])
    torch.save(rows, out_path)


@pytest.fixture(scope="module")
def prompts():
    return _prompt_ids()


@pytest.fixture(scope="module")
def stock_layers(tmp_path_factory):
    """Per prompt, stock vLLM's rows after each layer."""
    path = tmp_path_factory.mktemp("stock") / STOCK_ROWS
    subprocess.run([sys.executable, __file__, str(path)], check=True)
    return torch.load(path)


@pytest.fixture(scope="module")
def quail(prompts, stock_layers):
    """Quail's rows after each layer for the same prompts, plus the model.

    Built after the stock rows so the two model copies never share
    the GPU.
    """
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.chunk import pack_chunk
    from quail.backends.quail.executor.model import load_model
    from quail.backends.quail.executor.models.qwen3_5 import Qwen35Pipeline
    from quail.cost.budgets import PAGE_TOKENS
    from quail.specs import MODELS

    spec = MODELS[MODEL]
    model = load_model(spec.hf_name, max_batched_tokens=8192,
                       language_model_only=spec.language_model_only)
    arena = KVArena(n_layers=spec.layers, n_pages=2048,
                    page_tokens=PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes)
    pipeline = Qwen35Pipeline(model, arena, spec=spec)
    groups = []
    for index, ids in enumerate(prompts):
        key = ("review", index)
        split = len(ids) - SUFFIX_TOKENS
        arena.activate(key, len(ids), capacity_tokens=len(ids) + 64,
                       base_tokens=split)
        groups.append(dict(key=key, prefix=ids[:split], f=split,
                           suffixes=[ids[split:]]))
    chunk = pack_chunk(torch, arena, groups, attention_mode="unified")
    layers = list(pipeline.layers)
    per_prompt = [[] for _ in prompts]
    for depth in range(1, len(layers) + 1):
        # the stack cut after a layer returns that layer's output
        pipeline.layers = layers[:depth]
        with torch.inference_mode():
            hidden, residual = pipeline.backbone_rows(chunk)
            output = (hidden + residual).float().cpu()
        offset = 0
        for rows, ids in zip(per_prompt, prompts):
            rows.append(output[offset:offset + len(ids)])
            offset += len(ids)
    pipeline.layers = layers
    return per_prompt, model


def _relative_errors(stock_layers, quail_layers, layer):
    return torch.cat([
        (ours[layer] - stock[layer]).norm(dim=-1)
        / stock[layer].norm(dim=-1).clamp_min(1e-6)
        for stock, ours in zip(stock_layers, quail_layers)])


def test_no_layer_departs_from_stock_vllm(stock_layers, quail):
    quail_layers, _ = quail
    depth = len(stock_layers[0])
    assert all(len(rows) == depth for rows in quail_layers)
    medians = []
    for layer in range(depth):
        rel = _relative_errors(stock_layers, quail_layers, layer)
        medians.append(rel.median().item())
        print(json.dumps(dict(layer=layer, median=round(medians[-1], 4),
                              p99=round(rel.quantile(0.99).item(), 4))))
    # the first layer differs by kernel rounding alone
    assert medians[0] < 0.03
    # every later layer adds rounding of its own, never a jump: a
    # missing term or a wrong mask would double the error at once
    for before, after in zip(medians, medians[1:]):
        assert after < 2 * before + 0.01
    assert medians[-1] < 0.25


def test_answers_agree_with_stock_vllm(stock_layers, quail):
    from gigatoken import Tokenizer

    from quail.backends.quail.executor.model import text_model
    from quail.logical import true_false_ids
    from quail.specs import MODELS

    quail_layers, model = quail
    stock_rows = [rows[-1] for rows in stock_layers]
    rows = [rows[-1] for rows in quail_layers]
    spec = MODELS[MODEL]
    true_ids, false_ids = true_false_ids(
        Tokenizer(spec.hf_name).as_hf())
    columns = list(model.quail_answer_token_ids)
    weights = model.quail_answer_weights.float().cpu()
    norm = text_model(model).model.norm
    # the model's norms scale by one plus their weight
    weight = norm.weight.float().cpu() + 1.0

    def margin(hidden):
        last = hidden[-1]
        normed = last * torch.rsqrt(last.pow(2).mean() + norm.variance_epsilon)
        logits = (normed * weight) @ weights.T
        true = max(logits[columns.index(i)] for i in true_ids if i in columns)
        false = max(logits[columns.index(i)] for i in false_ids if i in columns)
        return (true - false).item()

    margins = [(margin(stock), margin(ours))
               for stock, ours in zip(stock_rows, rows)]
    print("TRUE minus FALSE logit at the last row, stock vs Quail:",
          [(round(a, 2), round(b, 2)) for a, b in margins])
    agree = sum((a > 0) == (b > 0) for a, b in margins)
    assert agree == len(margins)


if __name__ == "__main__":
    stock_rows_main(sys.argv[1])


# ---- saved state: a sequence split at a save point equals the whole one


@pytest.fixture(scope="module")
def state_run(prompts, quail):
    """A pipeline over an arena with state slots, sharing the loaded model."""
    from quail.backends.quail.executor.arena import KVArena
    from quail.backends.quail.executor.models.qwen3_5 import Qwen35Pipeline
    from quail.cost.budgets import PAGE_TOKENS
    from quail.specs import MODELS

    _, model = quail
    spec = MODELS[MODEL]
    arena = KVArena(n_layers=spec.layers, n_pages=1024,
                    page_tokens=PAGE_TOKENS, n_kv=spec.n_kv,
                    d_head=spec.d_head, dtype=torch.bfloat16,
                    layer_kv=spec.kv_shapes, state_layers=spec.linear_layer_set,
                    n_state_slots=8, state_shape=spec.state_shape,
                    conv_shape=spec.conv_shape)
    return arena, Qwen35Pipeline(model, arena, spec=spec)


def _layer_output(pipeline, chunk):
    with torch.inference_mode():
        hidden, residual = pipeline.backbone_rows(chunk)
    return (hidden + residual).float().cpu()


def _relative(ours, reference):
    return ((ours - reference).norm(dim=-1)
            / reference.norm(dim=-1).clamp_min(1e-6))


def test_state_saved_at_the_document_end_restores_the_sequence(prompts, state_run):
    from quail.backends.quail.executor.chunk import pack_chunk

    arena, pipeline = state_run
    ids = prompts[0]
    split = len(ids) - SUFFIX_TOKENS
    whole_key = ("whole", 0)
    arena.activate(whole_key, len(ids), capacity_tokens=len(ids) + 16,
                   base_tokens=split)
    whole = _layer_output(pipeline, pack_chunk(
        torch, arena, [dict(key=whole_key, prefix=ids[:split], f=split,
                            suffixes=[ids[split:]])],
        attention_mode="unified"))
    arena.free_key(whole_key)

    key = ("split", 0)
    arena.activate(key, split, capacity_tokens=len(ids) + 16, base_tokens=split,
                   slots=1)
    first = pack_chunk(torch, arena, [dict(key=key, prefix=ids[:split], f=split,
                                           suffixes=[], save_at=(split,))],
                       attention_mode="unified")
    assert [seg.save for seg in first.meta["state"]["segments"]] != [0]
    prefix_rows = _layer_output(pipeline, first)
    second = pack_chunk(torch, arena, [dict(key=key, prefix=None, f=split,
                                            suffixes=[ids[split:]])],
                        attention_mode="unified")
    suffix_rows = _layer_output(pipeline, second)
    arena.free_key(key)
    prefix_error = _relative(prefix_rows, whole[:split])
    suffix_error = _relative(suffix_rows, whole[split:])
    print(json.dumps(dict(prefix_median=prefix_error.median().item(),
                          suffix_median=suffix_error.median().item(),
                          suffix_max=suffix_error.max().item())))
    # the prefix rows are the same computation; the suffix rows start
    # from the saved state instead of running behind the prefix, and
    # differ by the kernels' chunk boundaries alone
    assert prefix_error.max().item() < 1e-3
    assert suffix_error.median().item() < 0.02
    assert suffix_error.max().item() < 0.1


def test_partners_read_the_kept_state_and_leave_it_unchanged(prompts, state_run):
    from quail.backends.quail.executor.chunk import pack_chunk

    arena, pipeline = state_run
    ids = prompts[1]
    anchor, frame = ids[:len(ids) - SUFFIX_TOKENS], ids[len(ids) - SUFFIX_TOKENS:
                                                         len(ids) - 4]
    partners = [ids[len(ids) - 4:], prompts[2][-4:], prompts[3][-4:]]
    f = len(anchor)
    separate = []
    for index, partner in enumerate(partners):
        key = ("one", index)
        arena.activate(key, f + len(frame) + len(partner),
                       capacity_tokens=f + 32, base_tokens=f)
        rows = _layer_output(pipeline, pack_chunk(
            torch, arena, [dict(key=key, prefix=anchor + frame, f=f + len(frame),
                                suffixes=[partner])],
            attention_mode="unified"))
        separate.append(rows[f + len(frame):])
        arena.free_key(key)

    key = ("anchor", 0)
    arena.activate(key, f, capacity_tokens=f + 32, base_tokens=f, slots=1)
    chunk = pack_chunk(torch, arena, [
        dict(key=key, prefix=anchor, f=f, suffixes=[frame],
             write_suffix_tokens=len(frame), save_at=(f + len(frame),)),
        dict(key=key, prefix=None, f=f + len(frame), suffixes=partners),
    ], attention_mode="unified")
    plan = chunk.meta["state"]
    assert [w["n"] for w in plan["waves"]] == [1, 3]
    together = _layer_output(pipeline, chunk)
    kept = arena.state_slot_at(key, f + len(frame))
    pools = [pipeline.arena.state_pools(layer) for layer in sorted(arena.state_layers)]
    before = [(s[kept].clone(), c[kept].clone()) for s, c in pools]
    # a second round of partners starts from the kept state again
    again = pack_chunk(torch, arena, [
        dict(key=key, prefix=None, f=f + len(frame), suffixes=partners)],
        attention_mode="unified")
    repeated = _layer_output(pipeline, again)
    for key_ in chunk.temporary_keys + again.temporary_keys:
        arena.free_key(key_)
    after = [(s[kept], c[kept]) for s, c in pools]
    assert all(torch.equal(b[0], a[0]) and torch.equal(b[1], a[1])
               for b, a in zip(before, after))
    arena.free_key(key)
    offset = f + len(frame)
    errors = []
    for index, partner in enumerate(partners):
        rows = together[offset:offset + len(partner)]
        errors.append(_relative(rows, separate[index]).median().item())
        assert torch.equal(rows, repeated[sum(map(len, partners[:index])):
                                          sum(map(len, partners[:index + 1]))])
        offset += len(partner)
    print(json.dumps(dict(partner_medians=errors)))
    assert max(errors) < 0.02
