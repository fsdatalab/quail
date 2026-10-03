"""Decision-2.0-Kai-0.6B on H100: Quail's answers against the model's own code.

Quail runs one filter over 500 IMDB test reviews and one join over
24 x 24 reviews. The model's own code (decision2's DecisionModel in
fp32 under bf16 autocast) then answers the same prompts two ways:
from its own tokenization of the prompt text, and from Quail's token
ids, which split tokens where a document meets the fixed text.

Prediction: cold startup, conversion included, under 60 s. Quail
agrees with the model's own tokenization on at least 97% of answers
and with its own ids on at least 99%. The filter's query time is under
5 s for about 150k input tokens.

    uv run modal run experiments/cells/decision_agreement.py \
        2>&1 | tee /tmp/decision_agreement.log

The summary is written to
/results/decision_agreement/<run>_summary.json on the quail-results
volume.
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    quail_image = gpu_image()
except ImportError:    # the reference container has no quail package
    quail_image = None

MODEL = "decision-2.0-kai-0.6b-bf16"
HF_NAME = "vllm-sr/Decision-2.0-Kai-0.6B"
REVISION = "881bee413681d80ebeac86afcda8b4138dae516e"
IMDB = ("https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
        "plain_text/test-00000-of-00001.parquet")
FILTER_TEMPLATE = "Is this movie review positive? {0}"
JOIN_TEMPLATE = "Do {0} and {1} express the same opinion of the movie?"
N_FILTER = 500
N_JOIN = 24
SEED = 20261003

# House rule: attach to the existing app, never a fresh one.
app = modal.App("quail-milestone1")
hf_cache = modal.Volume.from_name("quail-hf-cache", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache",
                                      create_if_missing=True)
results = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache,
           "/root/.cache/kernels": kernel_cache,
           "/results": results}
reference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==5.17.0", "safetensors",
                 "huggingface_hub", "numpy", "pyarrow")
)


def _reviews():
    """Balanced IMDB test samples: the filter set and two join sides."""
    import io
    import urllib.request

    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(urllib.request.urlopen(IMDB).read()))
    text, label = table["text"].to_pylist(), table["label"].to_pylist()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(text)).tolist()
    pos = [i for i in order if label[i] == 1]
    neg = [i for i in order if label[i] == 0]
    half = N_FILTER // 2
    chosen = pos[:half] + neg[:half]
    left = pos[half:half + N_JOIN // 2] + neg[half:half + N_JOIN // 2]
    right = (pos[half + N_JOIN // 2:half + N_JOIN]
             + neg[half + N_JOIN // 2:half + N_JOIN])

    def rows(indices):
        return [(text[i].replace("<br />", "\n"), label[i]) for i in indices]

    return rows(chosen), rows(left), rows(right)


@app.function(image=quail_image, gpu="H100!", memory=98304, timeout=3600,
              volumes=VOLUMES)
def quail_side(run: str) -> dict:
    """Run the filter and the join in a Quail Session; keep each prompt's ids."""
    import pyarrow as pa

    import quail
    from quail.logical import (
        ColumnRef,
        answer_row_offsets,
        bind_join_prompt,
        bind_prompt,
        render_filter_prompt_ids,
        render_join_prompt_ids,
    )
    from quail.physical import AiJoin
    from quail.planner.plan import EngineConfig

    reviews, left, right = _reviews()
    session = quail.Session(EngineConfig(model=MODEL, device="h100-sxm"))
    tok = session.tokenizer
    layout = session.model.prompt_layout
    session.register("reviews", quail.DocumentProvider.from_table(pa.table({
        "id": list(range(len(reviews))), "body": [t for t, _ in reviews]}),
        id_col="id"))
    for name, side in (("lefts", left), ("rights", right)):
        session.register(name, quail.DocumentProvider.from_table(pa.table({
            "id": list(range(len(side))), "body": [t for t, _ in side]}),
            id_col="id"))

    out = {"run": run, "offsets": answer_row_offsets(layout, tok)}
    filter_sql = ("SELECT r.id FROM reviews r WHERE AI_FILTER(PROMPT("
                  f"'{FILTER_TEMPLATE}', r.body))")
    runs = []
    for _ in range(2):
        t0 = time.perf_counter()
        result = session.sql(filter_sql).run()
        runs.append({"seconds": time.perf_counter() - t0,
                     "report": result.report,
                     "ids": sorted(r[0] for r in result.to_rows())})
    out["filter_runs"] = runs
    prompt = bind_prompt(FILTER_TEMPLATE, (ColumnRef("r", "reviews", "body"),),
                         tok, layout=layout)
    out["filter_ids"] = [render_filter_prompt_ids(prompt, tok(t), tok)
                         for t, _ in reviews]

    join_sql = ("SELECT a.id, b.id FROM lefts a JOIN rights b ON AI_FILTER("
                f"PROMPT('{JOIN_TEMPLATE}', a.body, b.body))")
    query = session.sql(join_sql)
    (node,) = [n for n in query.plan().nodes if isinstance(n, AiJoin)]
    anchor = node.stages[0].anchor
    t0 = time.perf_counter()
    result = query.run()
    out["join_run"] = {"seconds": time.perf_counter() - t0,
                       "report": result.report,
                       "pairs": sorted(result.to_rows())}
    prompt = bind_join_prompt(
        JOIN_TEMPLATE, (ColumnRef("a", "lefts", "body"),
                        ColumnRef("b", "rights", "body")), tok, layout=layout)
    a = ["a", "b"].index(anchor)
    out["join_anchor"] = anchor
    out["join_ids"] = [
        render_join_prompt_ids(prompt, [tok(x), tok(y)], a, tok)
        for x, _ in left for y, _ in right]
    session.close()
    return out


@app.function(image=reference_image, gpu="H100!", memory=65536,
              timeout=3600, volumes=VOLUMES)
def reference_side(quail_out: dict) -> dict:
    """Answer the same prompts with decision2's own model; compare."""
    import sys

    import torch
    from huggingface_hub import snapshot_download

    path = snapshot_download(HF_NAME, revision=REVISION)
    sys.path.insert(0, path)
    from decision2._vendor.dev2model.decision_model import (
        DecisionModel,
        collate,
        encode,
    )

    model, tokenizer = DecisionModel.from_checkpoint(path)
    model = model.cuda().eval()
    reviews, left, right = _reviews()
    options = [{"key": "false", "description": "No"},
               {"key": "true", "description": "Yes"}]

    def row(i, state, question):
        return {"id": str(i), "state": state, "task_type": "noul",
                "instructions": question, "options": options, "label": 0,
                "family": "imdb"}

    def from_ids(i, ids):
        last = len(ids) - 1
        no, yes, _ = quail_out["offsets"]
        return {"id": str(i), "ids": ids,
                "candidate_positions": [last - no, last - yes],
                "query_position": last, "label": 0, "keys": ["false", "true"],
                "task_type": "noul", "family": "imdb", "teacher_probs": None}

    @torch.no_grad()
    def answer(items):
        probs = []
        for start in range(0, len(items), 16):
            batch = collate(items[start:start + 16], tokenizer.pad_token_id)
            batch = {k: v.cuda() if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch)
            probs += torch.softmax(logits.float(), -1)[:, 1].tolist()
        return probs

    filter_question = FILTER_TEMPLATE.replace(" {0}", "")
    ref_text = answer([encode(row(i, t, filter_question), tokenizer, 8192)
                       for i, (t, _) in enumerate(reviews)])
    ref_ids = answer([from_ids(i, ids)
                      for i, ids in enumerate(quail_out["filter_ids"])])
    anchor = quail_out["join_anchor"]
    pairs = [(i, j) for i in range(len(left)) for j in range(len(right))]

    def join_state(i, j):
        docs = {"a": left[i][0], "b": right[j][0]}
        index = {"a": 0, "b": 1}
        other = "b" if anchor == "a" else "a"
        return (f"{docs[anchor]}\n\n(The document above is DOCUMENT "
                f"{{{index[anchor]}}}.)\n\nDOCUMENT {{{index[other]}}}:\n"
                f"{docs[other]}")

    join_text = answer([encode(row(k, join_state(i, j), JOIN_TEMPLATE),
                               tokenizer, 8192)
                        for k, (i, j) in enumerate(pairs)])
    join_ids = answer([from_ids(k, ids)
                       for k, ids in enumerate(quail_out["join_ids"])])

    def compare(quail_bits, text_probs, ids_probs, labels):
        text_bits = [int(p > 0.5) for p in text_probs]
        ids_bits = [int(p > 0.5) for p in ids_probs]
        n = len(labels)

        def share(x, y):
            return sum(int(a == b) for a, b in zip(x, y)) / n

        flipped = [abs(p - 0.5) for q, p in zip(quail_bits, text_probs)
                   if q != int(p > 0.5)]
        return {
            "n": n,
            "quail_vs_reference_text": share(quail_bits, text_bits),
            "quail_vs_reference_quail_ids": share(quail_bits, ids_bits),
            "reference_text_vs_quail_ids": share(text_bits, ids_bits),
            "accuracy_quail": share(quail_bits, labels),
            "accuracy_reference_text": share(text_bits, labels),
            "disagreement_margins": sorted(flipped),
            "max_prob_gap_text_vs_ids": max(
                abs(a - b) for a, b in zip(text_probs, ids_probs)),
        }

    passed = set(quail_out["filter_runs"][0]["ids"])
    filter_bits = [int(i in passed) for i in range(len(reviews))]
    pair_set = {tuple(p) for p in quail_out["join_run"]["pairs"]}
    join_bits = [int((i, j) in pair_set) for i, j in pairs]
    join_labels = [int(left[i][1] == right[j][1]) for i, j in pairs]
    summary = {
        "run": quail_out["run"],
        "model": MODEL,
        "revision": REVISION,
        "filter": compare(filter_bits, ref_text, ref_ids,
                          [lab for _, lab in reviews]),
        "join": compare(join_bits, join_text, join_ids, join_labels),
        "join_anchor": anchor,
        "filter_runs": [{k: r[k] for k in ("seconds", "report")}
                        for r in quail_out["filter_runs"]],
        "join_run": {k: quail_out["join_run"][k] for k in ("seconds", "report")},
        "filter_input_tokens": sum(len(x) for x in quail_out["filter_ids"]),
        "join_input_tokens": sum(len(x) for x in quail_out["join_ids"]),
    }
    path = f"/results/decision_agreement/{quail_out['run']}_summary.json"
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    results.commit()
    return {"path": path, **{k: summary[k] for k in ("filter", "join")}}


@app.local_entrypoint()
def main(quail_call: str = "", quail_only: bool = False):
    """Run both sides.

    Args:
        quail_call: A finished Quail side's function call id to reuse.
        quail_only: Stop after the Quail side, for its timings alone.
    """
    if not quail_call:
        quail_call = quail_side.spawn(time.strftime("%Y%m%d-%H%M%S")).object_id
    print(f"quail_side function call id: {quail_call}", flush=True)
    quail_out = modal.FunctionCall.from_id(quail_call).get()
    for r in quail_out["filter_runs"]:
        rep = r["report"]
        print(f"filter: {r['seconds']:.2f} s client, wall_s={rep.get('wall_s')}, "
              f"boot_s={rep.get('boot_s')}, boot={rep.get('boot')}, "
              f"rows={len(r['ids'])}", flush=True)
    rep = quail_out["join_run"]["report"]
    print(f"join: wall_s={rep.get('wall_s')}, "
          f"pairs={len(quail_out['join_run']['pairs'])}, "
          f"anchor={quail_out['join_anchor']}", flush=True)
    if quail_only:
        return
    ref = reference_side.spawn(quail_out)
    print(f"reference_side function call id: {ref.object_id}", flush=True)
    print(json.dumps(ref.get(), indent=2), flush=True)
