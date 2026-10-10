"""Span extraction by scoring document tokens, measured on SQuAD.

Three ways to answer an extractive question with a decoder model, on
the same prompt, with no training:

- free: greedy generation, then the text is aligned to the document
  (exact match first, then the longest common block); this is what a
  stock vLLM baseline would do.
- greedy copy: the LogitMatch rule, one decode step at a time; the
  first token must come from the document, each later token is the
  next document token or a closing quote, and the answer stops at the
  first step where the quote beats the copy.
- scored: the top-k start tokens are read in one step; after each
  candidate start the next FEED document tokens are fed in one pass,
  and every (start, end) span is scored as the product of its copy
  probabilities and its stop probability (exact-extract). The best
  span wins. The bound "no longer span can beat the best so far once
  the copy product falls below it" is replayed offline for each chunk
  size in CHUNKS to count the passes the engine would make.

Every method sees one model and one prompt per question. The scored
method is recorded under the model's full-vocabulary probabilities
and under probabilities renormalized over copy and stop.

The Kai model is a decision model: its backbone is Qwen3 0.6B with a
tied embedding, so it has next-token logits, but its fine-tuning
trained the decision head, not those logits. Qwen3 0.6B is the
same-size control.

Prediction (Qwen3 0.6B, 300 questions): scored with k=8 lands within
2 F1 points of greedy copy and at least 5 points above free
generation; the gold start token is among the 8 candidates for at
least 85% of questions; the bound closes within 3 tokens of the gold
end for half the questions; with 16-token chunks, at least 90% of
questions finish in one end pass. Kai scores lower on every number;
if its scored F1 is under 30, the executor work targets Qwen3.

    uv run modal run --detach -m experiments.cells.extract_spans \
        --model kai 2>&1 | tee /tmp/extract_spans_kai.log
    uv run modal run --detach -m experiments.cells.extract_spans \
        --model qwen3-0.6b 2>&1 | tee /tmp/extract_spans_qwen3_0_6b.log

The per-question records and the summary are written to
/results/extract_spans/<run>_<model>.json on the quail-results volume.
"""

import json
import time

import modal

try:
    from quail.bench.images import gpu_image
    image = gpu_image()
except ImportError:    # a container without the local quail package
    image = None

app = modal.App("quail-milestone1")
VOLUMES = {
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
MODELS = {
    "kai": {"hf": "vllm-sr/Decision-2.0-Kai-0.6B",
            "revision": "881bee413681d80ebeac86afcda8b4138dae516e",
            "backbone": "backbone", "chat": False},
    "qwen3-0.6b": {"hf": "Qwen/Qwen3-0.6B", "revision": "",
                   "backbone": "", "chat": True},
    "qwen3-4b": {"hf": "Qwen/Qwen3-4B", "revision": "",
                 "backbone": "", "chat": True},
}
SQUAD = ("https://huggingface.co/datasets/rajpurkar/squad/resolve/main/"
         "plain_text/validation-00000-of-00001.parquet")
N_ITEMS = 300
SEED = 20261010
TOP_TOKENS = 8      # start tokens read at the cue
MAX_STARTS = 8      # candidate start positions kept after expansion
FEED = 64           # document tokens fed after each candidate start
FREE_MAX_TOKENS = 48
CHUNKS = (8, 16, 32)
TOP_K = (1, 4, 8)

BODY_TEMPLATE = ("DOCUMENT:\n{0}\n\nQuote the exact words from the document "
                 "that answer the question, or answer none.\nQuestion: {1}")
RAW_PROMPT_TEMPLATE = "{0}\nANSWER: \""
CHAT_PROMPT_TEMPLATE = ("<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n"
                        "<think>\n\n</think>\n\n\"")


def _questions(n: int) -> list[dict]:
    """A seeded sample of SQuAD validation questions."""
    import io
    import urllib.request

    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(urllib.request.urlopen(SQUAD).read()))
    rows = table.to_pylist()
    order = np.random.default_rng(SEED).permutation(len(rows))[:n]
    return [{"id": rows[i]["id"], "context": rows[i]["context"],
             "question": rows[i]["question"],
             "answers": rows[i]["answers"]["text"],
             "answer_start": rows[i]["answers"]["answer_start"][0]}
            for i in order]


def _normalize(text: str) -> list[str]:
    """SQuAD's answer normalization: case, punctuation, articles, spaces."""
    import re
    import string

    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def _f1(prediction: str, answers: list[str]) -> tuple[float, float]:
    """SQuAD exact match and token F1 against the best reference."""
    from collections import Counter

    pred = _normalize(prediction)
    best_em, best_f1 = 0.0, 0.0
    for answer in answers:
        gold = _normalize(answer)
        best_em = max(best_em, float(pred == gold))
        common = sum((Counter(pred) & Counter(gold)).values())
        if common:
            p, r = common / len(pred), common / len(gold)
            best_f1 = max(best_f1, 2 * p * r / (p + r))
    return best_em, best_f1


def _align(text: str, context: str) -> tuple[str, str]:
    """Map generated text onto the context: the method and the span text."""
    import difflib

    text = text.strip()
    if not text:
        return "empty", ""
    if text in context:
        return "exact", text
    lowered = context.lower().find(text.lower())
    if lowered >= 0:
        return "case", context[lowered:lowered + len(text)]
    match = difflib.SequenceMatcher(None, context, text).find_longest_match(
        0, len(context), 0, len(text))
    if match.size >= len(text) / 2:
        return "fuzzy", context[match.a:match.a + match.size]
    return "none", text


class Scorer:
    """One model's backbone and output head on the GPU.

    Args:
        name: A key of MODELS.
    """

    def __init__(self, name: str):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Model

        spec = MODELS[name]
        path = snapshot_download(spec["hf"], revision=spec["revision"] or None)
        self.tok = AutoTokenizer.from_pretrained(path)
        if spec["backbone"]:
            self.backbone = Qwen3Model.from_pretrained(
                f"{path}/{spec['backbone']}", dtype=torch.bfloat16).cuda().eval()
            self.head = self.backbone.embed_tokens.weight
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.bfloat16).cuda().eval()
            self.backbone, self.head = model.model, model.lm_head.weight
        self.chat = spec["chat"]
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        self.quote_ids = [i for i, t in enumerate(vocab)
                          if isinstance(t, str) and t.startswith('"')]
        self.none_id = self.tok.encode("none", add_special_tokens=False)[0]
        self.eos_ids = {self.tok.eos_token_id,
                        self.tok.convert_tokens_to_ids("<|im_end|>")}
        self.stripped = {}
        for i, t in enumerate(vocab):
            if isinstance(t, str) and t.startswith("Ġ") and len(t) > 1:
                j = self.tok.convert_tokens_to_ids(t[1:])
                if j is not None and j != self.tok.unk_token_id:
                    self.stripped[i] = j

    def prompt(self, context: str, question: str) -> tuple[list[int], list, int]:
        """Token ids, the document's (token index, char offsets), prompt chars."""
        body = BODY_TEMPLATE.format(context, question)
        text = (CHAT_PROMPT_TEMPLATE if self.chat else RAW_PROMPT_TEMPLATE).format(body)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        start = text.index(context)
        end = start + len(context)
        doc = [(i, (a - start, b - start))
               for i, (a, b) in enumerate(enc["offset_mapping"])
               if a < end and b > start]
        return enc["input_ids"], doc, len(text)

    def rows(self, sequences: list[list[int]], rows: list[list[int]]):
        """Log-probabilities over the vocabulary at the given rows."""
        import torch

        width = max(len(s) for s in sequences)
        pad = self.tok.pad_token_id or 0
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for b, s in enumerate(sequences):
            ids[b, :len(s)] = torch.tensor(s)
            mask[b, :len(s)] = 1
        with torch.no_grad():
            hidden = self.backbone(input_ids=ids.cuda(), attention_mask=mask.cuda()
                                   ).last_hidden_state
            out = []
            for b, wanted in enumerate(rows):
                logits = hidden[b, wanted].float() @ self.head.float().T
                out.append(torch.log_softmax(logits, -1).cpu())
        return out

    def generate(self, ids: list[int]) -> str:
        """Greedy text after the prompt, cut at the closing quote or line end."""
        import torch

        out, past = [], None
        step = torch.tensor([ids]).cuda()
        with torch.no_grad():
            for _ in range(FREE_MAX_TOKENS):
                result = self.backbone(input_ids=step, past_key_values=past,
                                       use_cache=True)
                past = result.past_key_values
                logits = result.last_hidden_state[0, -1].float() @ self.head.float().T
                token = int(logits.argmax())
                if token in self.eos_ids:
                    break
                out.append(token)
                step = torch.tensor([[token]]).cuda()
        text = self.tok.decode(out)
        for stop in ('"', "\n"):
            text = text.split(stop)[0]
        return text


def _candidates(scorer, lp, doc_ids) -> tuple[list[tuple[int, int]], float, float]:
    """Start candidates as (position, token id), with the none and best logprobs.

    The allowed tokens are the document's tokens and, for a token that
    starts with a space, the same text without it; each maps back to
    every position it occurs at, in document order.
    """
    allowed = {}
    for pos, token in enumerate(doc_ids):
        allowed.setdefault(token, []).append((pos, token))
        if token in scorer.stripped:
            allowed.setdefault(scorer.stripped[token], []).append((pos, token))
    ranked = sorted(allowed, key=lambda t: -float(lp[t]))[:TOP_TOKENS]
    out, seen = [], set()
    for token in ranked:
        for pos, _ in allowed[token]:
            if pos not in seen and len(out) < MAX_STARTS:
                seen.add(pos)
                out.append((pos, token))
    return out, float(lp[scorer.none_id]), float(lp[ranked[0]])


def _spans(scorer, prompt_ids, doc_ids, candidates):
    """Per candidate: start, copy and stop logprobs along the fed tokens."""
    import torch

    sequences, rows = [], []
    for pos, first in candidates:
        fed = [first, *doc_ids[pos + 1:pos + 1 + FEED]]
        sequences.append(prompt_ids + fed)
        rows.append(list(range(len(prompt_ids) - 1, len(prompt_ids) + len(fed))))
    quote = torch.tensor(scorer.quote_ids)
    out = []
    for (pos, first), lp in zip(candidates, scorer.rows(sequences, rows)):
        fed = [first, *doc_ids[pos + 1:pos + 1 + FEED]]
        nxt = doc_ids[pos + 1:pos + 1 + len(fed)]
        copy = [float(lp[i + 1, nxt[i]]) if i < len(nxt) else float("-inf")
                for i in range(len(fed))]
        stop = [float(torch.logsumexp(lp[i + 1, quote], 0)) for i in range(len(fed))]
        out.append({"pos": pos, "first": first, "start": float(lp[0, first]),
                    "copy": copy, "stop": stop, "fed": len(fed)})
    return out


def _scores(branch, renormalized: bool) -> list[float]:
    """Score of the span ending after each fed token."""
    import math

    copy, stop = branch["copy"], branch["stop"]
    if renormalized:
        both = [math.log(math.exp(c) + math.exp(s)) if c > -math.inf else s
                for c, s in zip(copy, stop)]
        copy = [c - b for c, b in zip(copy, both)]
        stop = [s - b for s, b in zip(stop, both)]
    total, out = branch["start"], []
    for c, s in zip(copy, stop):
        out.append(total + s)
        total += c
    return out


def _passes(branches, chunk: int) -> int:
    """End passes the bound needs when the engine feeds `chunk` tokens a pass."""
    scores = [_scores(b, False) for b in branches]
    running = []
    for b in branches:
        total, run = b["start"], []
        for c in b["copy"]:
            total += c
            run.append(total)
        running.append(run)
    fed, passes = 0, 0
    open_branches = list(range(len(branches)))
    while open_branches and fed < FEED:
        fed += chunk
        passes += 1
        best = max(max(s[:fed]) for s in scores)
        open_branches = [i for i in open_branches
                         if fed < branches[i]["fed"] and running[i][fed - 1] >= best]
    return passes


def _close_at(branch, best_other: float) -> int:
    """Tokens fed after the start before no longer span can win."""
    scores = _scores(branch, False)
    total, best = branch["start"], best_other
    for i, c in enumerate(branch["copy"]):
        best = max(best, scores[i])
        total += c
        if total < best:
            return i + 1
    return branch["fed"]


@app.function(image=image, gpu="H100!", memory=65536, timeout=7200,
              volumes=VOLUMES)
def measure(model: str, n: int, run: str) -> dict:
    """Score every question three ways and summarize."""
    import os
    import statistics

    scorer = Scorer(model)
    items = []
    t0 = time.perf_counter()
    for q in _questions(n):
        prompt_ids, doc, _ = scorer.prompt(q["context"], q["question"])
        doc_ids = [prompt_ids[i] for i, _ in doc]
        offsets = [span for _, span in doc]
        gold_lo = q["answer_start"]
        gold_hi = gold_lo + len(q["answers"][0])
        gold = [p for p, (a, b) in enumerate(offsets) if a < gold_hi and b > gold_lo]
        gold_start, gold_end = (gold[0], gold[-1]) if gold else (-1, -1)

        text = scorer.generate(prompt_ids)
        how, aligned = _align(text, q["context"])
        rec = {"id": q["id"], "answers": q["answers"], "prompt_tokens": len(prompt_ids),
               "doc_tokens": len(doc_ids), "gold_start": gold_start,
               "gold_len": gold_end - gold_start + 1,
               "free": {"text": text, "aligned": aligned, "how": how,
                        "em_f1": _f1(aligned, q["answers"])}}

        (lp,) = scorer.rows([prompt_ids], [[len(prompt_ids) - 1]])
        candidates, none_lp, best_lp = _candidates(scorer, lp[0], doc_ids)
        rec["none_wins"] = none_lp > best_lp
        rec["gold_rank"] = next((i for i, (p, _) in enumerate(candidates)
                                 if p == gold_start), -1)
        branches = _spans(scorer, prompt_ids, doc_ids, candidates)

        def span_text(pos, length):
            return q["context"][offsets[pos][0]:offsets[pos + length - 1][1]].strip()

        greedy = branches[0]
        j = next((i for i, (c, s) in enumerate(zip(greedy["copy"], greedy["stop"]))
                  if s > c), greedy["fed"] - 1)
        rec["greedy"] = {"text": span_text(greedy["pos"], j + 1),
                         "rounds": j + 2}
        rec["greedy"]["em_f1"] = _f1(rec["greedy"]["text"], q["answers"])
        rec["scored"] = {}
        for renormalized in (False, True):
            for k in TOP_K:
                best = max(((s, b["pos"], j) for b in branches[:k]
                            for j, s in enumerate(_scores(b, renormalized))),
                           key=lambda t: t[0])
                answer = span_text(best[1], best[2] + 1)
                rec["scored"][f"{'renorm' if renormalized else 'full'}_k{k}"] = {
                    "text": answer, "em_f1": _f1(answer, q["answers"])}
        rec["passes"] = {str(m): _passes(branches, m) for m in CHUNKS}
        if rec["gold_rank"] >= 0:
            branch = branches[rec["gold_rank"]]
            others = [max(_scores(b, False)) for i, b in enumerate(branches)
                      if i != rec["gold_rank"]]
            close = _close_at(branch, max(others) if others else float("-inf"))
            rec["past_gold_end"] = close - rec["gold_len"]
        items.append(rec)
    seconds = time.perf_counter() - t0

    def mean(values):
        return statistics.fmean(values) if values else None

    def em_f1(select):
        pairs = [select(r) for r in items]
        return {"em": 100 * mean([p[0] for p in pairs]),
                "f1": 100 * mean([p[1] for p in pairs])}

    past = [r["past_gold_end"] for r in items if "past_gold_end" in r]
    summary = {
        "run": run, "model": model, "hf": MODELS[model]["hf"], "n": len(items),
        "seconds": seconds, "feed": FEED, "top_tokens": TOP_TOKENS,
        "max_starts": MAX_STARTS,
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in items]),
        "doc_tokens_mean": mean([r["doc_tokens"] for r in items]),
        "free": em_f1(lambda r: r["free"]["em_f1"]),
        "free_alignment": {how: sum(r["free"]["how"] == how for r in items)
                           for how in ("exact", "case", "fuzzy", "none", "empty")},
        "greedy": em_f1(lambda r: r["greedy"]["em_f1"]),
        "greedy_rounds_mean": mean([r["greedy"]["rounds"] for r in items]),
        "scored": {key: em_f1(lambda r, key=key: r["scored"][key]["em_f1"])
                   for key in items[0]["scored"]},
        "start_recall": {str(k): mean([0 <= r["gold_rank"] < k for r in items])
                         for k in TOP_K},
        "none_wins": sum(r["none_wins"] for r in items),
        "past_gold_end": {"n": len(past), "mean": mean(past),
                          "median": statistics.median(past) if past else None,
                          "p95": (sorted(past)[int(0.95 * len(past)) - 1]
                                  if past else None)},
        "one_pass_share": {str(m): mean([r["passes"][str(m)] == 1 for r in items])
                           for m in CHUNKS},
        "passes_mean": {str(m): mean([r["passes"][str(m)] for r in items])
                        for m in CHUNKS},
    }
    path = f"/results/extract_spans/{run}_{model}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "items": items}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.local_entrypoint()
def main(model: str = "kai", n: int = N_ITEMS):
    """Measure one model.

    Args:
        model: A key of MODELS.
        n: Questions to sample.
    """
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = measure.spawn(model, n, run)
    print(f"measure function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
