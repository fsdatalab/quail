"""Span extraction by scoring document tokens, measured on SQuAD through vLLM.

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

Every method sees one model and one prompt per question, through
vLLM's public API on the checkpoint Quail runs. The copy probabilities
are vLLM's prompt logprobs of the document tokens fed after the start;
the stop probability at each position is the prompt logprob of one
closing-quote token appended there, so a merged token such as `".` is
not counted as a stop. The scored method is recorded under the
model's full-vocabulary probabilities and under probabilities
renormalized over copy and stop.

Prediction (Qwen3 4B fp8, 300 questions): scored with k=8 lands
within 2 F1 points of free generation; the gold start token is among
the 8 candidates for at least 70% of questions; the bound closes at
the gold end for half the questions; with 16-token chunks, at least
85% of questions finish in one end pass.

    uv run modal run --detach -m experiments.cells.extract_spans \
        --model qwen3-4b-fp8 2>&1 | tee /tmp/extract_spans_qwen3_4b_fp8.log

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
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
    "/results": modal.Volume.from_name("quail-results", create_if_missing=True),
}
MODELS = {
    "qwen3-4b-fp8": {"spec": "qwen3-4b-fp8", "chat": True},
    "kai": {"spec": "decision-2.0-kai-0.6b-bf16", "chat": False},
}
SQUAD = ("https://huggingface.co/datasets/rajpurkar/squad/resolve/main/"
         "plain_text/validation-00000-of-00001.parquet")
N_ITEMS = 300
SEED = 20261010
TOP_TOKENS = 8      # start tokens read at the cue
MAX_STARTS = 8      # candidate start positions kept after expansion
CUE_LOGPROBS = 1024    # vocabulary entries read at the cue
FEED = 48           # document tokens fed after each candidate start
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


def _same_word(token) -> str:
    """The key under which tokens differing only in a leading space or case meet."""
    if not isinstance(token, str):
        return ""
    text = token[1:] if token.startswith("Ġ") else token
    return text.lower() if text and "Ġ" not in text and "Ċ" not in text else ""


class Scorer:
    """One model on vLLM, with the tokenizer and the token tables.

    Args:
        name: A key of MODELS.
    """

    def __init__(self, name: str):
        from transformers import AutoTokenizer
        from vllm import LLM

        from quail.backends.quail.executor.model import engine_args, resolve_model_path
        from quail.specs import MODELS as SPECS

        spec = SPECS[MODELS[name]["spec"]]
        path = resolve_model_path(spec.hf_name, spec.revision)
        args = engine_args(path)
        if "hf_overrides" in args:    # a Decision 2.0 package's backbone
            from quail.backends.vllm_decision import register

            register()
        self.llm = LLM(**args, max_model_len=4096, enable_prefix_caching=True,
                       gpu_memory_utilization=0.85, max_logprobs=CUE_LOGPROBS,
                       max_num_seqs=512, max_num_batched_tokens=32768,
                       disable_log_stats=True)
        if "tokenizer" in args:
            # the package root's config names custom model code, which
            # AutoTokenizer would ask to run; the tokenizer is Qwen's
            from transformers import Qwen2TokenizerFast

            self.tok = Qwen2TokenizerFast.from_pretrained(args["tokenizer"])
        else:
            self.tok = AutoTokenizer.from_pretrained(args["model"])
        self.chat = MODELS[name]["chat"]
        vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
        self.vocab = vocab
        self.quote_id = self.tok.convert_tokens_to_ids('"')
        self.none_id = self.tok.encode("none", add_special_tokens=False)[0]
        # a document token may be written without its leading space or
        # in another case after the opening quote; every vocabulary
        # token with the same text up to those counts as the same start
        self.variants = {}
        for i, t in enumerate(vocab):
            key = _same_word(t)
            if key:
                self.variants.setdefault(key, []).append(i)

    def prompt(self, context: str, question: str) -> tuple[list[int], list]:
        """Token ids and the document's (token index, char offsets)."""
        body = BODY_TEMPLATE.format(context, question)
        text = (CHAT_PROMPT_TEMPLATE if self.chat else RAW_PROMPT_TEMPLATE).format(body)
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        start = text.index(context)
        end = start + len(context)
        doc = [(i, (a - start, b - start))
               for i, (a, b) in enumerate(enc["offset_mapping"])
               if a < end and b > start]
        return enc["input_ids"], doc

    def _run(self, sequences: list[list[int]], **params) -> list:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        return self.llm.generate(
            [TokensPrompt(prompt_token_ids=s) for s in sequences],
            SamplingParams(temperature=0.0, **params), use_tqdm=False)

    def free(self, prompts: list[list[int]]) -> list[str]:
        """Greedy text after each prompt, cut at the closing quote or line end."""
        outputs = self._run(prompts, max_tokens=FREE_MAX_TOKENS, stop=['"', "\n"])
        return [o.outputs[0].text for o in outputs]

    def cues(self, sequences: list[list[int]]) -> list[dict]:
        """Logprobs of the CUE_LOGPROBS likeliest next tokens, per sequence."""
        from collections import defaultdict

        outputs = self._run(sequences, max_tokens=1, logprobs=CUE_LOGPROBS,
                            detokenize=False)
        result = []
        for output in outputs:
            lp = defaultdict(lambda: float("-inf"))
            lp.update({t: v.logprob
                       for t, v in output.outputs[0].logprobs[0].items()})
            result.append(lp)
        return result

    def teacher_forced(self, sequences: list[list[int]]) -> list[list[float]]:
        """The logprob of each token given the ones before it, per sequence."""
        outputs = self._run(sequences, max_tokens=1, prompt_logprobs=0,
                            detokenize=False)
        return [[float("nan") if entry is None else entry[token].logprob
                 for token, entry in zip(seq, o.prompt_logprobs)]
                for seq, o in zip(sequences, outputs)]


def _candidates(scorer, prompt_ids, lp, doc_ids) -> tuple[
        list[tuple[int, int, float]], float, float, int]:
    """Start candidates as (position, token id, score), none and best logprobs.

    Every document token is scored as the best of its variants (same
    text up to a leading space and case). A start token that occurs at
    several positions gets one more step: the logprob of each
    position's next document token after it, so the positions are
    told apart as a trie walk would. Positions rank by start logprob
    plus that step. The last value is the number of decode rounds the
    start took: 1, or 2 when any start token was ambiguous.
    """
    groups = {}
    for pos, token in enumerate(doc_ids):
        key = _same_word(scorer.vocab[token]) or token
        groups.setdefault(key, []).append(pos)
    best = {}
    for key in groups:
        ids = scorer.variants.get(key, [key]) if isinstance(key, str) else [key]
        best[key] = max(ids, key=lambda i: float(lp[i]))
    ranked = sorted(groups, key=lambda k: -float(lp[best[k]]))[:TOP_TOKENS]
    ambiguous = [key for key in ranked if len(groups[key]) > 1]
    follow = dict(zip(ambiguous, scorer.cues(
        [prompt_ids + [best[key]] for key in ambiguous]))) if ambiguous else {}
    scored = []
    for key in ranked:
        first = best[key]
        for pos in groups[key]:
            score = float(lp[first])
            if key in follow:
                after = doc_ids[pos + 1] if pos + 1 < len(doc_ids) else scorer.quote_id
                score += float(follow[key][after])
            scored.append((pos, first, score))
    scored.sort(key=lambda t: -t[2])
    return (scored[:MAX_STARTS], float(lp[scorer.none_id]),
            float(lp[best[ranked[0]]]), 2 if ambiguous else 1)


def _spans(scorer, prompt_ids, doc_ids, candidates):
    """Per candidate: start, copy and stop logprobs along the fed tokens.

    One teacher-forced sequence per candidate gives the start and copy
    logprobs; one more per fed position, ending in the closing quote,
    gives the stop logprob there.
    """
    n = len(prompt_ids)
    feeds = [[first, *doc_ids[pos + 1:pos + 1 + FEED]] for pos, first, _ in candidates]
    copies = scorer.teacher_forced([
        prompt_ids + fed + doc_ids[pos + len(fed):pos + len(fed) + 1]
        for (pos, _, _), fed in zip(candidates, feeds)])
    stops = scorer.teacher_forced([
        prompt_ids + fed[:i + 1] + [scorer.quote_id]
        for fed in feeds for i in range(len(fed))])
    out, at = [], 0
    for (pos, first, _), fed, lp in zip(candidates, feeds, copies):
        copy = [lp[n + i + 1] if n + i + 1 < len(lp) else float("-inf")
                for i in range(len(fed))]
        stop = [stops[at + i][-1] for i in range(len(fed))]
        at += len(fed)
        out.append({"pos": pos, "first": first, "start": lp[n],
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
    questions = _questions(n)
    prompts = [scorer.prompt(q["context"], q["question"]) for q in questions]
    t0 = time.perf_counter()
    free_texts = scorer.free([ids for ids, _ in prompts])
    items = []
    for q, (prompt_ids, doc), text in zip(questions, prompts, free_texts):
        doc_ids = [prompt_ids[i] for i, _ in doc]
        offsets = [span for _, span in doc]
        gold_lo = q["answer_start"]
        gold_hi = gold_lo + len(q["answers"][0])
        gold = [p for p, (a, b) in enumerate(offsets) if a < gold_hi and b > gold_lo]
        gold_start, gold_end = (gold[0], gold[-1]) if gold else (-1, -1)

        how, aligned = _align(text, q["context"])
        rec = {"id": q["id"], "answers": q["answers"], "prompt_tokens": len(prompt_ids),
               "doc_tokens": len(doc_ids), "gold_start": gold_start,
               "gold_len": gold_end - gold_start + 1,
               "gold_token": scorer.vocab[doc_ids[gold_start]] if gold else "",
               "free": {"text": text, "aligned": aligned, "how": how,
                        "em_f1": _f1(aligned, q["answers"])}}

        (lp,) = scorer.cues([prompt_ids])
        candidates, none_lp, best_lp, rounds = _candidates(
            scorer, prompt_ids, lp, doc_ids)
        rec["none_wins"] = none_lp > best_lp
        rec["start_rounds"] = rounds
        rec["gold_rank"] = next((i for i, (p, _, _) in enumerate(candidates)
                                 if p == gold_start), -1)
        rec["cue_top"] = [(scorer.vocab[t], round(v, 2)) for t, v in
                          sorted(lp.items(), key=lambda kv: -kv[1])[:TOP_TOKENS]]
        rec["candidates"] = [(p, scorer.vocab[t], round(v, 2))
                             for p, t, v in candidates]
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
        "run": run, "model": model, "spec": MODELS[model]["spec"], "n": len(items),
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
def main(model: str = "qwen3-4b-fp8", n: int = N_ITEMS):
    """Measure one model.

    Args:
        model: A key of MODELS.
        n: Questions to sample.
    """
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = measure.spawn(model, n, run)
    print(f"measure function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
