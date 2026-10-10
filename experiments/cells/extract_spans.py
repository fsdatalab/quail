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
- lines: the document is shown as numbered lines of at most 100
  characters, and the model answers with a START-END line range; the
  answer is the text of those lines. Recorded as text F1, as whether
  the range contains the reference span, and as whether it is exactly
  the reference's line range.
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
85% of questions finish in one end pass. The lines baseline contains
the reference span for at least 75% of questions and matches its line
range exactly for at least 60%, with text F1 near 20 to 25.

    uv run modal run --detach -m experiments.cells.extract_spans \
        --model qwen3-4b-fp8 --dataset squad \
        2>&1 | tee /tmp/extract_spans_qwen3_4b_fp8_squad.log

The datasets are SQuAD (short answers in paragraphs) and CUAD (contract
clauses, 31 words at the median, in a 2,500-character window of the
contract around the first reference). Every setting and the prompt are
the same on both. Prediction for CUAD on Qwen3 4B fp8: free generation
and scored both land under 50 F1, within 5 points of each other; the
gold start is among the 8 candidates for at least 60% of questions;
about a third of the references are longer than the 48 fed tokens; the
lines baseline contains the reference for at least 60% of questions
but matches its line range exactly for under 40%.

The per-question records and the summary are written to
/results/extract_spans/<run>_<model>_<dataset>.json on the
quail-results volume.
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
CUAD = ("theatticusproject/cuad", "CUAD_v1/CUAD_v1.json")
# a CUAD question sees this many characters of its contract around the
# first reference clause, which starts at least CUAD_MARGIN in
CUAD_WINDOW = 2500
CUAD_MARGIN = 200
N_ITEMS = 300
SEED = 20261010
TOP_TOKENS = 8      # start tokens read at the cue
MAX_STARTS = 8      # candidate start positions kept after expansion
CUE_LOGPROBS = 1024    # vocabulary entries read at the cue
FEED = 48           # document tokens fed after each candidate start
FREE_MAX_TOKENS = 48
CHUNKS = (8, 16, 32)
TOP_K = (1, 4, 8)

BODY_TEMPLATE = ("DOCUMENT:\n{0}\n\nAnswer the question with the shortest exact "
                 "phrase copied from the document: a name, a number, a date, or a "
                 "few words, never a whole sentence. If the document does not "
                 "answer it, answer none.\nQuestion: {1}")
RAW_PROMPT_TEMPLATE = "{0}\nANSWER: \""
CHAT_TURN_TEMPLATE = ("<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\n\n")
CHAT_PROMPT_TEMPLATE = CHAT_TURN_TEMPLATE + '"'
# the lines baseline: the document is shown as numbered lines of at
# most LINE_WIDTH characters, and the model answers with line numbers
LINE_WIDTH = 100
LINE_MAX_TOKENS = 8
LINE_BODY_TEMPLATE = ("DOCUMENT, as numbered lines:\n{0}\n\nAnswer the question with "
                      "the line numbers of the fewest lines that contain the "
                      "answer, as START-END, for example 3-3 or 5-6. If the "
                      "document does not answer it, answer none.\nQuestion: {1}")
RAW_LINE_TEMPLATE = "{0}\nANSWER: "


def _squad_questions(n: int) -> list[dict]:
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


def _cuad_questions(n: int) -> list[dict]:
    """A seeded sample of answerable CUAD questions on contract windows.

    Each question's document is a CUAD_WINDOW-character window of its
    contract holding the first reference clause, cut at whitespace,
    with the clause starting at a seeded offset into it. The references
    kept are those inside the window, the first one first. The
    question is the category name and the dataset's question text.
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CUAD[0], CUAD[1], repo_type="dataset")
    with open(path) as f:
        contracts = json.load(f)["data"]
    pairs = [(para["context"], qa) for doc in contracts
             for para in doc["paragraphs"] for qa in para["qas"] if qa["answers"]]
    rng = np.random.default_rng(SEED)
    out = []
    for i in rng.permutation(len(pairs))[:n]:
        context, qa = pairs[i]
        first = qa["answers"][0]
        a0, a1 = first["answer_start"], first["answer_start"] + len(first["text"])
        width = max(CUAD_WINDOW, a1 - a0 + 2 * CUAD_MARGIN)
        before = int(rng.integers(CUAD_MARGIN, width - (a1 - a0) - CUAD_MARGIN + 1))
        lo = max(0, a0 - before)
        hi = min(len(context), lo + width)
        while lo > 0 and not context[lo - 1].isspace():
            lo -= 1
        while hi < len(context) and not context[hi].isspace():
            hi += 1
        answers = [a["text"] for a in qa["answers"]
                   if lo <= a["answer_start"]
                   and a["answer_start"] + len(a["text"]) <= hi]
        category = qa["question"].split('"')[1]
        question = qa["question"].split("Details:")[-1].strip()
        out.append({"id": qa["id"], "context": context[lo:hi],
                    "question": f"{category}: {question}", "answers": answers,
                    "answer_start": a0 - lo})
    return out


def _questions(dataset: str, n: int) -> list[dict]:
    """A seeded sample of a dataset's questions."""
    return _squad_questions(n) if dataset == "squad" else _cuad_questions(n)


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


def _lines(context: str) -> list[tuple[int, int]]:
    """Character ranges of the context cut into lines of at most LINE_WIDTH.

    A cut falls at the last space before the limit when there is one,
    so words stay whole; the text itself is unchanged.
    """
    out, start = [], 0
    while start < len(context):
        end = min(start + LINE_WIDTH, len(context))
        if end < len(context):
            space = context.rfind(" ", start + 1, end + 1)
            if space > start:
                end = space
        out.append((start, end))
        start = end
        while start < len(context) and context[start] == " ":
            start += 1
    return out


def _parse_lines(text: str, count: int) -> tuple[int, int] | None:
    """The START-END line range in an answer, 1-based and within the document."""
    import re

    match = re.search(r"(\d+)\s*(?:-|to|–)\s*(\d+)|(\d+)", text)
    if not match:
        return None
    a, b = (match.group(1), match.group(2)) if match.group(1) else (
        match.group(3), match.group(3))
    a, b = int(a), int(b)
    if not 1 <= a <= b <= count:
        return None
    return a, b


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

    def free(self, prompts: list[list[int]], max_tokens: int = FREE_MAX_TOKENS,
             stop=('"', "\n")) -> list[str]:
        """Greedy text after each prompt, cut at a stop string."""
        outputs = self._run(prompts, max_tokens=max_tokens, stop=list(stop))
        return [o.outputs[0].text for o in outputs]

    def line_prompt(self, context: str, question: str) -> list[int]:
        """Token ids of the lines-baseline prompt for one question."""
        lines = _lines(context)
        numbered = "\n".join(f"{i + 1}: {context[a:b]}"
                             for i, (a, b) in enumerate(lines))
        body = LINE_BODY_TEMPLATE.format(numbered, question)
        text = (CHAT_TURN_TEMPLATE if self.chat else RAW_LINE_TEMPLATE).format(body)
        return self.tok.encode(text, add_special_tokens=False)

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


def _best_start(scorer, lp, key, token) -> tuple[int, list[int]]:
    """The likeliest token that begins this document token, and the rest.

    The model may write a word without its leading space, in another
    case, or split differently from the document: any token whose text
    is a prefix of the word, up to case, can start it. A shorter one
    is followed by the rest of the word as the document writes it.
    """
    if not isinstance(key, str) or not key.isascii():
        ids = scorer.variants.get(key, [token]) if isinstance(key, str) else [token]
        return max(ids, key=lambda i: float(lp[i])), []
    options = [(i, j) for j in range(1, len(key) + 1)
               for i in scorer.variants.get(key[:j], [])]
    first, j = max(options, key=lambda o: float(lp[o[0]]))
    text = scorer.vocab[token]
    text = text[1:] if text.startswith("Ġ") else text
    rest = scorer.tok.encode(text[j:], add_special_tokens=False) if j < len(key) else []
    return first, rest


def _candidates(scorer, prompt_ids, lp, doc_ids
                ) -> tuple[list[dict], float, float, int]:
    """Start candidates, the none and best logprobs, and the rounds taken.

    Every document token is scored as the best of the tokens that can
    begin it (`_best_start`). A start token that occurs at several
    positions gets one more step: the logprob of each position's next
    token after it, so the positions are told apart as a trie walk
    would. Positions rank by start logprob plus that step. The rounds
    are 1, or 2 when any start token was ambiguous.
    """
    groups = {}
    for pos, token in enumerate(doc_ids):
        key = _same_word(scorer.vocab[token]) or token
        groups.setdefault(key, []).append(pos)
    best, filler = {}, {}
    for key, positions in groups.items():
        best[key], filler[key] = _best_start(scorer, lp, key, doc_ids[positions[0]])
    ranked = sorted(groups, key=lambda k: -float(lp[best[k]]))[:TOP_TOKENS]
    ambiguous = [key for key in ranked if len(groups[key]) > 1]
    follow = dict(zip(ambiguous, scorer.cues(
        [prompt_ids + [best[key]] for key in ambiguous]))) if ambiguous else {}
    scored = []
    for key in ranked:
        for pos in groups[key]:
            score = float(lp[best[key]])
            if key in follow:
                after = (filler[key] + doc_ids[pos + 1:pos + 2] + [scorer.quote_id])[0]
                score += float(follow[key][after])
            scored.append({"pos": pos, "first": best[key], "score": score,
                           "filler": filler[key]})
    scored.sort(key=lambda c: -c["score"])
    return (scored[:MAX_STARTS], float(lp[scorer.none_id]),
            float(lp[best[ranked[0]]]), 2 if ambiguous else 1)


def _spans(scorer, prompt_ids, doc_ids, candidates):
    """Per candidate: start, copy and stop logprobs along the fed tokens.

    One teacher-forced sequence per candidate gives the start and copy
    logprobs; one more per fed position, ending in the closing quote,
    gives the stop logprob there.
    """
    n = len(prompt_ids)
    feeds = [[c["first"], *c["filler"],
              *doc_ids[c["pos"] + 1:c["pos"] + 1 + FEED - len(c["filler"])]]
             for c in candidates]
    copies = scorer.teacher_forced([
        prompt_ids + fed + doc_ids[c["pos"] + len(fed) - len(c["filler"]):][:1]
        for c, fed in zip(candidates, feeds)])
    stops = scorer.teacher_forced([
        prompt_ids + fed[:i + 1] + [scorer.quote_id]
        for fed in feeds for i in range(len(fed))])
    out, at = [], 0
    for c, fed, lp in zip(candidates, feeds, copies):
        copy = [lp[n + i + 1] if n + i + 1 < len(lp) else float("-inf")
                for i in range(len(fed))]
        stop = [stops[at + i][-1] for i in range(len(fed))]
        at += len(fed)
        out.append({"pos": c["pos"], "first": c["first"], "start": lp[n],
                    "copy": copy, "stop": stop, "fed": len(fed),
                    "extra": len(c["filler"])})
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
def measure(model: str, dataset: str, n: int, run: str) -> dict:
    """Score every question four ways and summarize."""
    import os
    import statistics

    scorer = Scorer(model)
    questions = _questions(dataset, n)
    prompts = [scorer.prompt(q["context"], q["question"]) for q in questions]
    t0 = time.perf_counter()
    free_texts = scorer.free([ids for ids, _ in prompts])
    line_prompts = [scorer.line_prompt(q["context"], q["question"])
                    for q in questions]
    line_texts = scorer.free(line_prompts, LINE_MAX_TOKENS, ("\n",))
    items = []
    for q, (prompt_ids, doc), text, line_ids, line_text in zip(
            questions, prompts, free_texts, line_prompts, line_texts):
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
        lines = _lines(q["context"])
        gold_lines = [i + 1 for i, (a, b) in enumerate(lines)
                      if a < gold_hi and b > gold_lo]
        gold_range = (gold_lines[0], gold_lines[-1]) if gold_lines else None
        chosen = _parse_lines(line_text, len(lines))
        line_answer = (q["context"][lines[chosen[0] - 1][0]:lines[chosen[1] - 1][1]]
                       if chosen else "")
        rec["lines"] = {
            "text": line_text, "range": chosen, "gold_range": gold_range,
            "prompt_tokens": len(line_ids), "parsed": chosen is not None,
            "contains": bool(chosen and gold_range
                             and chosen[0] <= gold_range[0]
                             and gold_range[1] <= chosen[1]),
            "exact": chosen is not None and chosen == gold_range,
            "em_f1": _f1(line_answer, q["answers"])}

        (lp,) = scorer.cues([prompt_ids])
        candidates, none_lp, best_lp, rounds = _candidates(
            scorer, prompt_ids, lp, doc_ids)
        rec["none_wins"] = none_lp > best_lp
        rec["start_rounds"] = rounds
        rec["gold_rank"] = next((i for i, c in enumerate(candidates)
                                 if c["pos"] == gold_start), -1)
        rec["cue_top"] = [(scorer.vocab[t], round(v, 2)) for t, v in
                          sorted(lp.items(), key=lambda kv: -kv[1])[:TOP_TOKENS]]
        rec["candidates"] = [(c["pos"], scorer.vocab[c["first"]],
                              round(c["score"], 2), len(c["filler"]))
                             for c in candidates]
        branches = _spans(scorer, prompt_ids, doc_ids, candidates)

        def span_text(branch, j):
            """The document text of the span ending after fed token j."""
            end = branch["pos"] + max(0, j - branch["extra"])
            return q["context"][offsets[branch["pos"]][0]:offsets[end][1]].strip()

        greedy = branches[0]
        j = next((i for i, (c, s) in enumerate(zip(greedy["copy"], greedy["stop"]))
                  if s > c), greedy["fed"] - 1)
        rec["greedy"] = {"text": span_text(greedy, j),
                         "rounds": j + 2}
        rec["greedy"]["em_f1"] = _f1(rec["greedy"]["text"], q["answers"])
        rec["scored"] = {}
        for renormalized in (False, True):
            for k in TOP_K:
                best = max(((s, b, j) for b in branches[:k]
                            for j, s in enumerate(_scores(b, renormalized))),
                           key=lambda t: t[0])
                answer = span_text(best[1], best[2])
                rec["scored"][f"{'renorm' if renormalized else 'full'}_k{k}"] = {
                    "text": answer, "em_f1": _f1(answer, q["answers"])}
        rec["passes"] = {str(m): _passes(branches, m) for m in CHUNKS}
        if rec["gold_rank"] >= 0:
            branch = branches[rec["gold_rank"]]
            others = [max(_scores(b, False)) for i, b in enumerate(branches)
                      if i != rec["gold_rank"]]
            close = _close_at(branch, max(others) if others else float("-inf"))
            rec["past_gold_end"] = close - rec["gold_len"] - branch["extra"]
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
        "run": run, "model": model, "spec": MODELS[model]["spec"],
        "dataset": dataset, "n": len(items), "seconds": seconds, "feed": FEED,
        "top_tokens": TOP_TOKENS, "max_starts": MAX_STARTS,
        "answer_words_mean": mean([len(q["answers"][0].split()) for q in questions]),
        "answer_tokens_mean": mean([r["gold_len"] for r in items]),
        "answers_longer_than_feed": sum(r["gold_len"] > FEED for r in items),
        "prompt_tokens_mean": mean([r["prompt_tokens"] for r in items]),
        "doc_tokens_mean": mean([r["doc_tokens"] for r in items]),
        "free": em_f1(lambda r: r["free"]["em_f1"]),
        "free_alignment": {how: sum(r["free"]["how"] == how for r in items)
                           for how in ("exact", "case", "fuzzy", "none", "empty")},
        "lines": em_f1(lambda r: r["lines"]["em_f1"]),
        "lines_contains_gold": mean([r["lines"]["contains"] for r in items]),
        "lines_exact_range": mean([r["lines"]["exact"] for r in items]),
        "lines_unparsed": sum(not r["lines"]["parsed"] for r in items),
        "lines_prompt_tokens_mean": mean([r["lines"]["prompt_tokens"] for r in items]),
        "lines_per_document_mean": mean([len(_lines(q["context"]))
                                         for q in questions]),
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
    path = f"/results/extract_spans/{run}_{model}_{dataset}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "items": items}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", dataset: str = "squad", n: int = N_ITEMS):
    """Measure one model on one dataset.

    Args:
        model: A key of MODELS.
        dataset: "squad" or "cuad".
        n: Questions to sample.
    """
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    call = measure.spawn(model, dataset, n, run)
    print(f"measure function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
