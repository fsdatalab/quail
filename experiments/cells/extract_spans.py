"""Span extraction by locating lines and scoring document tokens, through vLLM.

One method, with two paths chosen by the document's own line count:

- A document of at most 2 lines is answered by scoring. The top-k
  start tokens are read at the open quote in one step; after each
  candidate start the next CHUNK document tokens are fed in one pass,
  and every (start, end) span is scored as the product of its copy
  probabilities and its stop probability (exact-extract). A start
  whose copy product is still above the best span found gets another
  CHUNK tokens, up to MAX_PASSES passes; the best span wins.
- A document of 3 or more lines is answered by locating, then
  scoring. The document is shown with its non-empty lines numbered,
  as `cat -n` would, and an answer template of a Lines field and a
  Phrase field; the cue "Lines: " has the model write a range a-b; the
  prompt continues with the Phrase field's open quote, and
  the start step above runs with the tokens of lines a to b as the
  only allowed starts, the first token of line a always among them.
  From each start the tokens up to the start of line b are fed in one
  pass, since they are inside the answer, and from there the chunked
  scoring above finds the end. A range the model does not write as
  numbers falls back to the scoring path on the plain document.

Every question goes through vLLM's public API on the checkpoint Quail
runs. The copy probabilities are vLLM's prompt logprobs of the fed
tokens; the stop probability at each scored position is the prompt
logprob of one closing-quote token appended there, so a merged token
such as `".` is not counted as a stop. The first token may be written
without its leading space, in another case, or as a prefix of the
document's token; the rest of the word is then fed as the document
writes it.

The record keeps, per question, the path taken, the line range and
whether it contains the reference, the rank of the reference's start
among the candidates, the end passes, and the SQuAD exact match and
F1 of the answer; for the locating path also the F1 of lines a to b
as they are, for comparison.

Prediction (Qwen3 4B fp8, 300 questions each): on SQuAD every
paragraph is one line, so the scoring path runs alone and repeats its
82.2 F1 (/results/extract_spans/20261011T001712Z_qwen3-4b-fp8_squad.json).
On CUAD, in 2,500-character contract windows, the locating path runs
for at least 270 questions; its F1 is at least 50, above the 44.6 of
answering with the whole lines; the reference's start is among the
candidates for at least 70% of the questions whose range contains it;
the end takes at most 2 chunk passes on average.

    uv run modal run --detach -m experiments.cells.extract_spans \
        --model qwen3-4b-fp8 --dataset cuad \
        2>&1 | tee /tmp/extract_spans_qwen3_4b_fp8_cuad.log

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
CHUNK = 16          # document tokens fed per end pass
MAX_PASSES = 8      # end passes a question may take: a runaway guard
LOCATE_MIN_LINES = 3    # documents with fewer lines are scored directly
LINE_MAX_TOKENS = 8
TOP_K = (1, 4, 8)

# what an answer is, the same on both paths; the question carries the
# length, a date or a whole clause
INSTRUCTION_TEXT = (
    "Copy the words that answer the question from the document. Copy them "
    "exactly. Use the fewest words that answer it. If the document does not "
    "answer the question, write none.")
FORMAT_TEXT = "Answer in this format:"
LINES_FORMAT_TEXT = "Lines: <first line>-<last line>"
PHRASE_FORMAT_TEXT = 'Phrase: "<copied words>"'
BODY_TEMPLATE = ("DOCUMENT:\n{0}\n\nQuestion: {1}\n\n" + INSTRUCTION_TEXT + "\n"
                 + FORMAT_TEXT + "\n" + PHRASE_FORMAT_TEXT)
LINE_BODY_TEMPLATE = ("DOCUMENT, as numbered lines:\n{0}\n\nQuestion: {1}\n\n"
                      + INSTRUCTION_TEXT + "\n" + FORMAT_TEXT + "\n"
                      + LINES_FORMAT_TEXT + "\n" + PHRASE_FORMAT_TEXT)
# the answer is begun for the model: the scoring path's answer opens
# with the phrase's quote, the locating path's with the lines field,
# so the model writes the range and nothing else
PHRASE_CUE = 'Phrase: "'
LINE_CUE = "Lines: "
BEGINS_CUE = "\n" + PHRASE_CUE
CHAT_TURN_TEMPLATE = ("<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\n\n")
RAW_PROMPT_TEMPLATE = "{0}\n" + PHRASE_CUE
CHAT_PROMPT_TEMPLATE = CHAT_TURN_TEMPLATE + PHRASE_CUE
RAW_LINE_TEMPLATE = "{0}\n" + LINE_CUE
CHAT_LINE_TEMPLATE = CHAT_TURN_TEMPLATE + LINE_CUE


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


def _lines(context: str) -> list[tuple[int, int]]:
    """Character ranges of the context's non-empty lines, as `cat -n` numbers them.

    Blank lines get no number, so the model does not have to count
    them; the text of each line is unchanged.
    """
    out, start = [], 0
    for line in context.split("\n"):
        end = start + len(line)
        if line.strip():
            out.append((start, end))
        start = end + 1
    return out


def _parse_lines(text: str, count: int) -> tuple[int, int] | None:
    """The START-END line range in an answer, 1-based and within the document."""
    import re

    # the range the cue begins, or one named as lines in a sentence; a
    # number elsewhere may be the span's text
    span = r"(\d+)(?:\s*(?:-|to|–|and)\s*(\d+))?"
    match = (re.match(rf"\s*{span}(?!\d)", text)
             or re.search(rf"\blines?\s+{span}", text, re.I))
    if not match:
        return None
    a, b = int(match.group(1)), int(match.group(2) or match.group(1))
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

    def _doc_tokens(self, text: str, pieces: list[tuple[int, int, int]]) -> tuple:
        """Token ids of a prompt, and its document tokens with context offsets.

        Args:
            text: The prompt text.
            pieces: (offset in text, offset in context, length) of each
                stretch of document text the prompt shows verbatim.

        Returns:
            The token ids, and (token index, (context start, context end))
            for each token overlapping one of the pieces, clipped to it:
            a line's first token carries the space after its number,
            and its last may carry the line break.
        """
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        doc = []
        for i, (a, b) in enumerate(enc["offset_mapping"]):
            for off, ctx, length in pieces:
                lo, hi = max(a, off), min(b, off + length)
                if hi > lo:
                    doc.append((i, (ctx + lo - off, ctx + hi - off)))
                    break
        return enc["input_ids"], doc

    def prompt(self, context: str, question: str) -> tuple[list[int], list]:
        """The scoring path's prompt: token ids and the document tokens."""
        body = BODY_TEMPLATE.format(context, question)
        text = (CHAT_PROMPT_TEMPLATE if self.chat else RAW_PROMPT_TEMPLATE).format(body)
        return self._doc_tokens(text, [(text.index(context), 0, len(context))])

    def line_prompt(self, context: str, question: str) -> tuple[str, list]:
        """The locating path's prompt text and its lines' pieces.

        Returns:
            The text, and (offset in text, offset in context, length) of
            each line's text in it.
        """
        lines = _lines(context)
        numbered, pieces, at = [], [], 0
        for i, (a, b) in enumerate(lines):
            head = f"{i + 1}: "
            pieces.append((at + len(head), a, b - a))
            numbered.append(head + context[a:b])
            at += len(head) + (b - a) + 1
        body = LINE_BODY_TEMPLATE.format("\n".join(numbered), question)
        text = (CHAT_LINE_TEMPLATE if self.chat else RAW_LINE_TEMPLATE).format(body)
        shift = text.index(numbered[0])
        return text, [(off + shift, ctx, length) for off, ctx, length in pieces]

    def _run(self, sequences: list[list[int]], **params) -> list:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        return self.llm.generate(
            [TokensPrompt(prompt_token_ids=s) for s in sequences],
            SamplingParams(temperature=0.0, **params), use_tqdm=False)

    def generate(self, prompts: list[list[int]], max_tokens: int,
                 stop=("\n",)) -> list[str]:
        """Greedy text after each prompt, cut at a stop string."""
        outputs = self._run(prompts, max_tokens=max_tokens, stop=list(stop))
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


def _candidates(scorer, prompt_ids, lp, ids, doc, region=None
                ) -> tuple[list[dict], float, float, int]:
    """Start candidates, the none and best logprobs, and the rounds taken.

    Every document token in the region is scored as the best of the
    tokens that can begin it (`_best_start`). A start token that occurs
    at several positions gets one more step: the logprob of each
    position's next token after it, so the positions are told apart as
    a trie walk would. Positions rank by start logprob plus that step.
    With a region, its first token is always a candidate. The rounds
    are 1, or 2 when any start token was ambiguous.

    Args:
        scorer: The model.
        prompt_ids: The prompt the cue row ends.
        lp: The cue row's logprobs.
        ids: The prompt's token ids.
        doc: The prompt's document tokens, (index, (context start, end)).
        region: Context character range the start must lie in, or None.
    """
    inside = [(i, span) for i, span in doc
              if region is None or region[0] <= span[0] < region[1]]
    groups = {}
    for i, span in inside:
        key = _same_word(scorer.vocab[ids[i]]) or ids[i]
        groups.setdefault(key, []).append((i, span))
    best, filler = {}, {}
    for key, occurrences in groups.items():
        best[key], filler[key] = _best_start(scorer, lp, key, ids[occurrences[0][0]])
    ranked = sorted(groups, key=lambda k: -float(lp[best[k]]))[:TOP_TOKENS]
    ambiguous = [key for key in ranked if len(groups[key]) > 1]
    follow = dict(zip(ambiguous, scorer.cues(
        [prompt_ids + [best[key]] for key in ambiguous]))) if ambiguous else {}

    def candidate(key, i, span):
        score = float(lp[best[key]])
        if key in follow:
            after = (filler[key] + ids[i + 1:i + 2] + [scorer.quote_id])[0]
            score += float(follow[key][after])
        return {"char": span[0], "end": span[1], "first": best[key],
                "score": score, "filler": filler[key]}

    scored = sorted((candidate(key, i, span) for key in ranked
                     for i, span in groups[key]), key=lambda c: -c["score"])
    scored = scored[:MAX_STARTS]
    if inside and all(c["char"] != inside[0][1][0] for c in scored):
        i, span = inside[0]
        key = _same_word(scorer.vocab[ids[i]]) or ids[i]
        if key not in best:
            best[key], filler[key] = _best_start(scorer, lp, key, ids[i])
        scored.append(candidate(key, i, span))
    return (scored, float(lp[scorer.none_id]),
            float(lp[best[ranked[0]]]) if ranked else float("-inf"),
            2 if ambiguous else 1)


def _spans(scorer, prompt_ids, context, candidates, skip_to=None
           ) -> tuple[list[dict], int]:
    """Per candidate: start, copy and stop logprobs along the fed tokens.

    The first pass feeds, for every candidate, the document up to the
    character `skip_to` (the start of the range's last line, when the
    range has more than one line) and then CHUNK tokens more; later
    passes feed CHUNK tokens to each open candidate. Ends are scored
    only from `skip_to` on. One teacher-forced sequence per candidate
    gives the start and copy logprobs; one more per scored position,
    ending in the closing quote, gives the stop logprob there. A
    candidate stays open while its copy product is at least the best
    span score so far, it has document left, and fewer than MAX_PASSES
    passes have run.

    Returns:
        The candidates' branches and the number of passes.
    """
    n = len(prompt_ids)
    branches = []
    for c in candidates:
        rest = scorer.tok(context[c["end"]:], return_offsets_mapping=True,
                          add_special_tokens=False)
        stream = [c["first"], *c["filler"], *rest["input_ids"]]
        ends = ([c["end"]] * (1 + len(c["filler"]))
                + [c["end"] + b for _, b in rest["offset_mapping"]])
        skip = sum(e <= skip_to for e in ends) if skip_to else 0
        branches.append({"char": c["char"], "first": c["first"], "start": None,
                         "copy": [], "stop": [], "fed": 0, "ends": ends,
                         "stream": stream, "skip": min(skip, len(stream))})
    open_branches = list(range(len(branches)))
    passes = 0
    while open_branches and passes < MAX_PASSES:
        passes += 1
        news = {}
        for i in open_branches:
            b = branches[i]
            take = (b["skip"] if passes == 1 else 0) + CHUNK
            news[i] = b["stream"][b["fed"]:b["fed"] + take]
        copies = scorer.teacher_forced([
            prompt_ids + branches[i]["stream"][:branches[i]["fed"] + len(news[i]) + 1]
            for i in open_branches])
        scored_rows = [(i, j) for i in open_branches for j in range(len(news[i]))
                       if branches[i]["fed"] + j >= branches[i]["skip"]]
        stops = dict(zip(scored_rows, scorer.teacher_forced([
            prompt_ids + branches[i]["stream"][:branches[i]["fed"] + j + 1]
            + [scorer.quote_id] for i, j in scored_rows]))) if scored_rows else {}
        for i, lp in zip(open_branches, copies):
            b = branches[i]
            if b["start"] is None:
                b["start"] = lp[n]
            for j in range(len(news[i])):
                row = n + b["fed"] + j + 1
                b["copy"].append(lp[row] if row < len(lp) else float("-inf"))
                b["stop"].append(stops[(i, j)][-1] if (i, j) in stops
                                 else float("-inf"))
            b["fed"] += len(news[i])
        best = max(max(_scores(b)) for b in branches if b["fed"])
        open_branches = [
            i for i in open_branches
            if branches[i]["fed"] < len(branches[i]["stream"])
            and branches[i]["start"] + sum(branches[i]["copy"]) >= best]
    for b in branches:
        del b["stream"]
    return branches, passes


def _scores(branch) -> list[float]:
    """Score of the span ending after each fed token."""
    total, out = branch["start"], []
    for c, s in zip(branch["copy"], branch["stop"]):
        out.append(total + s)
        total += c
    return out


def _close_at(branch, best_other: float) -> int:
    """Tokens fed after the start before no longer span can win."""
    scores = _scores(branch)
    total, best = branch["start"], best_other
    for i, c in enumerate(branch["copy"]):
        best = max(best, scores[i])
        total += c
        if total < best:
            return i + 1
    return branch["fed"]


def _answer(context, branches, k) -> str:
    """The best span over the first k branches, or "" when none was fed."""
    spans = [(s, b, j) for b in branches[:k] for j, s in enumerate(_scores(b))]
    if not spans:
        return ""
    _, b, j = max(spans, key=lambda t: t[0])
    return context[b["char"]:b["ends"][j]].strip()


@app.function(image=image, gpu="H100!", memory=65536, timeout=7200,
              volumes=VOLUMES)
def measure(model: str, dataset: str, n: int, run: str) -> dict:
    """Answer every question and summarize."""
    import os
    import statistics

    scorer = Scorer(model)
    questions = _questions(dataset, n)
    t0 = time.perf_counter()
    located = [i for i, q in enumerate(questions)
               if len(_lines(q["context"])) >= LOCATE_MIN_LINES]
    line_texts = {}
    if located:
        prompts = {i: scorer.line_prompt(questions[i]["context"],
                                         questions[i]["question"]) for i in located}
        answers = scorer.generate(
            [scorer.tok.encode(prompts[i][0], add_special_tokens=False)
             for i in located], LINE_MAX_TOKENS)
        line_texts = dict(zip(located, answers))
    items = []
    for qi, q in enumerate(questions):
        context = q["context"]
        lines = _lines(context)
        gold_lo = q["answer_start"]
        gold_hi = gold_lo + len(q["answers"][0])
        rec = {"id": q["id"], "answers": q["answers"], "lines": len(lines),
               "gold_chars": [gold_lo, gold_hi], "path": "scored"}
        region = skip_to = None
        if qi in line_texts:
            text, pieces = prompts[qi]
            chosen = _parse_lines(line_texts[qi], len(lines))
            gold_lines = [i + 1 for i, (a, b) in enumerate(lines)
                          if a < gold_hi and b > gold_lo]
            rec["range"] = chosen
            rec["range_text"] = line_texts[qi]
            rec["gold_range"] = (gold_lines[0], gold_lines[-1]) if gold_lines else None
            if chosen:
                a, b = chosen
                region = (lines[a - 1][0], lines[b - 1][1])
                skip_to = lines[b - 1][0] if b > a else None
                rec["path"] = "located"
                rec["contains"] = bool(gold_lines and a <= gold_lines[0]
                                       and gold_lines[-1] <= b)
                rec["whole_lines_em_f1"] = _f1(context[region[0]:region[1]],
                                               q["answers"])
                ids, doc = scorer._doc_tokens(text + f"{a}-{b}" + BEGINS_CUE, pieces)
            else:
                rec["path"] = "fallback"
        if region is not None and not any(region[0] <= span[0] < region[1]
                                          for _, span in doc):
            region = skip_to = None    # the range maps to no tokens
            rec["path"] = "fallback"
        if region is None:
            ids, doc = scorer.prompt(context, q["question"])
        rec["prompt_tokens"] = len(ids)
        (lp,) = scorer.cues([ids])
        candidates, none_lp, best_lp, rounds = _candidates(
            scorer, ids, lp, ids, doc, region)
        rec["none_wins"] = none_lp > best_lp
        rec["start_rounds"] = rounds
        # a candidate's offset may include the token's leading space
        rec["gold_rank"] = next(
            (i for i, c in enumerate(candidates)
             if c["char"] <= gold_lo and not context[c["char"]:gold_lo].strip()), -1)
        rec["candidates"] = [(c["char"], scorer.vocab[c["first"]],
                              round(c["score"], 2), len(c["filler"]))
                             for c in candidates]
        branches, passes = _spans(scorer, ids, context, candidates, skip_to)
        rec["passes"] = passes
        rec["fed_tokens"] = sum(b["fed"] for b in branches)
        rec["scored"] = {}
        for k in TOP_K:
            answer = _answer(context, branches, k)
            rec["scored"][f"k{k}"] = {"text": answer,
                                      "em_f1": _f1(answer, q["answers"])}
        if rec["gold_rank"] >= 0:
            branch = branches[rec["gold_rank"]]
            others = [max(_scores(b)) for i, b in enumerate(branches)
                      if i != rec["gold_rank"]]
            close = _close_at(branch, max(others) if others else float("-inf"))
            gold_tokens = sum(e <= gold_hi for e in branch["ends"])
            rec["past_gold_end"] = close - gold_tokens
        items.append(rec)
    seconds = time.perf_counter() - t0

    def mean(values):
        values = list(values)
        return statistics.fmean(values) if values else None

    def em_f1(rows, select):
        pairs = [select(r) for r in rows]
        return ({"em": 100 * mean(p[0] for p in pairs),
                 "f1": 100 * mean(p[1] for p in pairs)} if pairs else None)

    by_path = {path: [r for r in items if r["path"] == path]
               for path in ("scored", "located", "fallback")}
    past = [r["past_gold_end"] for r in items if "past_gold_end" in r]
    summary = {
        "run": run, "model": model, "spec": MODELS[model]["spec"],
        "dataset": dataset, "n": len(items), "seconds": seconds, "chunk": CHUNK,
        "max_passes": MAX_PASSES, "top_tokens": TOP_TOKENS,
        "max_starts": MAX_STARTS, "locate_min_lines": LOCATE_MIN_LINES,
        "answer_words_mean": mean(len(q["answers"][0].split()) for q in questions),
        "lines_per_document_mean": mean(r["lines"] for r in items),
        "prompt_tokens_mean": mean(r["prompt_tokens"] for r in items),
        "paths": {path: len(rows) for path, rows in by_path.items()},
        "answer": {f"k{k}": em_f1(items, lambda r, k=k: r["scored"][f"k{k}"]["em_f1"])
                   for k in TOP_K},
        "answer_by_path": {path: em_f1(rows, lambda r: r["scored"]["k8"]["em_f1"])
                           for path, rows in by_path.items()},
        "whole_lines": em_f1(by_path["located"], lambda r: r["whole_lines_em_f1"]),
        "range_contains_gold": mean(r["contains"] for r in by_path["located"]),
        "start_recall": {str(k): mean(0 <= r["gold_rank"] < k for r in items)
                         for k in TOP_K},
        "start_recall_when_contained": mean(
            r["gold_rank"] >= 0 for r in by_path["located"] if r["contains"]),
        "start_rounds_mean": mean(r["start_rounds"] for r in items),
        "none_wins": sum(r["none_wins"] for r in items),
        "past_gold_end": {"n": len(past), "mean": mean(past),
                          "median": statistics.median(past) if past else None,
                          "p95": (sorted(past)[int(0.95 * len(past)) - 1]
                                  if past else None)},
        "passes_mean": mean(r["passes"] for r in items),
        "one_pass_share": mean(r["passes"] == 1 for r in items),
        "passes_hit_cap": sum(r["passes"] == MAX_PASSES for r in items),
        "fed_tokens_mean": mean(r["fed_tokens"] for r in items),
    }
    path = f"/results/extract_spans/{run}_{model}_{dataset}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"summary": summary, "items": items}, f, indent=1)
    VOLUMES["/results"].commit()
    return {"path": path, **summary}


@app.local_entrypoint()
def main(model: str = "qwen3-4b-fp8", dataset: str = "cuad", n: int = N_ITEMS):
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
