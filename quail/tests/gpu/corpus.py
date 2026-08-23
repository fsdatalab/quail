"""The milestone 1 corpora.

Two workloads ported from the exploration (same seeds, same
truncation). 2026-08-21: the planted values and answer instructions
changed from YES/NO to TRUE/FALSE to match the engine's constrained
readout (true_false_ids), so answer counts no longer compare against
pre-change artifacts; walls still do (token counts move by a few
tokens per document):

- the 10,000-document five-filter IMDB corpus with planted [FLAGS]
  lines (filter_cells.json / filter_cells_bf16.json);
- the BioDEX 100-report x 2,560-term join sample (join2way.json).
"""

MODEL = "Qwen/Qwen3-4B-FP8"
WORKLOAD_SEED = 20260731
FLAG_SEED = 424242
DATA_SEED = 20260817
N_FILTERS = 5
SELECTIVITY = (0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8)

PREAMBLE = ("You will be shown a patient report and one candidate "
            "medical reaction term. Decide whether the report "
            "describes that reaction as something the patient "
            "experienced.\n\nREPORT:\n")


# ------------------------------------------------------ the filter set

def build_pool(n_docs):
    """The seeded 10,000-document IMDB sample, truncated to n_docs.
    Fixed by WORKLOAD_SEED, so every run sees the same documents in
    the same order."""
    import numpy as np
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = []
    for split in ("train", "test"):
        path = hf_hub_download(
            "stanfordnlp/imdb",
            f"plain_text/{split}-00000-of-00001.parquet",
            repo_type="dataset")
        frames.append(pd.read_parquet(path)["text"])
    pool = list(frames[0]) + list(frames[1])
    rng = np.random.default_rng(WORKLOAD_SEED)
    idx = sorted(rng.choice(len(pool), size=10_000, replace=False))
    return [pool[i] for i in idx[:n_docs]]


def flags_line(flags):
    return "\n\n[FLAGS] " + " ".join(
        f"FLAG_{j+1}={'TRUE' if f else 'FALSE'}"
        for j, f in enumerate(flags))


def question(j):
    """Filter j's question. Every question shares a 33-token preamble;
    the executor keeps its KV with the document after stage 1, the way
    chain mode's rewind kept it resident."""
    return (f"\n\nExample: if the line said [FLAGS] FLAG_9=FALSE, then "
            f"FLAG_9 has value FALSE.\nInstruction: output only the value "
            f"of FLAG_{j} from the [FLAGS] line above.\nFLAG_{j}=")


def build_corpus(tok, n_docs, seed_offset=100, n_filters=None):
    """Tokenized documents with their planted flag lines, the
    tokenized questions, and the flag truth table. The default keeps
    every 5-filter corpus byte-identical to the committed cells.

    Returns (body_ids, q_ids, flags)."""
    import numpy as np

    n = n_filters or N_FILTERS
    rng = np.random.default_rng(FLAG_SEED + seed_offset)
    flags = (rng.random((n_docs, n))
             < np.array(SELECTIVITY[:n])[None, :]).astype(int)
    docs = build_pool(n_docs)
    bodies = [d + flags_line(f) for d, f in zip(docs, flags)]
    body_ids = tok(bodies, add_special_tokens=False)["input_ids"]
    q_ids = [tok(question(j + 1), add_special_tokens=False)["input_ids"]
             for j in range(n)]
    return body_ids, q_ids, flags


# -------------------------------------------------------- the join set

NWAY_PREAMBLE = ("You will be shown a report document and a candidate "
                 "document, each carrying a planted key line. Answer "
                 "from those lines only.\n\nREPORT DOCUMENT:\n")
N_A = 100
N_B = 100
N_C = 100
X_VALUES = 50          # a_j carries X = j % 50
B_GATED = 20           # b_i for i >= 80 gets an X no A document has
Y_VALUES = 25          # c_j carries Y = j % 25


def nway_truth():
    """The planted ground truth, from the key assignment alone."""
    ans1 = {b: [1 if (b < N_B - B_GATED and a % X_VALUES == b % X_VALUES)
                else 0 for a in range(N_A)] for b in range(N_B)}
    ans2_full = {b: [1 if c % Y_VALUES == b % Y_VALUES else 0
                     for c in range(N_C)] for b in range(N_B)}
    return ans1, ans2_full


def nway_corpus(tokenizer):
    """Three planted collections from the IMDB pool: B documents are
    ~4k tokens (12 reviews concatenated), A and C single reviews.
    Byte-identical to the committed nway3 corpus."""
    reviews = build_pool(1500)
    a_docs = [f"{reviews[j]}\n\n[KEY] X={j % X_VALUES}"
              for j in range(N_A)]
    c_docs = [f"{reviews[N_A + j]}\n\n[KEY] Y={j % Y_VALUES}"
              for j in range(N_C)]
    b_docs = []
    base = N_A + N_C
    for i in range(N_B):
        body = "\n\n".join(reviews[base + i * 12: base + (i + 1) * 12])
        x = (i % X_VALUES) if i < N_B - B_GATED else 1000 + i
        b_docs.append(f"{body}\n\n[KEYS] X={x} Y={i % Y_VALUES}")

    pre = tokenizer(NWAY_PREAMBLE, add_special_tokens=False)["input_ids"]
    b_prefix = [pre + tokenizer(d, add_special_tokens=False)["input_ids"]
                for d in b_docs]

    def suffix(doc, key):
        return tokenizer(
            f"\n\nCANDIDATE DOCUMENT:\n{doc}\n"
            f"Instruction: answer TRUE if the [KEY] {key} value in the "
            f"candidate equals the [KEYS] {key} value in the report "
            f"document, FALSE otherwise.\nANSWER=",
            add_special_tokens=False)["input_ids"]

    a_suffix = [suffix(d, "X") for d in a_docs]
    c_suffix = [suffix(d, "Y") for d in c_docs]
    return b_prefix, a_suffix, c_suffix


def pair_suffix_text(term):
    return (f"\n\nCANDIDATE REACTION: {term}\n"
            f"Instruction: answer TRUE if the report above describes "
            f"this reaction, FALSE otherwise.\nANSWER=")


def biodex_sample(tokenizer, n_reports=100, vocab_cap=3718,
                  max_report_tokens=3500, seed=DATA_SEED):
    """Reports and the reaction-term vocabulary, tokenized. Streamed:
    the pool is the first 2,000 usable rows in dataset order, the
    reports a seeded choice from it. The committed run's pool yields
    2,560 distinct terms under the cap, so pairs = 100 x 2,560."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("BioDEX/BioDEX-Reactions", split="train",
                      streaming=True)

    def reactions_of(row):
        return [t.strip() for t in str(row.get("reactions", "")).split(",")
                if t.strip()]

    freq = {}
    rows = []
    for row in ds:
        terms = reactions_of(row)
        text = str(row.get("fulltext_processed") or row.get("abstract"))
        if not terms or len(text) < 200:
            continue
        rows.append((text, terms))
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
        if len(rows) >= 2000:
            break
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    vocab = [t for t, _ in sorted(freq.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    vocab = vocab[:vocab_cap]
    vset = set(vocab)
    keep = [(text, [t for t in terms if t in vset])
            for text, terms in rows]
    keep = [r for r in keep if r[1]][:n_reports]
    if len(keep) < n_reports:
        raise RuntimeError(f"only {len(keep)} usable reports")

    pre_ids = tokenizer(PREAMBLE, add_special_tokens=False)["input_ids"]
    prefixes, gold = [], []
    for text, terms in keep:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        prefixes.append(pre_ids + ids[:max_report_tokens])
        gold.append(terms)
    suffixes = [tokenizer(pair_suffix_text(t),
                          add_special_tokens=False)["input_ids"]
                for t in vocab]
    return dict(prefixes=prefixes, suffixes=suffixes, vocab=vocab,
                gold=gold, preamble_tokens=len(pre_ids),
                max_report_tokens=max_report_tokens)
