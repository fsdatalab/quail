"""Cell definitions for the calibration sweeps.

A cell is one engine step with a chosen composition: N requests, each
computing `new` fresh tokens against `cached` already-resident tokens.
The families come from the cost-model calibration design:

    c1     prefill composition: N requests of c fresh tokens, no cache
    alpha  one request of h fresh tokens, h swept to 16,384
    c2     c-token suffix over an h-token cached document, N at once
    c4     mixed steps: fresh prefills and cached suffixes together
    c5     same totals, different per-request cached-length spread

This module is pure: no vLLM, no numpy, no tokenizer. The runner
(experiments/modal_calibrate.py) turns cells into real requests; the
fits (quail/plan/fit.py) turn measured rows back into models. Both
depend on the arithmetic here, so it is unit-tested without an engine
(tests/test_calib.py).

Every sequence length is a multiple of the 16-token KV block, so a
cached prefix always matches in whole blocks and the measured step
computes exactly the requested fresh tokens.
"""

BLOCK = 16
NONCE_TOKENS = 16

# The single boot every family runs against. One configuration, so no
# cell's result depends on which boot served it.
BOOT = dict(
    max_model_len=16_512,          # largest document 16,384 + suffix + margin
    max_num_batched_tokens=32_768,  # largest single cell (c1: 512 x 64)
    max_num_seqs=512,
    gpu_memory_utilization=0.90,
    cudagraph_capture=8_192,       # min(max_num_batched_tokens, 8192)
)

# Residency cap for cached cells: warm KV must stay comfortably under
# the pool so the recency rule never evicts what a round is about to
# read.
RESIDENT_FRACTION = 0.75


# ---- unique content ---------------------------------------------------

def make_nonce_ids(counter, alphabet):
    """A 16-token block encoding `counter` in base len(alphabet).

    vLLM's prefix cache hashes blocks from position 0, so a sequence
    that starts with a globally unique block shares no cache entry
    with any other sequence. `alphabet` is a list of token ids the
    runner picks once from the tokenizer; two ids suffice, more make
    the block look like text."""
    base = len(alphabet)
    if base < 2:
        raise ValueError("nonce alphabet needs at least 2 token ids")
    if counter < 0 or counter >= base ** NONCE_TOKENS:
        raise ValueError("nonce counter out of range")
    digits = []
    for _ in range(NONCE_TOKENS):
        digits.append(alphabet[counter % base])
        counter //= base
    return digits


def build_sequence(pool_ids, cursor, n_tokens, nonce_ids):
    """Exactly `n_tokens` token ids: the nonce block, then pool
    content from `cursor`, wrapping at the pool's end. Returns
    (ids, new_cursor). The pool is smaller than the total demand, so
    the cursor wraps; the nonce keeps wrapped content unique."""
    if n_tokens % BLOCK:
        raise ValueError(f"sequence length {n_tokens} is not a "
                         f"multiple of the {BLOCK}-token block")
    if len(nonce_ids) != NONCE_TOKENS:
        raise ValueError("nonce must be exactly one block")
    body = n_tokens - NONCE_TOKENS
    ids = list(nonce_ids)
    n = len(pool_ids)
    while body > 0:
        take = min(body, n - cursor)
        ids.extend(pool_ids[cursor:cursor + take])
        cursor = (cursor + take) % n
        body -= take
    return ids, cursor


# ---- step arithmetic --------------------------------------------------

def pair_count(requests):
    """P: attention pairs of a step. Each request's new tokens attend
    to its cached tokens and, causally, to each other."""
    return sum(r["new"] * r["cached"] + r["new"] * (r["new"] + 1) // 2
               for r in requests)


def batch_tokens(requests):
    """B: fresh tokens computed in the step."""
    return sum(r["new"] for r in requests)


def resident_tokens(requests):
    """KV tokens the step needs resident by its end: cached context
    plus the fresh tokens it writes."""
    return sum(r["cached"] + r["new"] for r in requests)


def new_blocks(requests):
    """A: KV blocks the step allocates. Cached lengths are whole
    blocks, so each request allocates exactly its fresh span."""
    return sum(-(-(r["cached"] + r["new"]) // BLOCK) - r["cached"] // BLOCK
               for r in requests)


def resident_blocks(requests):
    """R: KV blocks the step's requests hold by its end - the whole
    context in blocks, cached plus fresh. new_blocks counts only the
    fresh span; the scheduler's per-step block tables span all of R."""
    return sum(-(-(r["cached"] + r["new"]) // BLOCK) for r in requests)


def _cell(family, name, requests, warm=None):
    return dict(
        family=family,
        name=name,
        requests=requests,
        warm=warm or [],
        n=len(requests),
        b=batch_tokens(requests),
        p=pair_count(requests),
        a=new_blocks(requests),
        resident=resident_tokens(requests),
    )


# ---- the families -----------------------------------------------------

def family_c1():
    """Prefill composition. B = N*c caps at the boot budget 32,768 -
    nothing past it has ever been measured in this repo."""
    cells = []
    for c, n_max in ((64, 512), (256, 128), (512, 64)):
        n = 1
        while n <= n_max:
            cells.append(_cell(
                "c1", f"c1_c{c}_n{n}",
                [dict(new=c, cached=0) for _ in range(n)]))
            n *= 2
    return cells


ALPHA_LENGTHS = (512, 1024, 2048, 3072, 4096, 6144, 8192,
                 10240, 12288, 14336, 16384)
DRIFT_LENGTH = 4096


def family_alpha():
    """One request of h fresh tokens. Fits T_pre(h) = a1*h + a2*h*h;
    the h*h term is the attention work."""
    return [_cell("alpha", f"alpha_h{h}", [dict(new=h, cached=0)])
            for h in ALPHA_LENGTHS]


def drift_cell(tag):
    """The repeated reference cell (alpha at h=4,096) that runs at the
    start and end of the sweep; its spread is the drift number."""
    return _cell("drift", f"drift_{tag}",
                 [dict(new=DRIFT_LENGTH, cached=0)])


def family_c2():
    """Suffix over cached context: c fresh tokens against an h-token
    cached document, N documents at once. The warm list is the
    documents to prefill before the measured rounds."""
    cells = []
    for c in (16, 32, 64):
        for h in (2048, 4096, 8192, 16384):
            for n in (1, 4, 16, 32):
                cells.append(_cell(
                    "c2", f"c2_c{c}_h{h}_n{n}",
                    [dict(new=c, cached=h) for _ in range(n)],
                    warm=[h] * n))
            if h <= 4096:
                cells.append(_cell(
                    "c2", f"c2_c{c}_h{h}_n64",
                    [dict(new=c, cached=h) for _ in range(64)],
                    warm=[h] * 64))
    return cells


def family_c4():
    """Mixed steps: fresh 512-token prefills plus cached c=32 suffixes
    over 8,192-token documents, in one step. Tests that step cost adds
    across requests the way the model assumes."""
    cells = []
    for n_pre, n_cache in ((1, 8), (1, 16), (1, 32),
                           (2, 16), (2, 32), (4, 32)):
        reqs = ([dict(new=512, cached=0)] * n_pre
                + [dict(new=32, cached=8192)] * n_cache)
        cells.append(_cell(
            "c4", f"c4_pre{n_pre}_cache{n_cache}", reqs,
            warm=[8192] * n_cache))
    return cells


C5_SPREADS = dict(
    uniform=[8192] * 16,
    mild=[4096] * 8 + [12288] * 8,
    extreme=[512] * 8 + [15872] * 8,
    onehot=[7648] * 15 + [16352],
)


def family_c5():
    """Same B, P, N in every cell; only the per-request cached-length
    spread changes. Any time difference is the imbalance effect the
    additive model cannot see."""
    cells = []
    for name, lengths in C5_SPREADS.items():
        assert sum(lengths) == 131_072, name
        assert all(h % BLOCK == 0 for h in lengths), name
        cells.append(_cell(
            "c5", f"c5_{name}",
            [dict(new=32, cached=h) for h in lengths],
            warm=list(lengths)))
    return cells


FAMILIES = dict(c1=family_c1, alpha=family_alpha, c2=family_c2,
                c4=family_c4, c5=family_c5)


def cells_for(families):
    """The cells for a comma-separated family list, in run order.
    "all" is every engine family (c6 has no cells; the runner handles
    it separately)."""
    order = ("alpha", "c1", "c2", "c4", "c5")
    wanted = set(order) if families == "all" else {
        f.strip() for f in families.split(",") if f.strip() != "c6"}
    unknown = wanted - set(order)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    out = []
    for fam in order:
        if fam in wanted:
            out.extend(FAMILIES[fam]())
    return out


def feasible(cells, pool_tokens, boot=BOOT):
    """Drop cells the boot cannot run as one step: over the token
    budget, over the sequence cap, or holding more KV than the
    residency cap allows. Returns (kept, dropped_names)."""
    kept, dropped = [], []
    for cell in cells:
        warm_tokens = sum(cell["warm"])
        ok = (cell["b"] <= boot["max_num_batched_tokens"]
              and cell["n"] <= boot["max_num_seqs"]
              and warm_tokens + cell["b"]
              <= RESIDENT_FRACTION * pool_tokens)
        (kept if ok else dropped).append(cell if ok else cell["name"])
    return kept, dropped


# ---- verification -----------------------------------------------------

def check_step(cell, records):
    """Verify the measured rounds ran as exactly one step of the
    requested composition. `records` are the step-trace dicts the
    round produced. Returns (ok, reason)."""
    work = [r for r in records if r.get("tokens", 0) > 0]
    if len(work) != 1:
        return False, f"{len(work)} work steps, expected 1"
    rec = work[0]
    if rec["seqs"] != cell["n"]:
        return False, f"step held {rec['seqs']} requests, cell has {cell['n']}"
    if rec["tokens"] != cell["b"]:
        return False, (f"step computed {rec['tokens']} tokens, "
                       f"cell requests {cell['b']}")
    shapes = rec.get("shapes")
    if shapes is not None:
        want = sorted((r["new"], r["cached"]) for r in cell["requests"])
        got = sorted((int(s[0]), int(s[1])) for s in shapes)
        if want != got:
            return False, f"per-request shapes {got} != requested {want}"
    return True, ""


def check_cached(cell, cached_counts):
    """Verify the engine reported the cached-token count each request
    was designed to have: h for cached requests, 0 for fresh ones.
    `cached_counts` come from the client's request outputs, in the
    submission order of cell["requests"]."""
    want = [r["cached"] for r in cell["requests"]]
    if len(cached_counts) != len(want):
        return False, (f"{len(cached_counts)} outputs for "
                       f"{len(want)} requests")
    for i, (got, expect) in enumerate(zip(cached_counts, want)):
        if got != expect:
            return False, (f"request {i}: {got} cached tokens, "
                           f"designed for {expect}")
    return True, ""
