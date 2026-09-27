"""Every KV row a forward pass reads holds the tokens its request needs.

A fake model stands in for the network. Each packed row gets a hash of
its token and the hash its attention reads at the row before it, and
every write stores (token, position, hash) in the arena's rows. A
query row that reads a wrong page, a trimmed window, a page reused
while still borrowed, or a wrong position gets a hash no request has,
and the answer readout fails. Filters and joins run over random
corpora whose documents share prefixes, on both attention paths and
on a sliding-window arena. KV_SEEDS=400 runs more corpora per case.
"""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import cpu_staging, fake_pipeline, fake_torch

from quail.backends.quail.executor import loop
from quail.backends.quail.executor.arena import KVArena
from quail.execution.tokens import prefix_tree

torch = pytest.importorskip("torch")

PAGE = 16
WINDOW = 32
MASK = (1 << 61) - 1


def mix(previous, token):
    return (previous * 1_000_003 + token + 1) & MASK


def chain(tokens):
    h = 0
    for token in tokens:
        h = mix(h, token)
    return h


class Pool:
    """(token, position, hash) per physical row of one KV pool."""

    def __init__(self, pages):
        rows = pages * PAGE
        self.token = np.full(rows, -1, dtype=np.int64)
        self.position = np.full(rows, -1, dtype=np.int64)
        self.hash = np.full(rows, -1, dtype=np.int64)


class HashModel:
    """A forward pass whose answers are right only if every KV read is."""

    def __init__(self, arena, answers):
        self.full = Pool(arena.accounting.n_pages)
        self.sliding = (Pool(arena.sliding.n_pages) if arena.has_sliding
                        else None)
        self.answers = answers      # request hash -> answer bit
        self.checked = 0

    def forward_chunk(self, chunk):
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        meta = chunk.meta
        hashes = [None] * len(ids)
        if chunk.attention_mode == "unified":
            self._unified(meta["unified"], ids, positions, hashes)
        else:
            self._tree(meta, ids, positions, hashes)
        out = []
        for row in chunk.final_indices.tolist():
            h = hashes[row]
            assert h in self.answers, (
                f"row {row} at position {positions[row]} read KV no "
                f"request has")
            out.append(self.answers[h])
            self.checked += 1
        return out

    # ---- unified: one paged call per pool --------------------------------

    def _unified(self, unified, ids, positions, hashes):
        cu = unified["cu_q"].tolist()
        full = self._pool_maps(unified)
        sliding = (self._pool_maps(unified["sliding"])
                   if "sliding" in unified else None)

        def value(pool, maps, phys):
            written, copied = maps
            if phys in written:
                return (ids[written[phys]], positions[written[phys]],
                        row_hash(written[phys]))
            if phys in copied:
                return value(pool, maps, copied[phys])
            return (int(pool.token[phys]), int(pool.position[phys]),
                    int(pool.hash[phys]))

        seq_of = {}
        for i in range(len(cu) - 1):
            for row in range(cu[i], cu[i + 1]):
                seq_of[row] = i

        def phys(maps_table, i, view_row):
            pages = maps_table[i]
            return int(pages[view_row // PAGE]) * PAGE + view_row % PAGE

        table = unified["table"].tolist()
        used = unified["used"].tolist()

        def row_hash(row):
            if hashes[row] is not None:
                return hashes[row]
            i = seq_of[row]
            count = cu[i + 1] - cu[i]
            view = used[i] - count + row - cu[i]
            position = positions[row]
            assert view == position, "a query row's KV index is off"
            # a join's frame entry does not keep its canvas row, whose
            # answer nothing reads
            unkept = ids[row] in CANVAS and row == cu[i + 1] - 1
            assert unkept or full[0].get(phys(table, i, view)) == row, \
                "a query row is not written where it reads itself"
            previous = 0
            if view:
                token, pos, previous = value(
                    self.full, full, phys(table, i, view - 1))
                assert pos == position - 1, \
                    f"row at {position} reads position {pos} before it"
            hashes[row] = mix(previous, ids[row])
            return hashes[row]

        for row in range(len(ids)):
            row_hash(row)
        # the resident rows each sequence reads form one chain
        for i in range(len(cu) - 1):
            count = cu[i + 1] - cu[i]
            self._check_chain(lambda v, i=i: value(
                self.full, full, phys(table, i, v)), used[i] - count)
        if sliding is not None:
            s_table = unified["sliding"]["table"].tolist()
            s_used = unified["sliding"]["used"].tolist()
            for row in range(len(ids)):
                i = seq_of[row]
                count = cu[i + 1] - cu[i]
                view = s_used[i] - count + row - cu[i]
                origin = positions[row] - view
                for k in range(max(0, positions[row] - WINDOW + 1),
                               positions[row] + 1):
                    assert k >= origin, (
                        f"row at {positions[row]} needs window row {k}; "
                        f"the sliding pages start at {origin}")
                    got = value(self.sliding, sliding,
                                phys(s_table, i, k - origin))
                    want = value(self.full, full, phys(table, i, k))
                    assert got == want, \
                        f"sliding row {k} differs from the every-token row"
        self._commit(self.full, full, ids, positions, hashes)
        if sliding is not None:
            self._commit(self.sliding, sliding, ids, positions, hashes)

    @staticmethod
    def _pool_maps(pool):
        written = dict(zip(pool["dst"].tolist(), pool["src"].tolist()))
        copied = {}
        if pool["tail_src"] is not None:
            copied = dict(zip(pool["tail_dst"].tolist(),
                              pool["tail_src"].tolist()))
        return written, copied

    def _commit(self, pool, maps, ids, positions, hashes):
        written, copied = maps
        # the scatter lands before the tail copies read the pages
        for dst, src in written.items():
            pool.token[dst] = ids[src]
            pool.position[dst] = positions[src]
            pool.hash[dst] = hashes[src]
        for dst, src in copied.items():
            pool.token[dst] = pool.token[src]
            pool.position[dst] = pool.position[src]
            pool.hash[dst] = pool.hash[src]

    # ---- tree: causal segments plus paged reads ----------------------

    def _tree(self, meta, ids, positions, hashes):
        cu_a = meta["cu_a"].tolist()
        first_of_segment = set(cu_a[:-1])
        written = {}
        if meta["kv_src"] is not None:
            written = dict(zip(meta["kv_dst"].tolist(),
                               meta["kv_src"].tolist()))
        reads = meta["reads"]
        read_of = {}
        read_table, read_used = [], []
        if reads is not None:
            cu_q = reads["cu_q"].tolist()
            rows = reads["rows"].tolist()
            read_table = reads["table"].tolist()
            read_used = reads["used"].tolist()
            for i in range(len(cu_q) - 1):
                for row in rows[cu_q[i]:cu_q[i + 1]]:
                    read_of[row] = i

        def value(phys):
            if phys in written:
                src = written[phys]
                return ids[src], positions[src], row_hash(src)
            return (int(self.full.token[phys]),
                    int(self.full.position[phys]),
                    int(self.full.hash[phys]))

        def phys(i, view_row):
            return int(read_table[i][view_row // PAGE]) * PAGE + view_row % PAGE

        def row_hash(row):
            if hashes[row] is not None:
                return hashes[row]
            position = positions[row]
            if row not in first_of_segment:
                assert positions[row - 1] == position - 1, \
                    "a causal segment skips a position"
                previous = row_hash(row - 1)
            elif row in read_of:
                i = read_of[row]
                assert read_used[i] == position, \
                    f"row at {position} reads {read_used[i]} resident rows"
                _, pos, previous = value(phys(i, position - 1))
                assert pos == position - 1
            else:
                assert position == 0, \
                    f"row at {position} starts a segment and reads nothing"
                previous = 0
            hashes[row] = mix(previous, ids[row])
            return hashes[row]

        for row in range(len(ids)):
            row_hash(row)
        for i in range(len(read_used)):
            self._check_chain(lambda v, i=i: value(phys(i, v)), read_used[i])
        for dst, src in written.items():
            self.full.token[dst] = ids[src]
            self.full.position[dst] = positions[src]
            self.full.hash[dst] = hashes[src]

    @staticmethod
    def _check_chain(read, rows):
        previous = 0
        for v in range(rows):
            token, pos, h = read(v)
            assert pos == v, f"resident row {v} holds position {pos}"
            assert h == mix(previous, token), \
                f"resident row {v} was overwritten"
            previous = h


def corpus(rng, n):
    """Documents that repeat one another's starts, as agent snapshots do."""
    header = [rng.randrange(50) for _ in range(rng.choice([0, 5, 16, 20]))]
    docs = []
    while len(docs) < n:
        trace = list(header)
        for _ in range(rng.randrange(1, 5)):
            trace = trace + [rng.randrange(50)
                             for _ in range(rng.choice([1, 7, 16, 30, 45]))]
            docs.append(trace)
            if rng.random() < 0.2:
                docs.append(list(trace))          # an exact duplicate
            if rng.random() < 0.3 and len(trace) > 20:
                cut = rng.randrange(1, len(trace))
                docs.append(trace[:cut] + [50 + rng.randrange(20)])
    docs = docs[:n]
    rng.shuffle(docs)
    return docs


def make_arena(sliding, pages, sliding_pages):
    if sliding:
        return KVArena(n_layers=2, n_pages=pages, page_tokens=PAGE, n_kv=1,
                       d_head=1, dtype=torch.float32, device="cpu",
                       layer_kv=[(1, 1), (1, 1)], sliding_layers=(1,),
                       sliding_window=WINDOW, n_sliding_pages=sliding_pages)
    return KVArena(n_layers=1, n_pages=pages, page_tokens=PAGE, n_kv=1,
                   d_head=1, dtype=torch.float32, device="cpu")


def cpu_torch():
    fake = fake_torch()
    real = SimpleNamespace(**{n: getattr(torch, n) for n in dir(torch)
                              if not n.startswith("_")})
    real.cuda, real.inference_mode = fake.cuda, fake.inference_mode
    return real


IDENTITY = SimpleNamespace(submit=lambda v: v, result=lambda v: v, dtype=None)

# a diffusion model's one canvas row follows every suffix and answers
CANVAS = (99,)


def pipeline_for(model, canvas):
    return fake_pipeline(forward_chunk=model.forward_chunk,
                         tree_attention=True,
                         canvas_ids=CANVAS if canvas else ())

SEEDS = int(os.environ.get("KV_SEEDS", 6))
CASES = [(seed, path, sliding, canvas)
         for seed in range(SEEDS)
         for path, sliding, canvas in (("unified", False, False),
                                       ("tree", False, False),
                                       ("unified", True, False),
                                       ("unified", True, True))]


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_filter_reads_the_right_kv(monkeypatch, seed, path, sliding, canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(seed)
    docs = corpus(rng, 40)
    questions = [[90, 91, 92, 93], [90, 91, 94]][:rng.choice([1, 2])]
    # the answer depends on the tokens, so duplicates agree
    tail = list(CANVAS) if canvas else []
    answers = {}
    for doc in docs:
        for q in questions:
            answers.setdefault(chain(doc + q + tail), int(rng.random() < 0.7))
    truth = {(d, j): answers[chain(docs[d] + questions[j] + tail)]
             for d in range(len(docs)) for j in range(len(questions))}
    arena = make_arena(sliding, pages=rng.choice([40, 80, 400]),
                       sliding_pages=rng.choice([24, 60]))
    model = HashModel(arena, answers)
    pipeline = pipeline_for(model, canvas)
    tree = prefix_tree(docs, PAGE)
    got, _, _ = loop.run_filter(
        cpu_torch(), arena, pipeline, IDENTITY, docs, questions,
        max(map(len, docs)) + rng.choice([8, 60, 600]), arena_writes=True,
        arena_keys=[("d", i) for i in range(len(docs))],
        attention_mode=path, prefix_tree=tree)
    for d in range(len(docs)):
        want = []
        for j in range(len(questions)):
            want.append(int(truth[d, j]))
            if not truth[d, j]:
                break
        assert got[d] == want
    assert model.checked
    assert not arena.accounting.owned


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_join_reads_the_right_kv(monkeypatch, seed, path, sliding, canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(100 + seed)
    anchors = corpus(rng, 30)
    frame = [95, 96]
    partners = [[97, 60 + p, 98] for p in range(3)]
    answers, truth = join_answers(rng, anchors, frame, partners, canvas)
    arena = make_arena(sliding, pages=rng.choice([60, 400]),
                       sliding_pages=rng.choice([40, 80]))
    model = HashModel(arena, answers)
    pipeline = pipeline_for(model, canvas)
    out, _, _ = loop.run_join(
        cpu_torch(), arena, pipeline, IDENTITY, anchors, [partners],
        max(map(len, anchors)) + rng.choice([8, 60, 600]),
        stage_frames=[frame],
        anchor_keys=[("a", i) for i in range(len(anchors))],
        attention_mode=path, prefix_tree=prefix_tree(anchors, PAGE))
    for a in range(len(anchors)):
        assert out[0][a] == [int(truth[a, p]) for p in range(len(partners))]
    assert model.checked
    assert not arena.accounting.owned


def join_answers(rng, anchors, frame, partners, canvas):
    """Answers per request hash, and per (anchor, partner) the truth."""
    tail = list(CANVAS) if canvas else []
    # a frame entry's last row reads the anchor and its frame
    answers = {}
    for anchor in anchors:
        answers[chain(anchor + frame)] = 0
        answers[chain(anchor + frame + tail)] = 0
    for anchor in anchors:
        for partner in partners:
            answers.setdefault(chain(anchor + frame + partner + tail),
                               int(rng.random() < 0.5))
    truth = {(a, p): answers[chain(anchors[a] + frame + partners[p] + tail)]
             for a in range(len(anchors)) for p in range(len(partners))}
    return answers, truth


@pytest.mark.parametrize("seed,path,sliding,canvas", CASES)
def test_filter_survivors_feed_a_join(monkeypatch, seed, path, sliding,
                                      canvas):
    cpu_staging(monkeypatch)
    rng = random.Random(200 + seed)
    docs = corpus(rng, 40)
    question = [90, 91, 92]
    frame = [95, 96]
    partners = [[97, 60 + p, 98] for p in range(3)]
    tail = list(CANVAS) if canvas else []
    answers, truth = join_answers(rng, docs, frame, partners, canvas)
    passes = {}
    for doc in docs:
        h = chain(doc + question + tail)
        passes.setdefault(h, int(rng.random() < 0.6))
        answers.setdefault(h, passes[h])
    arena = make_arena(sliding, pages=rng.choice([80, 400]),
                       sliding_pages=rng.choice([60, 120]))
    model = HashModel(arena, answers)
    pipeline = pipeline_for(model, canvas)
    budget = max(map(len, docs)) + rng.choice([8, 60, 600])
    cpu = cpu_torch()
    source = loop.FilterStream(
        cpu, arena, pipeline, IDENTITY, docs, [question], budget,
        arena_writes=True, arena_keys=[("d", i) for i in range(len(docs))],
        hold_survivors=True, hold_extra_tokens=len(frame) + len(tail),
        prefix_tree=prefix_tree(docs, PAGE), attention_mode=path)
    keys = []
    out, _, _ = loop.run_join(
        cpu, arena, pipeline, IDENTITY, [], [partners], budget,
        stage_frames=[frame], anchor_keys=keys, anchor_source=source,
        attention_mode=path)
    survivors = {d for d in range(len(docs))
                 if passes[chain(docs[d] + question + tail)]}
    assert {key[1] for key in keys} == survivors
    for a, key in enumerate(keys):
        assert out[0][a] == [truth[key[1], p] for p in range(len(partners))]
    assert not arena.accounting.owned
