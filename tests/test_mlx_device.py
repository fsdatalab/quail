"""The MLX device implementation against mlx-lm's Qwen3 on a tiny model.

The shared packer and stage scheduler drive the MLX forward pass over
real KV pages. Each answer is scored from its final hidden state, and
the score must equal the one mlx-lm gives the request's whole token
sequence: a row that reads a wrong page, a page reused too early, or a
wrong rotary position changes it. In float32 the two agree to rounding;
in bf16, the dtype the Decision model runs in, to bf16 rounding.

Runs on Apple silicon with mlx, vllm-metal, and mlx-lm installed;
skipped elsewhere.
"""

import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.qwen3 import Model, ModelArgs  # noqa: E402
from test_kv_reads import corpus  # noqa: E402

from quail.backends.quail.executor import chunk as chunk_mod  # noqa: E402
from quail.backends.quail.executor import loop  # noqa: E402
from quail.backends.quail.executor.arena import KVArena  # noqa: E402
from quail.backends.quail.executor.mlx_device import kernels  # noqa: E402
from quail.backends.quail.executor.mlx_device.implementation import (  # noqa: E402
    MlxImplementation,
)
from quail.backends.quail.executor.mlx_device.loader import (  # noqa: E402
    load_decision_head,
    load_qwen3_weights,
)
from quail.backends.quail.executor.mlx_device.pools import MlxKVPools  # noqa: E402
from quail.backends.quail.executor.mlx_device.readout import (  # noqa: E402
    MlxDecisionChoices,
    MlxDecisionHead,
    MlxDecisionRows,
    MlxDecisions,
    MlxDecisionScores,
)
from quail.backends.quail.executor.mlx_device.weights import (  # noqa: E402
    Qwen3Config,
    Qwen3Weights,
)
from quail.backends.quail.executor.models import build_pipeline  # noqa: E402
from quail.backends.quail.executor.stages import Stage, run_stages  # noqa: E402
from quail.execution.tokens import prefix_tree  # noqa: E402

PAGE = 16
# the smallest head the paged kernels are built for is 64 wide
CONFIG = dict(
    model_type="qwen3", hidden_size=64, num_hidden_layers=2,
    intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=64, rms_norm_eps=1e-6, vocab_size=128,
    max_position_embeddings=32768, rope_theta=1000000.0,
    tie_word_embeddings=True)
DTYPES = {"float32": mx.float32, "bf16": mx.bfloat16}
# score differences a correct run may show against mlx-lm in float32
TOLERANCE = {"float32": 1e-4, "bf16": 0.1}


@pytest.fixture(params=sorted(DTYPES))
def tiny(request):
    """A tiny model in one dtype."""
    return Tiny(request.param)


def reference_model():
    mx.random.seed(7)
    model = Model(ModelArgs(**CONFIG))
    mx.eval(model.parameters())
    return model


class Probe:
    """Score each answer row with a fixed vector.

    The score depends on every KV row the answer read, so the scheduler
    returns a number to compare with mlx-lm's.
    """

    dtype = np.float32

    def __init__(self, hidden):
        rng = np.random.default_rng(3)
        vector = rng.standard_normal(hidden).astype(np.float32)
        self.vector = mx.array(vector / np.linalg.norm(vector))

    def submit(self, normed, rows_per_answer=None):
        scores = normed.astype(mx.float32) @ self.vector
        mx.async_eval(scores)
        return scores

    @staticmethod
    def result(handle):
        return np.array(handle)


class Tiny:
    """The tiny model as mlx-lm holds it and as the MLX pipeline runs it."""

    def __init__(self, dtype_name):
        self.dtype_name = dtype_name
        self.dtype = DTYPES[dtype_name]
        self.tolerance = TOLERANCE[dtype_name]
        self.model = reference_model()
        self.config = Qwen3Config.from_dict(CONFIG)
        self.weights = Qwen3Weights.from_tensors(
            dict(tree_flatten(self.model.parameters())), self.config,
            self.dtype, prefix="model.")
        self.probe = Probe(self.config.hidden)
        self.implementation = MlxImplementation()
        self.spec = SimpleNamespace(arch="qwen3")
        self._scores = {}

    def run(self, pages):
        """Return a fresh arena and the forward pass over it."""
        config = self.config
        arena = KVArena(
            n_layers=config.layers, n_pages=pages, page_tokens=PAGE,
            n_kv=config.n_kv, d_head=config.head_dim,
            pools=self.implementation.kv_pools(self.dtype))
        return arena, build_pipeline(self.spec, self.weights, arena,
                                     implementation="mlx")

    def hidden(self, tokens):
        """mlx-lm's final hidden states of one whole sequence, in float32."""
        return self.model.model(mx.array([list(tokens)]))[0]

    def score(self, tokens) -> float:
        """The probe's score of the sequence's last row."""
        key = tuple(tokens)
        if key not in self._scores:
            self._scores[key] = float(self.hidden(tokens)[-1] @ self.probe.vector)
        return self._scores[key]

    def close(self, got, tokens, what):
        want = self.score(tokens)
        assert abs(float(got) - want) < self.tolerance, (
            f"{what}: scored {float(got):.5f}, mlx-lm scores {want:.5f}")


def filter_stages(tiny, frame, tails, gated=True):
    """Filter stages; a gated document goes on while its score is positive."""
    def decide(a, row):
        return not gated or bool(row[0] > 0)

    return [Stage(suffixes=[tail], readout=tiny.probe, frame=frame, single=True,
                  decide=decide) for tail in tails]


def check_filter(tiny, docs, frame, tails, pages, budget, gated=True):
    """Run a filter chain and check every answer against mlx-lm.

    Returns:
        The fresh tokens the run computed, the tokens of the whole
        sequences it answered, the tokens read from a borrowed prefix,
        and the documents a gate dropped.
    """
    arena, pipeline = tiny.run(pages)
    stats = {}
    answers, _, tokens = run_stages(
        tiny.implementation, arena, pipeline, filter_stages(tiny, frame, tails, gated),
        docs, budget, anchor_keys=[("d", i) for i in range(len(docs))],
        prefix_tree=prefix_tree(docs, PAGE), attention_mode="unified",
        default_attention="unified", stats=stats)
    asked = dropped = 0
    for d, doc in enumerate(docs):
        for stage, tail in enumerate(tails):
            assert d in answers[stage], f"document {d} stage {stage}"
            sequence = doc + frame + tail
            tiny.close(answers[stage][d][0], sequence, f"document {d} stage {stage}")
            asked += len(sequence)
            if gated and tiny.score(sequence) <= 0:
                assert all(d not in later for later in answers[stage + 1:])
                dropped += stage < len(tails) - 1
                break
    assert not arena.accounting.owned
    return tokens, asked, stats["borrowed_tokens"], dropped


def test_filter_chains_match_whole_sequences(tiny):
    """Documents that share starts, three gated stages, arenas they fill."""
    frame, tails = [90, 91, 92], [[93, 94], [95], [96, 97, 98]]
    for seed, pages, extra in ((0, 40, 8), (1, 80, 60), (2, 400, 600)):
        rng = random.Random(seed)
        docs = corpus(rng, 40, (0, 5, 16, 20), (1, 7, 16, 30, 45))
        budget = max(map(len, docs)) + len(frame) + 3 + extra
        tokens, asked, borrowed, dropped = check_filter(
            tiny, docs, frame, tails, pages, budget)
        # retained and borrowed KV is not computed again
        assert tokens < asked
        assert borrowed and 0 < dropped < len(docs)
    # the smallest arena holds fewer tokens than its run computed
    assert 40 * PAGE < 1000


def test_a_read_of_the_wrong_pages_changes_the_score(tiny, monkeypatch):
    """The comparison with mlx-lm fails when two context pages trade places."""
    attention = kernels.paged_attention

    def swapped(q, k_pool, v_pool, *, table, **reads):
        if table.shape[1] > 1:
            table = mx.concatenate([table[:, 1:2], table[:, 0:1], table[:, 2:]], 1)
        return attention(q, k_pool, v_pool, table=table, **reads)

    monkeypatch.setattr(
        "quail.backends.quail.executor.models.qwen3_mlx.paged_attention", swapped)
    rng = random.Random(0)
    docs = corpus(rng, 40, (0, 5, 16, 20), (1, 7, 16, 30, 45))
    with pytest.raises(AssertionError, match="mlx-lm scores"):
        check_filter(tiny, docs, [90, 91, 92], [[93, 94]], 80, 200, gated=False)


def join_parts():
    return [100, 101], [[102, 60 + p, 103] for p in range(3)]


def check_join(tiny, anchors, pages, budget, retain, monkeypatch):
    frame, partners = join_parts()
    arena, pipeline = tiny.run(pages)
    keys = [("a", i) for i in range(len(anchors))]
    if retain:
        arena.retention_cap_pages = pages
    rows = {}

    def anchor_done(a, row):
        rows[a] = np.array(row)
        if retain:
            arena.retain(keys[a], len(anchors[a]))
        else:
            arena.free_key(keys[a])

    copies = []
    copy = pipeline.pools.copy
    monkeypatch.setattr(pipeline.pools, "copy",
                        lambda *args: (copies.append(args[0]), copy(*args)))
    loop.run_join(
        tiny.implementation, arena, pipeline, tiny.probe, anchors, [partners], budget,
        stage_frames=[frame], anchor_keys=keys, anchor_done=anchor_done,
        attention_mode="unified", prefix_tree=prefix_tree(anchors, PAGE))
    for a, anchor in enumerate(anchors):
        assert len(rows[a]) == len(partners)
        for p, partner in enumerate(partners):
            tiny.close(rows[a][p], anchor + frame + partner,
                       f"anchor {a} partner {p}")
    # partners after the first copy the anchor's last partial page
    assert copies
    evicted = arena.evicted_keys
    arena.evict_retained(pages)
    assert not arena.accounting.owned
    return evicted


@pytest.mark.parametrize("retain", [False, True])
def test_joins_match_whole_sequences(tiny, retain, monkeypatch):
    """Each anchor's partners read its KV; retained anchors are evicted for pages."""
    for seed, pages in ((0, 40), (1, 400)):
        rng = random.Random(100 + seed)
        anchors = corpus(rng, 30, (0, 5, 16, 20), (1, 7, 16, 30, 45))
        evicted = check_join(tiny, anchors, pages, max(map(len, anchors)) + 60,
                             retain, monkeypatch)
        if retain and pages == 40:
            # the anchors do not fit the arena together
            assert sum(map(len, anchors)) > pages * PAGE
            assert evicted


def test_filter_survivors_feed_a_join(tiny):
    """A join's partners read the KV its anchors kept from the filter."""
    frame, partners = join_parts()
    question = [90, 91, 92]
    rng = random.Random(200)
    docs = corpus(rng, 40, (0, 5, 16, 20), (1, 7, 16, 30, 45))
    arena, pipeline = tiny.run(80)
    # the join has its own readout, so each chunk's answer rows are
    # selected per readout on the device
    stages = filter_stages(tiny, [], [question]) + [
        Stage(suffixes=partners, readout=Probe(tiny.config.hidden), frame=frame)]
    out, _, _ = run_stages(
        tiny.implementation, arena, pipeline, stages, docs, max(map(len, docs)) + 60,
        anchor_keys=[("d", i) for i in range(len(docs))],
        prefix_tree=prefix_tree(docs, PAGE), attention_mode="unified")
    survivors = {d for d, doc in enumerate(docs) if tiny.score(doc + question) > 0}
    assert survivors and len(survivors) < len(docs)
    assert set(out[1]) == survivors
    for d in survivors:
        for p, partner in enumerate(partners):
            tiny.close(out[1][d][p], docs[d] + frame + partner,
                       f"anchor {d} partner {p}")
    assert not arena.accounting.owned


def test_a_join_reads_kv_an_earlier_join_retained(tiny):
    """A fresh anchor borrows the first page of an anchor retained earlier."""
    frame, partners = join_parts()
    anchors = [[1] * 64, [1] * 16 + [3] * 5]
    arena, pipeline = tiny.run(12)
    arena.retention_cap_pages = 12
    keys = [("a", i) for i in range(len(anchors))]

    def run(subset, done):
        loop.run_join(
            tiny.implementation, arena, pipeline, tiny.probe, subset, [partners], 200,
            stage_frames=[frame], anchor_keys=keys[:len(subset)],
            anchor_done=done, attention_mode="unified",
            prefix_tree=prefix_tree(subset, PAGE))

    run(anchors[:1], lambda a, row: arena.retain(keys[a], len(anchors[a])))
    assert arena.is_resident(keys[0])
    rows = {}

    def free(a, row):
        rows[a] = np.array(row)
        arena.free_key(keys[a])

    run(anchors, free)
    for a, anchor in enumerate(anchors):
        for p, partner in enumerate(partners):
            tiny.close(rows[a][p], anchor + frame + partner,
                       f"anchor {a} partner {p}")
    assert not arena.accounting.owned


def test_rows_far_into_a_document_keep_their_rotary_position(tiny):
    """Two stages of questions after documents of thousands of tokens."""
    rng = random.Random(5)
    docs = [[rng.randrange(120) for _ in range(n)] for n in (3000, 2100)]
    check_filter(tiny, docs, [], [[121, 122], [123]], 420, 3200, gated=False)


def test_bf16_error_is_the_size_of_mlx_lm_in_bf16():
    """Quail in bf16 is as close to float32 as mlx-lm in bf16 is."""
    tiny = Tiny("bf16")
    rng = random.Random(9)
    docs = [[rng.randrange(120) for _ in range(n)] for n in (40, 200, 333, 90)]
    tail = [121, 122, 123]
    arena, pipeline = tiny.run(80)
    answers, _, _ = run_stages(
        tiny.implementation, arena, pipeline, filter_stages(tiny, [], [tail]),
        docs, 800, attention_mode="unified", default_attention="unified")
    half = reference_model()
    half.set_dtype(mx.bfloat16)
    ours = theirs = 0.0
    for d, doc in enumerate(docs):
        tokens = doc + tail
        want = tiny.score(tokens)
        last = half.model(mx.array([tokens]))[0, -1].astype(mx.float32)
        theirs = max(theirs, abs(float(last @ tiny.probe.vector) - want))
        ours = max(ours, abs(float(answers[0][d][0]) - want))
    assert theirs > 0
    assert ours < 3 * theirs + 1e-3, (ours, theirs)


# ---- the kernels and the pools

def dense_attention(q, k, v, prefix):
    """Causal attention in float32; the queries are the context's last rows."""
    heads, kv_heads = q.shape[1], k.shape[1]
    k = np.repeat(k, heads // kv_heads, axis=1)
    v = np.repeat(v, heads // kv_heads, axis=1)
    scores = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(q.shape[-1])
    hidden = np.arange(k.shape[0])[None, :] > np.arange(q.shape[0])[:, None] + prefix
    scores = np.where(hidden[None], -np.inf, scores)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", weights, v)


@pytest.mark.parametrize("dtype_name", sorted(DTYPES))
@pytest.mark.parametrize("geometry", [(4, 2, 64), (16, 8, 128)])
def test_paged_attention_reads_a_shared_prefix_in_place(dtype_name, geometry):
    """Readers of one prefix share its pages and write their own rows."""
    n_q, n_kv, dim = geometry
    dtype = DTYPES[dtype_name]
    rng = np.random.default_rng(0)

    def rows(count, heads):
        values = rng.standard_normal((count, heads, dim)).astype(np.float32)
        return np.array(mx.array(values).astype(dtype).astype(mx.float32))

    prefix, fresh = 48, [20, 9, 33]
    pools = MlxKVPools(dtype)
    pools.build(PAGE, [(16, n_kv, dim)])
    prefix_k, prefix_v = rows(prefix, n_kv), rows(prefix, n_kv)
    pools.write(0, mx.array(prefix_k).astype(dtype), mx.array(prefix_v).astype(dtype),
                mx.array(np.arange(prefix), dtype=mx.int64))
    own = [(rows(n, n_q), rows(n, n_kv), rows(n, n_kv)) for n in fresh]
    tables, slots, page = [], [], prefix // PAGE
    for n in fresh:
        pages = list(range(page, page + -(-n // PAGE)))
        page = pages[-1] + 1
        tables.append(list(range(prefix // PAGE)) + pages)
        slots.extend(pages[i // PAGE] * PAGE + i % PAGE for i in range(n))
    width = max(map(len, tables))

    def stacked(part):
        return mx.array(np.concatenate([o[part] for o in own])).astype(dtype)

    pools.write(0, stacked(1), stacked(2), mx.array(slots, dtype=mx.int64))
    k_pool, v_pool = pools.paged_kv(0)
    out = kernels.paged_attention(
        stacked(0), k_pool, v_pool,
        table=mx.array([t + [0] * (width - len(t)) for t in tables], dtype=mx.int32),
        used=mx.array([prefix + n for n in fresh], dtype=mx.int32),
        cu_q=mx.array(np.concatenate([[0], np.cumsum(fresh)]), dtype=mx.int32),
        max_used=prefix + max(fresh), scale=dim ** -0.5)
    assert out.shape == (sum(fresh), n_q, dim) and out.dtype == dtype
    got = np.array(out.astype(mx.float32))
    row = 0
    for q, k, v in own:
        want = dense_attention(q, np.concatenate([prefix_k, k]),
                               np.concatenate([prefix_v, v]), prefix)
        error = np.abs(got[row:row + len(q)] - want)
        # a bf16 value keeps 8 significant bits
        allowed = 1e-5 if dtype_name == "float32" else 2.0 ** -7 * np.abs(want) + 2e-3
        assert (error <= allowed).all(), error.max()
        row += len(q)


def test_pools_copy_rows_and_rebuild():
    pools = MlxKVPools(mx.float32)
    pools.build(PAGE, [(8, 2, 64)] * 2)
    k = mx.arange(3 * 2 * 64, dtype=mx.float32).reshape(3, 2, 64)
    pools.write(1, k, k + 1, mx.array([5, 6, 40], dtype=mx.int64))
    pools.copy(1, mx.array([5, 40], dtype=mx.int32), mx.array([17, 18], dtype=mx.int64))
    k_rows, v_rows = pools.layer_kv(1)
    assert mx.array_equal(k_rows[mx.array([5, 6, 40, 17, 18])],
                          k[mx.array([0, 1, 2, 0, 2])])
    assert mx.array_equal(v_rows[18], k[2] + 1)
    assert not mx.any(k_rows[7]) and not mx.any(pools.paged_kv(0)[0])
    assert pools.nbytes == 2 * 2 * 8 * PAGE * 2 * 64 * 4
    pools.build(PAGE, [(8, 2, 64), (4, 2, 64)])
    assert pools.paged_kv(0)[0].shape == (8, PAGE, 2, 64)
    assert pools.paged_kv(1)[0].shape == (4, PAGE, 2, 64)
    assert not mx.any(pools.paged_kv(1)[0])


def test_a_geometry_without_a_kernel_is_refused():
    with pytest.raises(ValueError, match="head dim 16"):
        kernels.check_geometry(4, 2, 16, PAGE, mx.bfloat16)


def test_only_paged_unified_chunks_run():
    tiny = Tiny("float32")
    arena, pipeline = tiny.run(16)
    key = ("d", 0)
    arena.activate(key, 4, capacity_tokens=8, base_tokens=4)
    group = dict(key=key, prefix=[1, 2, 3, 4], f=4, suffixes=[[5, 6]], single=True)
    tree = chunk_mod.pack_chunk(
        tiny.implementation, arena, [group], attention_mode="tree")
    with pytest.raises(ValueError, match="unified attention only"):
        pipeline.forward_chunk(tree)
    unpaged = chunk_mod.pack_chunk(
        tiny.implementation, arena, [dict(group, key=("d", 1))],
        attention_mode="unified")
    with pytest.raises(ValueError, match="paged KV only"):
        pipeline.forward_chunk(unpaged)
    assert not pipeline.tree_attention and pipeline.needs_pages
    with pytest.raises(ValueError, match="no mlx forward pass"):
        build_pipeline(SimpleNamespace(arch="diffusion_gemma"), tiny.weights,
                       arena, implementation="mlx")


def test_the_pipeline_refuses_an_arena_of_another_shape():
    tiny = Tiny("float32")
    config = tiny.config
    arena = KVArena(n_layers=config.layers, n_pages=4, page_tokens=PAGE,
                    n_kv=config.n_kv + 1, d_head=config.head_dim,
                    pools=MlxKVPools(mx.float32))
    with pytest.raises(ValueError, match="KV shape"):
        build_pipeline(tiny.spec, tiny.weights, arena, implementation="mlx")
    arena = KVArena(n_layers=config.layers, n_pages=4, page_tokens=PAGE,
                    n_kv=config.n_kv, d_head=config.head_dim,
                    pools=MlxKVPools(mx.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        build_pipeline(tiny.spec, tiny.weights, arena, implementation="mlx")


def test_the_implementation_stages_selects_and_measures():
    implementation = MlxImplementation()
    staged = implementation.stage([[1, 2], [3, 4]], np.int32)
    assert staged.dtype == mx.int32 and staged.shape == (2, 2)
    ids = implementation.stage_tokens(np.array([5, 6, 7], dtype=np.int64))
    assert ids.tolist() == [5, 6, 7]
    rows = mx.arange(12).reshape(6, 2)
    assert implementation.select_rows(rows, np.array([4, 1])).tolist() == [
        [8, 9], [2, 3]]
    assert implementation.select_rows(rows, mx.array([0])).tolist() == [[0, 1]]
    # a test model's list of answers is selected on the host
    assert implementation.select_rows([10, 11, 12], np.array([2, 0])) == [12, 10]
    assert implementation.input_staging() is None
    start = implementation.record_event()
    implementation.synchronize()
    assert implementation.elapsed_ms(start, implementation.record_event()) >= 0
    assert implementation.peak_memory_bytes() > 0
    assert implementation.kv_pools().dtype == mx.bfloat16
    with pytest.raises(NotImplementedError, match="mlx has no score readout"):
        implementation.scores(object())


# ---- the decision head and its readouts

def head_weights(rng, hidden=64, width=16):
    def tensor(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "candidate_norm.weight": 1 + 0.1 * tensor(hidden),
        "candidate_norm.bias": 0.1 * tensor(hidden),
        "query_norm.weight": 1 + 0.1 * tensor(hidden),
        "query_norm.bias": 0.1 * tensor(hidden),
        "key.weight": tensor(width, hidden),
        "query.weight": tensor(width, hidden),
        "candidate_mlp.weight": tensor(width, hidden),
        "candidate_mlp.bias": tensor(width),
        "query_mlp.weight": tensor(width, hidden),
        "scalar.weight": tensor(1, width),
    }


def reference_scores(w, options, last):
    """The decision head in numpy float64, from its definition."""
    def layer_norm(x, weight, bias):
        centered = x - x.mean(-1, keepdims=True)
        return centered / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * weight + bias

    option = layer_norm(options.astype(np.float64), w["candidate_norm.weight"],
                        w["candidate_norm.bias"])
    query = layer_norm(last.astype(np.float64), w["query_norm.weight"],
                       w["query_norm.bias"])
    bilinear = ((option @ w["key.weight"].T)
                * (query @ w["query.weight"].T)[:, None, :]).sum(-1)
    hidden = (option @ w["candidate_mlp.weight"].T + w["candidate_mlp.bias"]
              + (query @ w["query_mlp.weight"].T)[:, None, :])
    gelu = 0.5 * hidden * (1 + np.vectorize(math.erf)(hidden / math.sqrt(2)))
    mlp = (gelu @ w["scalar.weight"].T)[..., 0]
    return bilinear / math.sqrt(w["key.weight"].shape[0]) + mlp


def head(rng):
    """A decision head in MLX and its weights in numpy."""
    weights = head_weights(rng)
    return MlxDecisionHead({k: mx.array(v) for k, v in weights.items()}), weights


def test_decision_head_matches_its_definition():
    rng = np.random.default_rng(1)
    ours, weights = head(rng)
    assert ours.head_dim == 16
    options = np.array(mx.array(rng.standard_normal((5, 2, 64))).astype(
        mx.bfloat16).astype(mx.float32))
    last = np.array(mx.array(rng.standard_normal((5, 64))).astype(
        mx.bfloat16).astype(mx.float32))
    got = ours.scores(mx.array(options).astype(mx.bfloat16),
                      mx.array(last).astype(mx.bfloat16))
    assert got.dtype == mx.float32
    assert np.allclose(np.array(got), reference_scores(weights, options, last),
                       atol=1e-4)


def test_decision_head_matches_the_cuda_head():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.readout import DecisionHead, DecisionRows

    rng = np.random.default_rng(1)
    ours, weights = head(rng)
    theirs = DecisionHead(torch, torch.nn.functional,
                          {k: torch.from_numpy(v) for k, v in weights.items()})
    offsets, counts = (5, 2, 0), [6, 1, 6]
    normed = rng.standard_normal((13, 64)).astype(np.float32)
    got = MlxDecisionRows(ours, offsets).scores(mx.array(normed), counts)
    want = DecisionRows(torch, theirs, offsets).scores(
        torch.from_numpy(normed), counts)
    assert np.allclose(np.array(got), want.numpy(), atol=1e-4)


def test_decision_rows_read_offsets_from_each_answers_trailing_rows():
    rng = np.random.default_rng(1)
    ours, weights = head(rng)
    # answers of 6 trailing rows, and a frame entry's single row
    offsets, counts = (5, 2, 0), [6, 1, 6]
    normed = rng.standard_normal((13, 64)).astype(np.float32)
    rows = MlxDecisionRows(ours, offsets)
    assert rows.trailing_rows == 6
    got = np.array(rows.scores(mx.array(normed), counts))
    read = np.array([[0, 3, 5], [6, 6, 6], [7, 10, 12]])
    want = reference_scores(weights, normed[read[:, :2]], normed[read[:, 2]])
    assert np.allclose(got, want, atol=1e-4)
    # every answer has the trailing rows when no counts are given
    got = np.array(rows.scores(mx.array(normed[:12])))
    read = np.array([[0, 3, 5], [6, 9, 11]])
    want = reference_scores(weights, normed[read[:, :2]], normed[read[:, 2]])
    assert np.allclose(got, want, atol=1e-4)


def test_decision_readouts_return_host_values():
    rng = np.random.default_rng(2)
    ours, weights = head(rng)
    offsets = (4, 2, 0)
    normed = rng.standard_normal((15, 64)).astype(np.float32)
    read = np.array([[0, 2, 4], [5, 7, 9], [10, 12, 14]])
    want = reference_scores(weights, normed[read[:, :2]], normed[read[:, 2]])
    rows = mx.array(normed)

    bits = MlxDecisions(ours, offsets)
    assert bits.dtype is None
    assert bits.result(bits.submit(rows)) == [int(b) for b in want[:, 1] > want[:, 0]]

    scores = MlxDecisionScores(ours, offsets)
    got = scores.result(scores.submit(rows))
    assert got.dtype == np.float32 == scores.dtype
    yes = 1 / (1 + np.exp(np.clip(want[:, 0] - want[:, 1], -700, 700)))
    assert np.allclose(got, yes, atol=1e-4)

    choices = MlxImplementation().decision_choices(ours, offsets)
    assert isinstance(choices, MlxDecisionChoices)
    assert choices.dtype == np.dtype((np.float32, (2,)))
    got = choices.result(choices.submit(rows))
    assert got.shape == (3, 2) and np.allclose(got, want, atol=1e-4)


def test_decisions_through_the_scheduler_match_whole_sequences(tiny):
    """Yes/no answers read three trailing rows of each request."""
    ours, weights = head(np.random.default_rng(4))
    offsets = (4, 2, 0)
    readout = MlxDecisions(ours, offsets)
    rng = random.Random(11)
    docs = corpus(rng, 30, (0, 16, 20), (7, 16, 30, 45))
    tails = [[110, 111, 112, 113, 114, 115], [116, 117, 118, 119, 120]]
    arena, pipeline = tiny.run(120)
    got, _, _ = loop.run_filter(
        tiny.implementation, arena, pipeline, readout, docs, tails,
        max(map(len, docs)) + 100, arena_writes=True, attention_mode="unified",
        prefix_tree=prefix_tree(docs, PAGE))
    margin = 5 * tiny.tolerance
    checked = 0
    for d, doc in enumerate(docs):
        for stage, tail in enumerate(tails):
            rows = np.array(tiny.hidden(doc + tail))[[-5, -3, -1]]
            scores = reference_scores(weights, rows[None, :2], rows[None, 2])[0]
            difference = float(scores[1] - scores[0])
            if abs(difference) > margin:
                assert got[d][stage] == int(difference > 0), (d, stage)
                checked += 1
            if not got[d][stage]:
                break
    assert checked > len(docs)
    assert not arena.accounting.owned


# ---- the loader

def write_package(directory, model, head, *, decision):
    """Save the tiny model as a checkpoint directory."""
    tensors = dict(tree_flatten(model.parameters()))
    config = dict(CONFIG)
    config["rope_parameters"] = {"rope_theta": config.pop("rope_theta")}
    weights = directory
    if decision:
        (directory / "config.json").write_text(json.dumps({"model_type": "decision2"}))
        weights = directory / "backbone"
        weights.mkdir()
        # the backbone's names lack the prefix, and its projections are bf16
        tensors = {name.removeprefix("model."): (
            array.astype(mx.bfloat16) if "proj" in name else array)
            for name, array in tensors.items()}
        mx.save_safetensors(str(directory / "decision_head.safetensors"), head)
    (weights / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(weights / "model.safetensors"), tensors)


@pytest.mark.parametrize("decision", [True, False])
def test_loader_reads_a_checkpoint_directory(tmp_path, decision):
    tiny = Tiny("bf16")
    saved = {k: mx.array(v)
             for k, v in head_weights(np.random.default_rng(6)).items()}
    write_package(tmp_path, tiny.model, saved, decision=decision)
    weights = load_qwen3_weights(tmp_path, mx.bfloat16)
    assert weights.config == tiny.config
    assert all(array.dtype == mx.bfloat16 for array in weights.arrays())
    assert weights.nbytes == tiny.weights.nbytes
    for ours, theirs in zip(weights.arrays(), tiny.weights.arrays()):
        assert mx.array_equal(ours, theirs)
    if decision:
        loaded = load_decision_head(tmp_path)
        assert loaded.head_dim == 16
        assert all(mx.array_equal(loaded.w[name], saved[name]) for name in saved)
