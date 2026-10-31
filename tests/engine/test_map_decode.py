"""AI.MAP's decode rows on a model that checks every KV read.

The model gives each row a hash of its token and of the KV it reads,
as tests/helpers/kv_checker.py does, and chooses the row's next token
from that hash. An answer comes out right only if every decode row
read its own prompt and answer so far at the right positions and
wrote its KV where the next row reads it.
"""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from helpers.fakes import cpu_staging, fake_pipeline
from helpers.kv_checker import HashModel, Setup, chain, cpu_torch, make_arena

from quail.backends.quail.executor.admission import pages_for
from quail.backends.quail.executor.answer_admission import HeadroomAdmission
from quail.backends.quail.executor.answer_survival import answer_survival
from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.operators.classify import ClassifyPart
from quail.backends.quail.executor.operators.filter import ConditionGate, FilterPart
from quail.backends.quail.executor.operators.map import MapPart
from quail.backends.quail.executor.parts import gated_stages
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.catalog.tokens import prefix_tree
from quail.cost.decode import DecodeCost
from quail.specs import DEVICES, QWEN3_4B_FP8

VOCABULARY = 1024
STOP = VOCABULARY - 1


def next_token(h: int) -> int:
    """The token the model chooses after a row whose hash is h."""
    return STOP if h % 5 == 0 else 10 + h % 900


class TokenModel(HashModel):
    """Answer each read row with a one-hot of the token its hash chooses."""

    def __init__(self, arena):
        super().__init__(arena, answers={}, canvas=())
        self.shared_reads = 0   # passes whose rows read shared pages together
        self.admitted = set()
        self.decode_slots = []

    def forward_chunk(self, chunk):
        self.admitted.update(chunk.fresh_keys)
        if chunk.decode_rows is not None:
            self.decode_slots.append(set(chunk.decode_rows.slots.tolist()))
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        hashes = [None] * len(ids)
        if chunk.attention_mode == "unified":
            self._unified(self._whole(chunk.meta), ids, positions, hashes)
        else:
            self._tree(chunk.meta, ids, positions, hashes)
        self._check_state(chunk.meta, ids, positions, hashes)
        rows = chunk.final_indices.tolist()
        out = torch.zeros(len(rows), VOCABULARY)
        out[torch.arange(len(rows)),
            torch.tensor([next_token(hashes[row]) for row in rows])] = 1.0
        return out


    def _whole(self, meta):
        """Reconstruct each sequence's full context with shared pages first."""
        unified, shared = meta["unified"], meta["shared"]
        if shared is None:
            return unified
        self.shared_reads += 1
        cu = unified["cu_q"].tolist()
        sequence = {row: i for i, (start, end) in enumerate(zip(cu, cu[1:]))
                    for row in range(start, end)}
        table = [list(row) for row in unified["table"].tolist()]
        used = unified["used"].tolist()
        rows, groups = shared["rows"].tolist(), shared["cu_q"].tolist()
        tokens = shared["used"].tolist()
        for g, pages in enumerate(shared["table"].tolist()):
            pages = pages[:-(-tokens[g] // self.page)]
            for i in {sequence[row] for row in rows[groups[g]:groups[g + 1]]}:
                table[i] = pages + table[i]
                used[i] += tokens[g]
        width = max(map(len, table))
        return dict(unified, used=torch.tensor(used), table=torch.tensor(
            [row + [0] * (width - len(row)) for row in table]))


class Bits:
    """A filter readout: a row passes when its chosen token is odd."""

    dtype = None
    trailing_rows = 1
    normalize = False

    @staticmethod
    def submit(normed):
        return (normed.argmax(dim=1) % 2).tolist()

    @staticmethod
    def result(handle):
        return handle


def greedy(prompt, limit):
    """The answer greedy decoding writes after a prompt."""
    tokens = []
    while len(tokens) < limit:
        token = next_token(chain(prompt + tokens))
        if token == STOP:
            break
        tokens.append(token)
    return tokens


def map_node(name, tail, limit, attention=""):
    return SimpleNamespace(attention=attention, spec=SimpleNamespace(
        name=name, prompt_token_parts=((), tuple(tail)), max_answer_tokens=limit,
        expected_answer_tokens=min(64, limit), aliases=("d",),
        output_columns=("d", name)))


def events_torch(lag):
    """CPU torch whose events report done only after lag queries."""
    torch_ = cpu_torch()

    def event(**kw):
        asked = []

        def query():
            asked.append(True)
            return len(asked) > lag

        return SimpleNamespace(record=lambda: None, query=query,
                               synchronize=lambda: None,
                               elapsed_time=lambda other: 1.0)

    torch_.cuda.Event = event
    return torch_


def classification_and_condition(ids):
    labels = [None] * len(ids)

    def classify(anchor, row):
        labels[anchor] = "keep" if row[0] else "drop"
        return True

    classification = ClassifyPart(
        SimpleNamespace(spec=SimpleNamespace(name="label", aliases=("d",),
                                              output_columns=("d", "label"))), ids,
        SimpleNamespace(labels=labels, stages=[Stage(
            suffixes=[[702]], frame=[700, 701], single=True,
            readout=Bits(), decide=classify)]))

    def accepts(table):
        assert table["label"][0].as_py() is not None
        return pa.array([table["label"][0].as_py() == "keep"])

    condition = ConditionGate(SimpleNamespace(
        column="label", predicate=SimpleNamespace(accepts=accepts)),
        classification, classification)
    return classification, condition


@pytest.mark.parametrize("path", ["unified", "tree"])
@pytest.mark.parametrize("lag", [0, 3])
@pytest.mark.parametrize("budget", [40, 400])
@pytest.mark.parametrize("question", [None, [700, 701]])
def test_map_answers_equal_greedy_decoding(monkeypatch, budget, question, lag,
                                           path):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(0)
    documents = [rng.integers(20, 900, size=int(n)).tolist()
                 for n in rng.integers(3, 30, size=12)]
    tails = [[500, 501, 502], [600, 601]]
    limits = [6, 4]
    setup = Setup(path=path, pages=200, budget=budget)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk,
                             tree_attention=True, canvas_ids=())
    torch_ = events_torch(lag)
    ids = list(range(len(documents)))
    maps = MapPart(None, [map_node("a", tails[0], limits[0]),
                          map_node("b", tails[1], limits[1])], ids, 0)
    parts = [maps]
    if question is not None:
        parts = [FilterPart(SimpleNamespace(question_token_ids=question), ids,
                            Bits(), 0), maps]
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], None, setup.page_tokens, len(stages) - 1)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], setup.page_tokens)
    maps.decoder = decoder
    stats = {}
    run_stages(torch_, arena, pipeline, stages, prefixes, budget,
               anchor_keys=keys, attention_mode=path, prefix_tree=tree,
               decoder=decoder, dependencies=dependencies, stats=stats)
    passed = [question is None
              or next_token(chain(document + question)) % 2 == 1
              for document in documents]
    for q, (tail, limit) in enumerate(zip(tails, limits)):
        assert maps.asked[q].tolist() == passed
        asked = np.flatnonzero(maps.asked[q])
        got = [answer.tolist() for answer in decoder.answers(q, maps._slots(q, asked))]
        want = [greedy(documents[d] + tail, limit) for d in asked]
        assert got == want, f"map {q}"
        # answers that stop early and answers cut at the limit
        assert {len(answer) == limit for answer in want} == {True, False}
    dropped = np.flatnonzero(~np.asarray(passed))
    for q in range(len(tails)):
        slots = maps._slots(q, dropped)
        assert not decoder.fed[slots].any()
        assert all(not decoded.intersection(slots) for decoded in model.decode_slots)
        if q:
            assert all(keys[a] not in model.admitted for a in slots)
    assert any(set(maps.slots[:, d]).issubset(decoded)
               for d in np.flatnonzero(passed) for decoded in model.decode_slots)
    assert stats["borrowed_tokens"] > 0
    assert decoder.rows_fed.min() > 0
    assert (question is None) == all(passed)
    assert not decoder.active
    assert not arena.accounting.owned


@pytest.mark.parametrize("pages", [6, 300])
@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("lag", [0, 3])
def test_conditions_gate_all_branches_and_limits_wait_for_every_answer(
        monkeypatch, pages, limit, lag):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(7)
    documents = [rng.integers(20, 900, size=24).tolist() for _ in range(12)]
    documents.sort(
        key=lambda document: next_token(chain(document + [700, 701, 702])) % 2)
    ids = list(range(len(documents)))
    tails, limits = [[500, 501], [600, 601], [650, 651]], [1, 7, 9]
    setup = Setup(path="unified", pages=pages, budget=200)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=())
    torch_ = events_torch(lag)
    classification, condition = classification_and_condition(ids)
    labels = classification.plan.labels
    maps = MapPart(None, [map_node(f"m{q}", tail, answer_limit, "tree")
                          for q, (tail, answer_limit) in enumerate(zip(tails, limits))],
                   ids, 0)
    stages = gated_stages([classification, condition, maps])
    assert len(stages) == 2
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], None, setup.page_tokens, 1)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], setup.page_tokens)
    maps.decoder = decoder
    survivors = []

    def settled(anchor, survived, row):
        if anchor < len(documents) and survived:
            survivors.append(anchor)
        arena.free_key(keys[anchor])

    stats = {}
    run_stages(torch_, arena, pipeline, stages, prefixes, setup.budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, on_settled=settled, stats=stats,
               limit=limit, survival=1.0)
    # a branch settles at its own end, so documents in flight when the
    # limit is reached still finish; the limit counts whole documents
    assert (len(survivors) == labels.count("keep") if limit is None
            else limit <= len(survivors) <= labels.count("keep"))
    assert "drop" in labels
    for q, (tail, answer_limit) in enumerate(zip(tails, limits)):
        assert set(np.flatnonzero(maps.asked[q])) == set(survivors)
        got = [answer.tolist() for answer in decoder.answers(
            q, maps._slots(q, survivors))]
        assert got == [greedy(documents[d] + tail, answer_limit) for d in survivors]
        for d, label in enumerate(labels):
            if label != "keep":
                slot = maps.slots[q, d]
                assert decoder.fed[slot] == 0
                if q:
                    assert keys[slot] not in model.admitted
    assert stats["borrowed_tokens"] > 0
    assert not decoder.active
    assert not arena.accounting.owned


@pytest.mark.parametrize("num_maps", [1, 2, 3])
@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("pages", [6, 300])
def test_fused_document_operators_equal_running_each_operator_alone(
        monkeypatch, num_maps, limit, pages):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(11)
    documents = [rng.integers(20, 900, size=24).tolist() for _ in range(20)]
    ids = list(range(len(documents)))
    nodes = [map_node(f"m{q}", tail, answer_limit, "tree")
             for q, (tail, answer_limit) in enumerate(zip(
                 [[500, 501], [600, 601], [650, 651]], [1, 7, 9]))][:num_maps]
    setup = Setup(path="unified", pages=pages, budget=200)
    question = SimpleNamespace(question_token_ids=[750, 751])

    def run(parts, ids, limit=None):
        arena = make_arena(setup)
        model = TokenModel(arena)
        pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=())
        torch_ = events_torch(3)
        stages = gated_stages(parts)
        prefixes, keys = [documents[d] for d in ids], [("d", d) for d in ids]
        maps = next((part for part in parts if isinstance(part, MapPart)), None)
        decoder, tree, dependencies = None, None, {}
        if maps is not None:
            prefixes, keys, tree, dependencies = maps.branch_inputs(
                prefixes, keys, tree, setup.page_tokens, len(stages) - 1)
            decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes),
                                 maps.answer_limits, [STOP], setup.page_tokens)
            maps.decoder = decoder
        every, _, _ = run_stages(
            torch_, arena, pipeline, stages, prefixes, setup.budget,
            anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
            decoder=decoder, limit=limit, survival=1.0)
        assert not arena.accounting.owned
        return every

    classification, condition = classification_and_condition(ids)
    maps = MapPart(None, nodes, ids, 0)
    run([FilterPart(question, ids, Bits()), classification, condition, maps],
        ids, limit)
    kept = np.flatnonzero(maps.asked[0]).tolist()
    fused = [{"id": ids[d], "label": classification.plan.labels[d], **{
        node.spec.name: maps.decoder.answers(q, maps._slots(q, [d]))[0].tolist()
        for q, node in enumerate(nodes)}} for d in kept]
    assert all(np.flatnonzero(asked).tolist() == kept for asked in maps.asked)

    every = run([FilterPart(question, ids, Bits())], ids)
    survivors = [ids[d] for d in sorted(every[0]) if every[0][d][0]]
    classification, condition = classification_and_condition(survivors)
    run([classification], survivors)
    separate = [{"id": d, "label": classification.plan.labels[a]}
                for a, d in enumerate(survivors)
                if condition.gate(("d", d)) is not Stage.DROP]
    survivors = [row["id"] for row in separate]
    for node in nodes:
        maps = MapPart(None, [node], survivors, 0)
        run([maps], survivors)
        for a, row in enumerate(separate):
            answer = maps.decoder.answers(0, [a])[0].tolist()
            assert answer == greedy(documents[row["id"]]
                                    + list(node.spec.prompt_token_parts[1]),
                                    node.spec.max_answer_tokens)
            row[node.spec.name] = answer
    assert len(separate) > 2
    assert fused == separate[:limit]


def batch_decode(tokens, skip_special_tokens=True):
    return [" ".join(str(int(token)) for token in answer) for answer in tokens]


def map_state(setup, arena, forward_chunk):
    pipeline = fake_pipeline(forward_chunk=forward_chunk, canvas_ids=())
    return SimpleNamespace(
        torch=events_torch(3), chunk_tokens=setup.budget, async_answers=Bits(),
        loaded_model=SimpleNamespace(
            arena=arena, pipeline=pipeline, input_staging=None,
            model_spec=QWEN3_4B_FP8,
            model=SimpleNamespace(quail_full_head=torch.eye(VOCABULARY)),
            vocabulary=SimpleNamespace(stop_ids=[STOP], tokenizer=SimpleNamespace(
                batch_decode=batch_decode))))


# every pair of settings appears in some case
@pytest.mark.parametrize("num_maps,limit,lag,budget,stop_modulus,path", [
    (3, None, 3, 20, 0, "tree"),
    (1, 2, 0, 200, 19, "tree"),
    (2, None, 3, 200, 19, "unified"),
    (2, 2, 0, 20, 0, "unified"),
    (1, None, 3, 20, 0, "unified"),
    (3, 2, 0, 200, 19, "unified"),
    (2, None, 0, 20, 19, "tree"),
    (3, 2, 3, 200, 0, "tree"),
])
def test_evicted_answers_recompute_exactly_with_gates_and_limits(
        monkeypatch, num_maps, limit, lag, budget, stop_modulus, path,
        state_slots=0, window=None):
    cpu_staging(monkeypatch)
    monkeypatch.setattr(__name__ + ".next_token", lambda h: (
        STOP if stop_modulus and h % stop_modulus == 0 else 10 + h % 900))
    rng = np.random.default_rng(19)
    documents = [[20, 21, 22, 23] + rng.integers(20, 400, size=4).tolist()
                 for _ in range(12)]
    ids = list(range(len(documents)))
    tails = [[500, 501], [600, 601], [650, 651]][:num_maps]
    limits = [24, 20, 28][:num_maps]
    setup = Setup(path=path, pages=16, page_tokens=4, budget=budget,
                  state_slots=state_slots, window=window, sliding_pages=16)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=(),
                             tree_attention=True)
    torch_ = events_torch(lag)
    classification, condition = classification_and_condition(ids)
    maps = MapPart(None, [map_node(f"m{q}", tail, answer_limit, "tree")
                          for q, (tail, answer_limit) in enumerate(zip(tails, limits))],
                   ids, 0)
    stages = gated_stages([classification, condition, maps])
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], prefix_tree(documents, setup.page_tokens),
        setup.page_tokens, 1)
    # A one-token prior predicts no growth, so answers outgrow their
    # initial pages and the tight arena evicts some.
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], setup.page_tokens,
                         expected_tokens=[1] * len(limits))
    maps.decoder = decoder
    stats = {}
    run_stages(torch_, arena, pipeline, stages, prefixes, budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats, limit=limit, attention_mode=path)
    asked = np.flatnonzero(maps.asked[0])
    assert len(asked) >= (limit or 1)
    assert all(classification.plan.labels[a] == "keep" for a in asked)
    for q, (tail, answer_limit) in enumerate(zip(tails, limits)):
        np.testing.assert_array_equal(maps.asked[q], maps.asked[0])
        got = [answer.tolist() for answer in decoder.answers(q, maps._slots(q, asked))]
        assert got == [greedy(documents[a] + tail, answer_limit) for a in asked]
    # one map in a roomy pass can finish within what the arena frees;
    # several always outgrow it
    if num_maps > 1:
        assert stats["num_evictions"] > 0
    # an evicted answer whose last token in flight ends it needs no recompute
    assert stats["recomputed_tokens"] == 0 or stats["num_evictions"] > 0
    if state_slots:
        assert stats["recomputed_tokens"] > 0
        assert model.state_starts.max() > 1
        assert len(arena.accounting.free_slots) == state_slots
    assert not decoder.active
    assert not arena.accounting.owned


@pytest.mark.parametrize("budget,lag", [(20, 0), (200, 3)])
@pytest.mark.parametrize("window", [None, 5])
def test_hybrid_answers_recompute_exactly_and_reuse_checkpoint_slots(
        monkeypatch, budget, lag, window):
    test_evicted_answers_recompute_exactly_with_gates_and_limits(
        monkeypatch, num_maps=3, limit=None, lag=lag, budget=budget,
        stop_modulus=0, path="unified", state_slots=24, window=window)


def test_eviction_prefers_long_remaining_answers_over_near_limit_answers():
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), 4,
                         [24, 6], [STOP], 4, expected_tokens=[12, 3])
    for a, q in enumerate([0, 0, 1, 1]):
        decoder.start(a, q, 8, [0, 1, 2])
    decoder.written[0][:2] = [2, 23]
    decoder.written[1][2:] = [1, 5]
    order = decoder.eviction_order().tolist()
    assert order[0] == 0
    assert order.index(2) < order.index(1)
    assert order.index(2) < order.index(3)
    decoder.pause(0)
    assert 0 not in decoder.eviction_order()
    curves = {q: answer_survival(decoder.lengths[q], decoder.written[q],
                                 decoder.expected_tokens[q], int(decoder.limits[q]))
              for q in range(len(decoder.limits))}
    np.testing.assert_array_equal(decoder.eviction_order(curves),
                                  decoder.eviction_order())


def test_recomputed_documents_keep_shared_reads_correct_at_later_map_stages(
        monkeypatch):
    cpu_staging(monkeypatch)
    monkeypatch.setattr(__name__ + ".next_token", lambda h: 10 + h % 900)
    documents = [list(range(20, 24)) + list(range(30 + a, 34 + a))
                 for a in range(8)]
    ids = list(range(len(documents)))
    setup = Setup(path="unified", pages=20, page_tokens=4, budget=24)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=())
    torch_ = events_torch(3)
    tails = [[500, 501], [600, 601], [650, 651], [670, 671]]
    limits = [16, 28, 20, 24]
    nodes = [map_node(f"m{q}", tail, answer_limit, "tree")
             for q, (tail, answer_limit) in enumerate(zip(tails, limits))]
    first = MapPart(None, nodes[:2], ids, 0)
    second = MapPart(None, nodes[2:], ids, 2)
    question = [750, 751]
    parts = [first, FilterPart(SimpleNamespace(question_token_ids=question), ids,
                               Bits(), 0), second]
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = first.branch_inputs(
        documents, [("d", a) for a in ids], prefix_tree(documents, 4), 4, 0)
    prefixes, keys, tree, added = second.branch_inputs(prefixes, keys, tree, 4, 2)
    dependencies.update(added)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], 4)
    first.decoder = second.decoder = decoder
    stats = {}
    run_stages(torch_, arena, pipeline, stages, prefixes, setup.budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats)
    passed = [bool(next_token(chain(document + question)) % 2)
              for document in documents]
    for q, (tail, answer_limit) in enumerate(zip(tails, limits)):
        part, index = (first, q) if q < 2 else (second, q - 2)
        np.testing.assert_array_equal(part.asked[index], True if q < 2 else passed)
        asked = np.flatnonzero(part.asked[index])
        got = [answer.tolist() for answer in decoder.answers(
            q, part._slots(index, asked))]
        assert got == [greedy(documents[a] + tail, answer_limit) for a in asked]
    assert stats["num_evictions"] > 0
    assert not arena.accounting.owned


def run_maps_under_eviction(monkeypatch, mode, num_maps, budget, lag, path,
                            documents=None, pages=20, stop_every=7):
    """Run maps over twelve documents, by default, in a small pool.

    Returns:
        The run's stats and the token model, after checking every answer
        equals greedy decoding.
    """
    cpu_staging(monkeypatch)
    monkeypatch.setenv("QUAIL_ANSWER_ADMISSION", mode)
    monkeypatch.setattr(__name__ + ".next_token", lambda h: (
        STOP if h % stop_every == 0 else 10 + h % 900))
    rng = np.random.default_rng(23)
    if documents is None:
        documents = [[20, 21, 22, 23]
                     + rng.integers(20, 400, size=4 + 4 * (i % 3)).tolist()
                     for i in range(12)]
    ids = list(range(len(documents)))
    tails = [[500, 501], [600, 601], [650, 651]][:num_maps]
    limits = [24, 20, 28][:num_maps]
    setup = Setup(path=path, pages=pages, page_tokens=4, budget=budget)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=(),
                             tree_attention=True)
    maps = MapPart(None, [map_node(f"m{q}", tail, answer_limit, "tree")
                          for q, (tail, answer_limit) in enumerate(zip(tails, limits))],
                   ids, 0)
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], prefix_tree(documents, setup.page_tokens),
        setup.page_tokens, 0)
    decoder = MapDecoder(events_torch(lag), torch.eye(VOCABULARY), len(prefixes),
                         limits, [STOP], setup.page_tokens,
                         expected_tokens=[1] * len(limits))
    maps.decoder = decoder
    stats = {}
    run_stages(events_torch(lag), arena, pipeline, gated_stages([maps]), prefixes,
               budget, anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats, attention_mode=path,
               decode_cost=DecodeCost.from_specs(QWEN3_4B_FP8, DEVICES["h100-sxm"]))
    for q, (tail, answer_limit) in enumerate(zip(tails, limits)):
        got = [answer.tolist() for answer in decoder.answers(q, maps._slots(q, ids))]
        assert got == [greedy(documents[a] + tail, answer_limit) for a in ids]
    assert not decoder.active
    assert not arena.accounting.owned
    return stats, model


@pytest.mark.parametrize("mode", ["vllm-like", "headroom"])
@pytest.mark.parametrize("num_maps", [1, 3])
@pytest.mark.parametrize("budget,lag", [(20, 0), (200, 3)])
@pytest.mark.parametrize("path", ["unified", "tree"])
def test_headroom_admission_answers_equal_greedy_decoding_under_eviction(
        monkeypatch, mode, num_maps, budget, lag, path):
    stats, model = run_maps_under_eviction(monkeypatch, mode, num_maps, budget,
                                           lag, path)
    assert stats["answer_admission"] == mode
    work = stats["pass_work"]
    assert work["replay_tokens"] == stats["recomputed_tokens"]
    assert work["decode_rows"] == sum(len(slots) for slots in model.decode_slots)
    assert work["sol_s"] > 0


@pytest.mark.parametrize("path", ["unified", "tree"])
def test_a_replay_reads_the_prefix_its_siblings_still_hold(monkeypatch, path):
    from quail.backends.quail.executor.stage_executor import StageExecutor

    kept, _ = run_maps_under_eviction(monkeypatch, "vllm-like", 3, 20, 0, path)
    monkeypatch.setattr(StageExecutor, "_lender", lambda self, a: (None, 0))
    whole, _ = run_maps_under_eviction(monkeypatch, "vllm-like", 3, 20, 0, path)
    assert kept["num_evictions"] and whole["num_evictions"]
    assert (kept["recomputed_tokens"] / kept["num_evictions"]
            < whole["recomputed_tokens"] / whole["num_evictions"])


def test_a_replay_reads_its_document_prefix_after_its_last_reader_leaves(
        monkeypatch):
    from quail.backends.quail.executor.stage_executor import StageExecutor

    rng = np.random.default_rng(5)
    documents = [rng.integers(20, 400, size=32).tolist() for _ in range(4)]
    run = dict(documents=documents, pages=20, stop_every=97)
    cached, _ = run_maps_under_eviction(monkeypatch, "vllm-like", 3, 200, 0,
                                        "tree", **run)
    monkeypatch.setattr(StageExecutor, "_cache_prefix", lambda self, a: None)
    dropped, _ = run_maps_under_eviction(monkeypatch, "vllm-like", 3, 200, 0,
                                         "tree", **run)
    assert cached["num_evictions"] == dropped["num_evictions"] > 0
    assert cached["recomputed_tokens"] < dropped["recomputed_tokens"]


def headroom_admission(written=0, ended=(), expected_tokens=16, limit=32,
                       num_documents=3, prefix_tokens=8):
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), num_documents,
                         [limit], [STOP], 4, expected_tokens=[expected_tokens])
    decoder.answer_slots = {0: np.arange(num_documents)}
    for a, length in ended:
        decoder.lengths[0][a] = length
        decoder.written[0][a] = length
    cost = DecodeCost.from_specs(QWEN3_4B_FP8, DEVICES["h100-sxm"])
    admission = HeadroomAdmission(decoder, [prefix_tokens] * num_documents, 6, {},
                                  [0] * num_documents,
                                  lambda tokens: pages_for(tokens, 4), cost)
    return decoder, admission


def test_vllm_like_admission_admits_when_the_prompt_fits_but_waits_for_replays():
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), 2,
                         [32], [STOP], 4, expected_tokens=[16])
    decoder.answer_slots = {0: np.arange(2)}
    decoder.start(0, 0, 10, [0, 1, 2, 3])
    admission = HeadroomAdmission(decoder, [8, 8], 6, {}, [0, 0],
                                  lambda tokens: pages_for(tokens, 4),
                                  choose_headroom=False)
    admission.refresh(np.array([0, -1]))
    assert admission.allows(1, 4, 4)
    assert not admission.allows(1, 3, 4)
    decoder.pause(0)
    admission.refresh(np.array([0, -1]), replay_pages=4)
    assert not admission.allows(1, 4, 4)
    assert admission.allows(1, 8, 4)
    assert admission.waits()


def test_vllm_like_admission_evicts_the_answer_started_or_resumed_last():
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), 3,
                         [32], [STOP], 4, expected_tokens=[16])
    decoder.answer_slots = {0: np.arange(3)}
    admission = HeadroomAdmission(decoder, [8, 8, 8], 6, {}, [0, 0, 0],
                                  lambda tokens: pages_for(tokens, 4),
                                  choose_headroom=False)
    for a in (1, 0, 2):
        decoder.start(a, 0, 10, [3 * a, 3 * a + 1, 3 * a + 2])
    assert admission.eviction_order().tolist() == [2, 0, 1]
    decoder.pause(1)
    decoder.resume(1, [3, 4, 5])
    assert admission.eviction_order().tolist() == [1, 2, 0]


def test_headroom_grows_with_long_answers_and_shrinks_when_pages_are_valuable():
    long_ended = [(0, 30), (1, 31)]
    _, admission = headroom_admission(ended=long_ended, num_documents=4,
                                      prefix_tokens=8000)
    admission.refresh(np.array([-2, -2, -1, -1]))
    roomy = admission._candidate_headroom(2)
    assert roomy > 0
    admission.page_value = 1.0
    assert admission._candidate_headroom(2) == 0
    _, short = headroom_admission(ended=[(0, 1), (1, 1)], expected_tokens=1,
                                  num_documents=4, prefix_tokens=8000)
    short.refresh(np.array([-2, -2, -1, -1]))
    assert short._candidate_headroom(2) < roomy


def test_a_page_is_priced_over_the_prompt_and_headroom_a_document_takes():
    _, admission = headroom_admission(ended=[(0, 30), (1, 31)], num_documents=4,
                                      prefix_tokens=8000)
    admission.refresh(np.array([-2, -2, -1, -1]))
    value = admission.page_value
    added = admission._added_work(2)
    base = admission.work if admission.work.tokens else added
    gain = admission.decode_cost.gain(base, added, admission.expected_tokens[2])
    prompt_pages = admission.page_cost(8000 + 6)
    headroom = int(admission._admission_headroom(2, value)[1].sum())
    assert headroom > 0
    assert value == gain / (prompt_pages + headroom)
    assert value < gain / prompt_pages


def test_headroom_of_running_answers_is_counted_against_new_documents():
    # rebuilding a long prompt costs more than a page saves the next document
    decoder, admission = headroom_admission(ended=[(2, 30)], prefix_tokens=8000)
    decoder.start(0, 0, 8006, list(range(2003)))
    admission.refresh(np.array([0, -1, -2]))
    held = admission.headroom_pages
    assert held > 0
    need = 4 + admission._candidate_headroom(1)
    assert admission.allows(1, held + need, 4)
    assert not admission.allows(1, held + need - 1, 4)


def test_bucket_curves_follow_their_own_answers_and_fall_back_to_the_map():
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), 40,
                         [64], [STOP], 4, expected_tokens=[8])
    decoder.answer_slots = {0: np.arange(40)}
    prefix_lengths = [8] * 20 + [256] * 19 + [1024]
    for a in range(39):
        decoder.lengths[0][a] = decoder.written[0][a] = 4 if a < 20 else 60
    cost = DecodeCost.from_specs(QWEN3_4B_FP8, DEVICES["h100-sxm"])
    admission = HeadroomAdmission(decoder, prefix_lengths, 6, {}, [0] * 40,
                                  lambda tokens: pages_for(tokens, 4), cost,
                                  bucket_answers=10)
    admission.refresh(np.array([-2] * 39 + [-1]))
    short = admission.bucket_curves[(0, 3)]
    long = admission.bucket_curves[(0, 8)]
    alone = admission.bucket_curves[(0, 10)]
    whole = answer_survival(decoder.lengths[0], decoder.written[0], 8, 64)
    assert short[10] < 0.3 < whole[10] < 0.6 < long[10]
    np.testing.assert_allclose(alone, whole)
    sparse = HeadroomAdmission(decoder, prefix_lengths, 6, {}, [0] * 40,
                               lambda tokens: pages_for(tokens, 4), cost,
                               bucket_answers=25)
    sparse.refresh(np.array([-2] * 39 + [-1]))
    np.testing.assert_allclose(sparse.bucket_curves[(0, 3)], whole)


def test_pages_have_value_to_waiting_documents_before_any_answer_runs():
    _, admission = headroom_admission(num_documents=4)
    admission.refresh(np.array([-1, -1, -1, -1]))
    assert admission.work.tokens == 0
    assert admission.page_value > 0


def test_a_borrowed_prefix_is_not_counted_twice_in_attention_pairs():
    from quail.backends.quail.executor.stage_executor import StageExecutor

    keys = ("passes", "decode_rows", "replay_tokens", "prefill_tokens", "pairs",
            "kv_read", "sol_s")
    executor = SimpleNamespace(
        prefix_lengths=[10, 10], decode_cost=None, pass_work=dict.fromkeys(keys, 0),
        borrowing=SimpleNamespace(shared=lambda a: 4 if a == 0 else 0),
        _entry_rows=lambda a, j, start, end, carried: (10 if carried else 0) + 2)
    # document 0 computes tokens 4..11 over 4 borrowed ones: keys 5..12
    StageExecutor._count_pass(executor, [(0, 0, 0, 1, True)], 8, None, ())
    assert executor.pass_work["pairs"] == sum(range(5, 13))
    assert executor.pass_work["kv_read"] == 4
    # a later suffix of document 1 follows its 10 prompt tokens: keys 11 and 12
    StageExecutor._count_pass(executor, [(1, 1, 0, 1, False)], 2, None, ())
    assert executor.pass_work["pairs"] == sum(range(5, 13)) + 11 + 12
