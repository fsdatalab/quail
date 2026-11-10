"""AI.MAP's decode rows on a model that checks every KV read.

The model gives each row a hash of its token and of the KV it reads,
as tests/helpers/kv_checker.py does, and chooses the row's next token
from that hash. An answer comes out right only if every decode row
read its own prompt and answer so far at the right positions and
wrote its KV where the next row reads it.
"""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from helpers.conditions import condition as sql_condition
from helpers.fakes import cpu_staging, fake_pipeline
from helpers.kv_checker import HashModel, Setup, chain, cpu_torch, make_arena

from quail.backends.quail.executor.admission import pages_for
from quail.backends.quail.executor.answer_admission import (
    HeadroomAdmission,
    _allocate,
    admission_cost,
    allocate_headroom,
)
from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.model_operator import gated_stages
from quail.backends.quail.executor.operators.classify import ClassifyOperator
from quail.backends.quail.executor.operators.filter import ConditionGate, FilterOperator
from quail.backends.quail.executor.operators.map import MapOperator
from quail.backends.quail.executor.stage_executor import StageExecutor
from quail.backends.quail.executor.stages import Stage
from quail.common.answers import answer_survival
from quail.common.decode import DecodeCost
from quail.common.tokens import prefix_tree
from quail.execution.answer_stops import answer_stop
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

    classification = ClassifyOperator(
        SimpleNamespace(spec=SimpleNamespace(name="label", aliases=("d",),
                                              output_columns=("d", "label"))), ids,
        SimpleNamespace(labels=labels, stages=[Stage(
            suffixes=[[702]], frame=[700, 701], single=True,
            readout=Bits(), decide=classify)]))

    condition = ConditionGate(SimpleNamespace(
        node_id="condition", tested_columns=("label",),
        predicate=sql_condition("label = 'keep'", {"label": pa.string()}, "d")),
        {"label": classification}, classification, None)
    return classification, condition


@pytest.mark.parametrize("budget,question,lag,path", [
    (40, None, 0, "unified"),
    (400, [700, 701], 3, "tree"),
])
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
    maps = MapOperator(None, [map_node("a", tails[0], limits[0]),
                          map_node("b", tails[1], limits[1])], ids, 0)
    parts = [maps]
    if question is not None:
        node = SimpleNamespace(question_token_ids=question, negated=False)
        parts = [FilterOperator(node, ids, Bits(), 0), maps]
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], None, setup.page_tokens, len(stages) - 1)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], setup.page_tokens)
    maps.decoder = decoder
    stats = {}
    StageExecutor(torch_, arena, pipeline, stages, prefixes, budget,
               anchor_keys=keys, attention_mode=path, prefix_tree=tree,
               decoder=decoder, dependencies=dependencies, stats=stats).run()
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


@pytest.mark.parametrize("pages,limit,lag", [(6, 2, 3), (300, None, 0)])
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
    maps = MapOperator(None, [map_node(f"m{q}", tail, answer_limit, "tree")
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
    StageExecutor(torch_, arena, pipeline, stages, prefixes, setup.budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, on_settled=settled, stats=stats,
               limit=limit, survival=1.0).run()
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


@pytest.mark.parametrize("num_maps,limit,pages", [
    (1, None, 300), (2, None, 6), (3, 2, 6)])
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
    question = SimpleNamespace(question_token_ids=[750, 751], negated=False)

    def run(parts, ids, limit=None):
        arena = make_arena(setup)
        model = TokenModel(arena)
        pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=())
        torch_ = events_torch(3)
        stages = gated_stages(parts)
        prefixes, keys = [documents[d] for d in ids], [("d", d) for d in ids]
        maps = next((part for part in parts if isinstance(part, MapOperator)), None)
        decoder, tree, dependencies = None, None, {}
        if maps is not None:
            prefixes, keys, tree, dependencies = maps.branch_inputs(
                prefixes, keys, tree, setup.page_tokens, len(stages) - 1)
            decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes),
                                 maps.answer_limits, [STOP], setup.page_tokens)
            maps.decoder = decoder
        every, _, _ = StageExecutor(
            torch_, arena, pipeline, stages, prefixes, setup.budget,
            anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
            decoder=decoder, limit=limit, survival=1.0).run()
        assert not arena.accounting.owned
        return every

    classification, condition = classification_and_condition(ids)
    maps = MapOperator(None, nodes, ids, 0)
    run([FilterOperator(question, ids, Bits()), classification, condition, maps],
        ids, limit)
    kept = np.flatnonzero(maps.asked[0]).tolist()
    fused = [{"id": ids[d], "label": classification.plan.labels[d], **{
        node.spec.name: maps.decoder.answers(q, maps._slots(q, [d]))[0].tolist()
        for q, node in enumerate(nodes)}} for d in kept]
    assert all(np.flatnonzero(asked).tolist() == kept for asked in maps.asked)

    every = run([FilterOperator(question, ids, Bits())], ids)
    survivors = [ids[d] for d in sorted(every[0]) if every[0][d][0]]
    classification, condition = classification_and_condition(survivors)
    run([classification], survivors)
    separate = [{"id": d, "label": classification.plan.labels[a]}
                for a, d in enumerate(survivors)
                if condition.gate(("d", d)) is not Stage.DROP]
    survivors = [row["id"] for row in separate]
    for node in nodes:
        maps = MapOperator(None, [node], survivors, 0)
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
            tokenizer=SimpleNamespace(stop_ids=[STOP], decode_batch=batch_decode)))


# every setting appears in some case
@pytest.mark.parametrize("num_maps,limit,lag,budget,stop_modulus,path", [
    (3, None, 3, 20, 0, "tree"),
    (1, 2, 0, 200, 19, "tree"),
    (2, 2, 0, 20, 0, "unified"),
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
    maps = MapOperator(None, [map_node(f"m{q}", tail, answer_limit, "tree")
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
    StageExecutor(torch_, arena, pipeline, stages, prefixes, budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats, limit=limit, attention_mode=path).run()
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


@pytest.mark.parametrize("budget,lag,window", [(20, 0, 5), (200, 3, None)])
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
    first = MapOperator(None, nodes[:2], ids, 0)
    second = MapOperator(None, nodes[2:], ids, 2)
    question = [750, 751]
    node = SimpleNamespace(question_token_ids=question, negated=False)
    parts = [first, FilterOperator(node, ids, Bits(), 0), second]
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = first.branch_inputs(
        documents, [("d", a) for a in ids], prefix_tree(documents, 4), 4, 0)
    prefixes, keys, tree, added = second.branch_inputs(prefixes, keys, tree, 4, 2)
    dependencies.update(added)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], 4)
    first.decoder = second.decoder = decoder
    stats = {}
    StageExecutor(torch_, arena, pipeline, stages, prefixes, setup.budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats).run()
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


def run_maps_under_eviction(monkeypatch, mode, num_maps, budget, lag, path):
    """Run maps over twelve documents in a 20-page pool.

    Returns:
        The run's stats, the token model, and each map's answers, after
        checking every answer equals greedy decoding.
    """
    cpu_staging(monkeypatch)
    monkeypatch.setenv("QUAIL_ANSWER_ADMISSION", mode)
    monkeypatch.setattr(__name__ + ".next_token", lambda h: (
        STOP if h % 7 == 0 else 10 + h % 900))
    rng = np.random.default_rng(23)
    documents = [[20, 21, 22, 23] + rng.integers(20, 400, size=4 + 4 * (i % 3)).tolist()
                 for i in range(12)]
    ids = list(range(len(documents)))
    tails = [[500, 501], [600, 601], [650, 651]][:num_maps]
    limits = [24, 20, 28][:num_maps]
    setup = Setup(path=path, pages=20, page_tokens=4, budget=budget)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=(),
                             tree_attention=True)
    maps = MapOperator(None, [map_node(f"m{q}", tail, answer_limit, "tree")
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
    StageExecutor(events_torch(lag), arena, pipeline, gated_stages([maps]), prefixes,
               budget, anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, stats=stats, attention_mode=path,
               decode_cost=DecodeCost.from_specs(QWEN3_4B_FP8,
                                                 DEVICES["h100-sxm"])).run()
    answers = []
    for q, (tail, answer_limit) in enumerate(zip(tails, limits)):
        got = [answer.tolist() for answer in decoder.answers(q, maps._slots(q, ids))]
        assert got == [greedy(documents[a] + tail, answer_limit) for a in ids]
        answers.append(got)
    assert not decoder.active
    assert not arena.accounting.owned
    return stats, model, answers


@pytest.mark.parametrize("mode,num_maps,budget,lag,path", [
    ("headroom", 3, 20, 0, "tree"),
    ("headroom=0", 1, 200, 3, "unified"),
])
def test_headroom_admission_answers_equal_greedy_decoding_under_eviction(
        monkeypatch, mode, num_maps, budget, lag, path):
    stats, model, _ = run_maps_under_eviction(monkeypatch, mode, num_maps, budget,
                                              lag, path)
    assert stats["answer_admission"] == mode
    work = stats["pass_work"]
    assert work["replay_tokens"] == stats["recomputed_tokens"]
    assert work["decode_rows"] == sum(len(slots) for slots in model.decode_slots)
    assert work["sol_s"] > 0


def test_a_documents_answers_admitted_one_at_a_time_equal_greedy_decoding(
        monkeypatch):
    # twelve documents, three maps each: map 0 is the document's own slot
    _, model, _ = run_maps_under_eviction(monkeypatch, "headroom", 3, 20, 2, "tree")
    slots = np.vstack((np.arange(12), np.arange(12, 36).reshape(12, 2).T))
    first_pass = {}
    for index, decoding in enumerate(model.decode_slots):
        for slot in decoding:
            first_pass.setdefault(slot, index)
    starts = [{first_pass[int(slot)] for slot in slots[:, document]}
              for document in range(12)]
    assert any(len(passes) > 1 for passes in starts)


def test_a_headroom_trace_is_written_only_once_the_run_returns(monkeypatch, tmp_path):
    monkeypatch.setenv("QUAIL_HEADROOM_TRACE", str(tmp_path))
    arenas = []
    built = make_arena

    def kept_arena(setup):
        arenas.append(built(setup))
        return arenas[-1]

    monkeypatch.setattr(__name__ + ".make_arena", kept_arena)
    run_maps_under_eviction(monkeypatch, "headroom", 3, 20, 2, "tree")
    assert not list(tmp_path.iterdir())
    (arena,) = arenas
    arena.write_trace_files()
    (path,) = tmp_path.iterdir()
    trace = json.loads(path.read_text())
    assert trace["rebuilds"] and trace["page_tokens"] == 4
    assert not arena.trace_files


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


def test_zero_headroom_admits_when_the_prompt_fits_but_waits_for_replays():
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


def test_a_new_document_leaves_running_headroom_alone_when_pages_are_ample():
    decoder, admission = headroom_admission(ended=[(2, 30)], prefix_tokens=8000)
    decoder.start(0, 0, 8006, list(range(2003)))
    admission.refresh(np.array([0, -1, -2]), free_pages=1000)
    held = int(admission._remaining(np.array([0]))[0])
    assert held > 0
    assert not admission.allows(1, 3, 4)
    for prompt_pages in (4, 1):
        assert admission.allows(1, 1000, prompt_pages)
        assert admission._remaining(np.array([0]))[0] == held
    admission.admit(1)
    assert admission._remaining(np.array([0]))[0] == held
    assert admission.headroom_pages + 4 <= 1000


def test_a_second_document_in_a_round_shrinks_the_first_ones_headroom_to_fit():
    decoder, admission = headroom_admission(prefix_tokens=8, limit=128,
                                            expected_tokens=64)
    admission.refresh(np.array([-1, -1, -1]), free_pages=30)
    assert admission.allows(0, 30, 4)
    admission.admit(0)
    first = admission.headroom_pages
    assert first == 30 - 4
    assert admission.allows(1, 26, 4)
    admission.admit(1)
    assert admission.headroom_pages < first
    assert admission.headroom_pages <= 30 - 8


def test_headroom_that_pays_off_only_after_two_pages_is_taken_whole_or_not_at_all():
    risk = np.array([[10.0, 10.0, 0.0], [5.0, 4.0, 4.0]])
    assert allocate_headroom(risk, 2).tolist() == [2, 0]
    assert allocate_headroom(risk, 1).tolist() == [0, 1]
    assert allocate_headroom(risk, 0).tolist() == [0, 0]


def test_headroom_searched_from_a_nearby_price_matches_a_search_from_scratch():
    rng = np.random.default_rng(3)
    for _ in range(50):
        # rows of falling risk, inf past each row's limit
        risk = np.cumsum(-rng.exponential(1.0, size=(40, 17)), axis=1)
        risk = risk - risk.min(axis=1, keepdims=True)
        limits = rng.integers(4, 17, size=40)
        risk[np.arange(17) > limits[:, None]] = np.inf
        budget = int(rng.integers(0, 200))
        chosen, price = _allocate(risk, budget)
        assert price is not None and chosen.sum() <= budget
        assert np.array_equal(allocate_headroom(risk, budget), chosen)
        for hint in (price / 3, price, price * 3):
            assert np.array_equal(_allocate(risk, budget, hint)[0], chosen)


def test_the_quick_bound_never_exceeds_the_rebuild_seconds_admission_adds():
    _, admission = headroom_admission(num_documents=60)
    rng = np.random.default_rng(4)

    def grids(rows):
        risk = np.cumsum(-rng.exponential(1.0, size=(rows, 17)), axis=1)
        risk = risk - risk.min(axis=1, keepdims=True)
        risk[np.arange(17) > rng.integers(4, 17, size=rows)[:, None]] = np.inf
        return risk

    for _ in range(50):
        admission.held_slots = np.arange(40)
        admission.held_base = np.zeros(40, dtype=np.int64)
        admission.held_risk = grids(40)
        budget = int(rng.integers(20, 200))
        admission._hold(admission._allocate_held(budget), budget)
        candidate = grids(int(rng.integers(1, 5)))
        left = budget - int(rng.integers(0, 20))
        extra, _ = admission_cost(admission.held_risk, admission.held_pages,
                                  candidate, left)
        bound = admission._least_extra(admission._held_bound(), candidate, left)
        assert bound <= extra + 1e-9


def test_admitting_a_document_costs_the_rebuild_risk_it_pushes_onto_held_answers():
    held = np.array([[100.0, 80.0, 0.0]])
    candidate = np.array([[3.0, 3.0, 3.0]])
    extra, chosen = admission_cost(held, np.array([2]), candidate, 1)
    assert extra == 80.0
    assert chosen.tolist() == [1, 0]
    extra, _ = admission_cost(held, np.array([2]), candidate, 2)
    assert extra == 0.0


def test_survival_curves_match_a_full_recount_after_answers_end_unseen():
    decoder, admission = headroom_admission(num_documents=60, limit=32,
                                            expected_tokens=8)
    admission.bucket_answers = 1
    stages = np.full(60, -1)
    admission.refresh(stages, free_pages=1000)
    rng = np.random.default_rng(5)
    for a in range(40):
        decoder.start(a, 0, 14, [4 * a, 4 * a + 1, 4 * a + 2, 4 * a + 3])
        decoder.written[0][a] = rng.integers(0, 20)
    admission.refresh(stages, free_pages=1000)
    # half end as a settle ends them; slot 50 starts and ends between refreshes
    for a in [*range(0, 40, 2), 50]:
        decoder.lengths[0][a] = decoder.written[0][a] = rng.integers(1, 32)
        decoder.map[a] = -1
        decoder.ended_log.append(np.array([a]))
    decoder.written[0][1] += 3
    admission.refresh(stages, free_pages=1000)
    recount = HeadroomAdmission(decoder, [8] * 60, 6, {}, [0] * 60,
                                lambda tokens: pages_for(tokens, 4),
                                admission.decode_cost, bucket_answers=1)
    recount.refresh(stages, free_pages=1000)
    assert np.array_equal(admission.map_counts, recount.map_counts)
    assert np.allclose(admission.whole, recount.whole)
    assert np.allclose(admission.curves, recount.curves)


def test_eviction_takes_remaining_tokens_from_each_answers_length_bucket():
    # short prompts end after 3 tokens, long ones run to the limit; two
    # answers that wrote 2 tokens each differ only in their prompt length
    num_documents = 20
    decoder = MapDecoder(events_torch(0), torch.eye(VOCABULARY), num_documents,
                         [24], [STOP], 4, expected_tokens=[12])
    decoder.answer_slots = {0: np.arange(num_documents)}
    prefixes = [8] * 10 + [2000] * 10
    for a in [*range(2, 10), *range(12, 20)]:
        decoder.lengths[0][a] = decoder.written[0][a] = 3 if a < 10 else 24
    admission = HeadroomAdmission(decoder, prefixes, 6, {}, [0] * num_documents,
                                  lambda tokens: pages_for(tokens, 4),
                                  choose_headroom=False, bucket_answers=1)
    decoder.start(0, 0, 14, [0, 1, 2, 3])
    decoder.start(10, 0, 2006, [4, 5, 6, 7])
    decoder.written[0][[0, 10]] = 2
    assert decoder.eviction_order()[0] == 0
    assert admission.eviction_order()[0] == 10


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


class LetterTokenizer:
    """Write token t as the letter t mod 26, so a test can read answers as text."""

    @staticmethod
    def decode_batch(batch):
        return ["".join(chr(97 + int(token) % 26) for token in tokens)
                for tokens in batch]


def test_a_map_answer_stops_once_its_test_is_decided(monkeypatch):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(3)
    documents = [rng.integers(20, 900, size=int(n)).tolist()
                 for n in rng.integers(3, 30, size=16)]
    tail, limit = [500, 501, 502], 8
    wanted = [greedy(document + tail, limit) for document in documents]
    text = LetterTokenizer.decode_batch(wanted)
    # the letter most answers start with; the others are rejected at once
    first = max(set(found[0] for found in text if found), key=[
        found[:1] for found in text].count)
    test = sql_condition(f"a LIKE '{first}%'", {"a": pa.string()}, "d")
    setup = Setup(path="unified", pages=200, budget=400)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=())
    torch_ = events_torch(2)
    ids = list(range(len(documents)))
    node = map_node("a", tail, limit)
    node.stop_tests, node.stop_when_true = (test,), True
    maps = MapOperator(None, [node], ids, 0)
    stages = gated_stages([maps])
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids], None, setup.page_tokens, 0)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), [limit],
                         [STOP], setup.page_tokens)
    decoder.tokenizer = LetterTokenizer()
    decoder.answer_stops[0] = answer_stop(node)
    maps.decoder = decoder
    StageExecutor(torch_, arena, pipeline, stages, prefixes, 400, anchor_keys=keys,
               prefix_tree=tree, decoder=decoder, dependencies=dependencies).run()
    got = [answer.tolist() for answer in decoder.answers(0, maps._slots(0, ids))]
    # an answer the test settles stops after its first token, either way
    assert got == [want[:1] for want in wanted]
    assert decoder.stopped_early[0] == sum(1 for want in wanted if want)
    assert decoder.rows_fed[0] < sum(map(len, wanted))
    assert not decoder.active
    assert not arena.accounting.owned

