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

from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.operators.classify import ClassifyPart
from quail.backends.quail.executor.operators.filter import ConditionGate, FilterPart
from quail.backends.quail.executor.operators.map import MapPart
from quail.backends.quail.executor.parts import gated_stages
from quail.backends.quail.executor.stages import Stage, run_stages
from quail.catalog.tokens import prefix_tree

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
        rows = chunk.final_indices.tolist()
        out = torch.zeros(len(rows), VOCABULARY)
        out[torch.arange(len(rows)),
            torch.tensor([next_token(hashes[row]) for row in rows])] = 1.0
        return out


    def _whole(self, meta):
        """The unified call as one sequence per row, shared pages first.

        A row that reads its group's shared pages in a second call
        reads them, then its own pages, as one logical sequence.
        """
        unified, shared = meta["unified"], meta["shared"]
        if shared is None:
            return unified
        self.shared_reads += 1
        cu = unified["cu_q"].tolist()
        sequence = {cu[i]: i for i in range(len(cu) - 1) if cu[i + 1] - cu[i] == 1}
        table = [list(row) for row in unified["table"].tolist()]
        used = unified["used"].tolist()
        rows, groups = shared["rows"].tolist(), shared["cu_q"].tolist()
        tokens = shared["used"].tolist()
        for g, pages in enumerate(shared["table"].tolist()):
            pages = pages[:-(-tokens[g] // self.page)]
            for row in rows[groups[g]:groups[g + 1]]:
                table[sequence[row]] = pages + table[sequence[row]]
                used[sequence[row]] += tokens[g]
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
        name=name, prompt_token_parts=((), tuple(tail)), max_answer_tokens=limit))


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
        SimpleNamespace(spec=SimpleNamespace(name="label")), ids,
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


@pytest.mark.parametrize("path", ["unified", "tree"])
@pytest.mark.parametrize("lag", [0, 3])
@pytest.mark.parametrize("planned", ["unified", "tree"])
def test_maps_alone_decode_as_branches_that_share_the_document(monkeypatch, path,
                                                              lag, planned):
    cpu_staging(monkeypatch)
    rng = np.random.default_rng(1)
    # every document starts with the same preamble, longer than a page,
    # so a branch may borrow it from another document's branch
    preamble = list(range(1, 21))
    documents = [preamble + rng.integers(20, 900, size=int(n)).tolist()
                 for n in rng.integers(3, 60, size=10)]
    tails = [[500, 501, 502], [600, 601], [650, 651, 652, 653]]
    limits = [6, 4, 5]
    setup = Setup(path=path, pages=300, budget=200)
    arena = make_arena(setup)
    model = TokenModel(arena)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk,
                             tree_attention=True, canvas_ids=())
    torch_ = events_torch(lag)
    ids = list(range(len(documents)))
    maps = MapPart(None, [map_node(f"m{q}", tail, limit, planned)
                          for q, (tail, limit) in enumerate(zip(tails, limits))],
                   ids, 0)
    prefixes, keys, tree, dependencies = maps.branch_inputs(
        documents, [("d", i) for i in ids],
        prefix_tree(documents, setup.page_tokens), setup.page_tokens, 0)
    decoder = MapDecoder(torch_, torch.eye(VOCABULARY), len(prefixes), limits,
                         [STOP], setup.page_tokens)
    maps.decoder = decoder
    stats = {}
    run_stages(torch_, arena, pipeline, gated_stages([maps]), prefixes, setup.budget,
               anchor_keys=keys, attention_mode=path, prefix_tree=tree,
               dependencies=dependencies,
               decoder=decoder, stats=stats, gpu_timing=True)
    for q, (tail, limit) in enumerate(zip(tails, limits)):
        assert maps.asked[q].all()
        got = [answer.tolist() for answer in decoder.answers(
            q, maps._slots(q, ids))]
        assert got == [greedy(document + tail, limit) for document in documents]
    # the branches of a document borrow its prompt's whole pages, and
    # read them once together on a unified pass when the planner chose tree
    assert stats["borrowed_tokens"] > 0
    assert (path == "unified" and planned == "tree") == (model.shared_reads > 0)
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
            assert all(decoder.lengths[q][maps.slots[q, anchor]] >= 0
                       for q in range(len(tails)))
            survivors.append(anchor)
        arena.free_key(keys[anchor])

    stats = {}
    run_stages(torch_, arena, pipeline, stages, prefixes, setup.budget,
               anchor_keys=keys, prefix_tree=tree, dependencies=dependencies,
               decoder=decoder, on_settled=settled, stats=stats,
               limit=limit, survival=1.0)
    assert len(survivors) == (labels.count("keep") if limit is None else limit)
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
