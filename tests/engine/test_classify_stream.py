"""Test classification within filter and join pipelines."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from helpers.conditions import condition as sql_condition
from helpers.fakes import (
    DOC,
    FRAME,
    PARTNER,
    QUESTION,
    SETTINGS,
    FakeModel,
    cpu_arena,
    fake_pack,
    fake_pipeline,
    fake_readout,
    fake_torch,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import chunk as chunk_mod
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.execution.finished import reporting
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    AiScore,
    ClassifySpec,
    Filter,
    PhysicalGraph,
    PortRef,
    Scan,
    ScoreSpec,
    input_ports,
)
from quail.specs import DEVICES, MODELS

# the classification frame is written after the document like a join
# frame; the cue is the tail's last token
CLASSIFY_FRAME = 4500
CUE = 5000
# one-token labels, read at the cue row
LABEL_IDS = ((13,), (14,))
TARGETS = [13, 14]


# the anchor note (its frame) and partner label of a classification
# of joined rows
NOTE = 4600
PARTNER_LABEL = 4700


class LabelingModel(FakeModel):
    """Answer filter and join suffixes from truth, and label documents."""

    def __init__(self, filter_truth, join_truth, label_truth,
                 joined_truth=None):
        super().__init__(filter_truth, join_truth)
        self.label_truth = label_truth
        self.joined_truth = joined_truth

    def forward_chunk(self, chunk):
        rows = []
        for spec in chunk.specs:
            for index, suffix in enumerate(spec["suffixes"]):
                head = suffix[0]
                if suffix[-1] == CUE and head != CUE:
                    # a joined row's block: the partner document, the
                    # question, then the cue, whose row is read; the partner
                    # label is in the frame
                    partner = head - PARTNER
                    wanted = LABEL_IDS[self.joined_truth[(spec["key"][1], partner)]]
                    for depth in range(int(spec["read_rows"][index])):
                        rows.append(np.asarray(
                            [-1.0 if token == wanted[depth] else -5.0
                             for token in TARGETS], np.float32))
                elif head == CUE:
                    # each row's log probabilities over the targets: the
                    # document's label token at that depth wins
                    wanted = LABEL_IDS[self.label_truth[spec["key"][1]]]
                    for depth in range(len(suffix)):
                        rows.append(np.asarray(
                            [-1.0 if token == wanted[depth] else -5.0
                             for token in TARGETS], np.float32))
                elif head >= FRAME:
                    rows.append(0)
                elif head >= PARTNER:
                    rows.append(self.join_truth[spec["key"]][head - PARTNER])
                else:
                    document = spec["key"][1]
                    rows.append(self.filter_truth[document][head - QUESTION])
            heads = [suffix[0] for suffix in spec["suffixes"]]
            # with a filter, a document's prefix packs at its stage
            if (self.filter_truth and spec["prefix"] is not None and heads
                    and heads[0] != QUESTION):
                raise AssertionError(
                    "a document packed its prefix past the filter stage")
        self.launched.append(chunk.specs)
        return rows


def fused_graph(*, classify_only=False):
    """Build two classifications with label filters between them.

    Args:
        classify_only: Leave out the AI filter before the classifications
            and the join after them.
    """
    spec = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_rows=14, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="letters", share_prefixes=True)
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiFilter(
            node_id="filter:r", inputs=input_ports((PortRef("input:r", "rows:r"),)),
            alias="r", arena_writes=True, question_token_ids=(QUESTION,),
            written_position=0, question_tokens=1, preamble_tokens=0,
            selectivity=0.8, expected_rows=14 * 0.8),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("filter:r", "rows:r"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(
            node_id="label:r",
            inputs=input_ports((PortRef("classify:r", "rows:r"),)),
            predicate=sql_condition(
                "topic IN ('a')", {"topic": pa.string()}, "r", "reviews"),
            aliases=("r",),
            selectivity=0.5, written_position=1),
        Filter(
            node_id="label:again",
            inputs=input_ports((PortRef("label:r", "rows:r"),)),
            predicate=sql_condition(
                "topic IN ('a')", {"topic": pa.string()}, "r", "reviews"),
            aliases=("r",),
            selectivity=1.0, written_position=2),
        AiClassify(
            node_id="classify:kind",
            inputs=input_ports((PortRef("label:again", "rows:r"),)),
            backend_name="quail", model="qwen3-4b-fp8",
            spec=spec.replace(name="kind", labels=("b", "a"),
                         prompt_token_parts=((), (CLASSIFY_FRAME + 1, CUE)))),
        AiJoin(
            node_id="group:0", anchor="r", anchor_resident="filter",
            inputs=input_ports((PortRef("classify:kind", "rows:r"),
                                PortRef("input:p", "rows:p"))),
            written_position=2, exec_index=0, partners=("p",), semantics="full",
            selectivity=0.5, expected_rows=1, anchor_frame_tokens=1,
            pair_tail_tokens=0, tuple_tokens=0, pairs_from="",
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),), tail_token_ids=()),
    ]
    if classify_only:
        nodes = [node for node in nodes if node.node_id not in (
            "input:p", "filter:r", "group:0")]
        nodes = [node.replace(inputs=input_ports((PortRef("input:r", "rows:r"),)))
                 if node.node_id == "classify:r" else node for node in nodes]
        return PhysicalGraph(tuple(nodes), PortRef("classify:kind", "rows:r"))
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "rows:r"))


@pytest.mark.parametrize("classify_only", [False])
def test_streamed_classification_labels_survivors_with_their_kv_resident(
        monkeypatch, classify_only):
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    num_documents, num_partners = 14, 4
    documents = {
        "r": [[DOC + d] * (10 + 2 * d) for d in range(num_documents)],
        "p": [[PARTNER + i] * 20 for i in range(num_partners)],
    }
    filter_truth = [[d % 3 != 0] for d in range(num_documents)]
    label_truth = [d % 2 for d in range(num_documents)]           # even: "a"
    join_truth = {("r", d): [(d + i) % 3 == 0 for i in range(num_partners)]
                  for d in range(num_documents)}
    model = LabelingModel([] if classify_only else filter_truth,
                          join_truth, label_truth)
    events = []
    monkeypatch.setattr(
        "quail.backends.quail.executor.operators.classify.answer_sink",
        lambda: lambda event: events.append((len(model.launched), event)))
    torch = fake_torch()
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpus=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=fake_readout(),
        answer_rows=object(), chunk_tokens=120)
    def submit(rows, rows_per_answer=None):
        # (answers, rows, targets), padded; a frame entry's one row is
        # not read and carries no log probabilities
        if rows_per_answer is None:
            rows_per_answer = [1] * len(rows)
        padded = np.full((len(rows_per_answer), 1, len(TARGETS)), np.nan,
                         np.float32)
        start = 0
        for answer, count in enumerate(rows_per_answer):
            for offset in range(count):
                row = rows[start + offset]
                padded[answer, offset] = (row if hasattr(row, "shape")
                                          else np.zeros(len(TARGETS)))
            start += count
        return padded

    execution.loaded_model.label_readout = fake_readout(
        targets=np.asarray(TARGETS), rows=1,
        dtype=np.dtype((np.float32, (1, len(TARGETS)))), submit=submit)
    state = {
        "gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                               execution=execution, spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "documents": documents,
    }
    result = execute_single_graph(
        state, SETTINGS, fused_graph(classify_only=classify_only))
    assert not arena.accounting.owned

    passed = [d for d in range(num_documents) if classify_only or filter_truth[d][0]]
    batches = [(step, event) for step, event in events
               if event["node"] == "classify:r"]
    assert sorted(row for _, event in batches for row in event["rows"]) == passed
    assert batches[0][0] < len(model.launched)
    accepted = [d for d in passed if label_truth[d] == 0]
    matched = [d for d in accepted if any(join_truth[("r", d)])]
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    labels = outputs[PortRef("classify:r", "rows:r")]
    assert labels.column("r").to_pylist() == passed
    assert labels.column("topic").to_pylist() == [
        ("a", "b")[label_truth[d]] for d in passed]
    assert outputs[PortRef("classify:r", "rows:r")].column("r").to_pylist() \
        == passed
    assert outputs[PortRef("label:r", "rows:r")].column("r").to_pylist() \
        == accepted
    assert metrics["label:again"]["input_rows"] == len(accepted)
    later = outputs[PortRef("classify:kind", "rows:r")]
    assert later.to_pydict() == {
        "r": accepted, "kind": ["b"] * len(accepted),
        "topic": ["a"] * len(accepted)}
    prefill = sum(len(documents["r"][d]) for d in range(num_documents))
    if classify_only:
        assert metrics["classify:r"]["fresh_tokens"] == prefill + 2 * num_documents
    else:
        assert outputs[PortRef("group:0", "rows:r")].column("r").to_pylist() \
            == matched
        assert metrics["filter:r"]["fresh_tokens"] == prefill + num_documents
        assert metrics["classify:r"]["fresh_tokens"] == 2 * len(passed)
    prefills = [entry["key"][1] for chunk in model.launched for entry in chunk
                if entry["prefix"] is not None]
    assert sorted(prefills) == list(range(num_documents))
    assert metrics["classify:r"]["evaluated_documents"] == len(passed)
    assert metrics["classify:r"]["suffix_tokens"] == len(passed)
    assert metrics["classify:kind"]["fresh_tokens"] == 2 * len(accepted)
    assert metrics["classify:kind"]["evaluated_documents"] == len(accepted)
    # some chunk held documents at the sequence's stage beside documents
    # at the classification's or the join's
    heads = [{suffix[0] for spec in specs for suffix in spec["suffixes"]}
             for specs in model.launched]
    if not classify_only:
        assert any(QUESTION in chunk and (CUE in chunk or chunk & {FRAME})
                   or CLASSIFY_FRAME in chunk and FRAME in chunk
                   for chunk in heads)


def joined_graph(two_joins=False):
    """Build a join followed by classification of its retained pairs."""
    spec = ClassifySpec(
        name="stance", aliases=("r", "p"), query_template="", arguments=(),
        expected_rows=3, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="letters",
        join_layout=((NOTE,), (PARTNER_LABEL,)))
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiJoin(
            node_id="group:0", anchor="r", anchor_resident="fresh",
            inputs=input_ports((PortRef("input:r", "rows:r"),
                                PortRef("input:p", "rows:p"))),
            written_position=2, exec_index=0, partners=("p",), semantics="full",
            selectivity=0.5, expected_rows=3, anchor_frame_tokens=1,
            pair_tail_tokens=0, tuple_tokens=0, pairs_from="",
            frame_token_ids=(FRAME,), label_token_ids=(("p", ()),), tail_token_ids=()),
        AiClassify(
            node_id="classify:rp",
            inputs=input_ports((PortRef("group:0", "join_answers:2"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec),
    ]
    if two_joins:
        nodes.insert(2, Scan(node_id="input:q", alias="q", input_id="q"))
        join = nodes[3]
        nodes.insert(4, join.replace(
            node_id="group:1", written_position=3, exec_index=1, partners=("q",),
            inputs=input_ports((PortRef("group:0", "rows:r"),
                                PortRef("input:q", "rows:q"))),
            frame_token_ids=(FRAME + 1,), label_token_ids=(("q", ()),)))
    return PhysicalGraph(tuple(nodes), PortRef("classify:rp", "rows"))


@pytest.mark.parametrize("two_joins", [True])
def test_a_classification_of_joined_rows_runs_after_its_join_on_the_anchors_kv(
        monkeypatch, two_joins):

    graph = joined_graph(two_joins)
    pipelines = {tuple(
        member.node_id for member in pipeline)
        for pipeline in graph.pipelines().values()}
    assert pipelines == {("group:0", "group:1", "classify:rp") if two_joins
                      else ("group:0", "classify:rp")}

    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    documents = {"r": [[DOC + d] * (5 + d) for d in range(3)],
            "p": [[PARTNER + d, PARTNER + d] for d in range(2)]}
    # r0 pairs with both partners, r1 with p1 only, r2 with none
    join_truth = {("r", 0): [1, 1], ("r", 1): [0, 1], ("r", 2): [0, 0]}
    if two_joins:
        # the second join keeps q0; the classification reads p1 for r1
        documents["q"] = [[PARTNER + 2] * 2]
        join_truth = {key: values + [1] for key, values in join_truth.items()}
    joined_truth = {(0, 0): 0, (0, 1): 1, (1, 1): 0}
    model = LabelingModel([], join_truth, [], joined_truth)
    torch = fake_torch()
    torch.nn = SimpleNamespace(functional=None)
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpus=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=fake_readout(),
        answer_rows=object(), chunk_tokens=120)
    from quail.backends.quail.executor import label_stages as label_stages_module

    class FakeReadout:
        trailing_rows = 1
        normalize = False

        def __init__(self, targets, rows):
            self.targets = np.asarray(targets)
            self.rows = rows
            self.dtype = np.dtype((np.float32, (rows, len(targets))))

        def submit(self, rows, rows_per_answer=None):
            if rows_per_answer is None:
                rows_per_answer = [1] * len(rows)
            padded = np.full((len(rows_per_answer), self.rows,
                              len(self.targets)), np.nan, np.float32)
            start = 0
            for answer, count in enumerate(rows_per_answer):
                for offset in range(count):
                    padded[answer, offset] = rows[start + offset]
                start += count
            return padded

        def result(self, rows):
            return rows

    monkeypatch.setattr(label_stages_module, "full_output_head",
                        lambda model: SimpleNamespace(shape=(1, 1),
                                                      dtype="fake"))
    monkeypatch.setattr(
        label_stages_module, "AsyncLabelLogprobs",
        lambda torch, F, head, targets, rows, normalize: FakeReadout(
            targets, rows))
    state = {
        "gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                               execution=execution, spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "documents": documents,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    assert outputs[PortRef("group:0", "rows:r")].column("r").to_pylist() \
        == [0, 1]
    labels = outputs[PortRef("classify:rp", "rows")]
    assert labels.column("r").to_pylist() == [0, 0, 1]
    assert labels.column("p").to_pylist() == [0, 1, 1]
    assert labels.column("stance").to_pylist() == ["a", "b", "a"]
    # the join packed the anchors once; the classification packed, per
    # anchor kept, its note and the partner label and, per pair, the
    # partner document, the question, and the cue: no second prefill
    block = 2 + 1
    assert metrics["classify:rp"]["fresh_tokens"] == 2 * (1 + 1) + 3 * (block + 1)
    assert metrics["classify:rp"]["evaluated_document_pairs"] == 3
    # the join packed every anchor, its frame, and both partners' suffixes
    assert sum(value["fresh_tokens"] for name, value in metrics.items()
               if name.startswith("group:")) == sum(map(len, documents["r"])) \
        + 3 * 1 + 6 * 2 + (2 * (1 + 2) if two_joins else 0)


def test_a_filter_after_a_classification_asks_over_the_resident_kv(monkeypatch):
    # the second filter's question follows the document in KV, after
    # the label test; the next classification reads the survivors
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    num_documents = 14
    documents = {"r": [[DOC + d] * (10 + 2 * d) for d in range(num_documents)]}
    filter_truth = [[d % 3 != 0, d % 4 != 0] for d in range(num_documents)]
    label_truth = [d % 2 for d in range(num_documents)]
    model = LabelingModel(filter_truth, {}, label_truth)
    torch = fake_torch()
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpus=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=fake_readout(),
        answer_rows=object(), chunk_tokens=120)

    def submit(rows, rows_per_answer=None):
        # a frame entry's one row is not read and carries no log
        # probabilities
        padded = np.full((len(rows), 1, len(TARGETS)), np.nan, np.float32)
        for answer, row in enumerate(rows):
            padded[answer, 0] = (row if hasattr(row, "shape")
                                 else np.zeros(len(TARGETS)))
        return padded

    execution.loaded_model.label_readout = fake_readout(
        targets=np.asarray(TARGETS), rows=1,
        dtype=np.dtype((np.float32, (1, len(TARGETS)))), submit=submit)
    graph = fused_graph(classify_only=True)
    nodes = [node for node in graph.nodes if node.node_id != "classify:kind"]
    nodes.insert(0, Scan(node_id="input:r", alias="r", input_id="r"))
    nodes = list(dict.fromkeys(nodes))
    spec = next(node for node in nodes if node.node_id == "classify:r").spec
    nodes = [
        node for node in nodes if node.node_id in ("input:r", "label:r")
        or isinstance(node, AiClassify)]
    nodes = [
        nodes[0],
        AiFilter(
            node_id="filter:r", inputs=input_ports((PortRef("input:r", "rows:r"),)),
            alias="r", arena_writes=True, question_token_ids=(QUESTION,),
            written_position=0, question_tokens=1, preamble_tokens=0,
            selectivity=0.8, expected_rows=14 * 0.8),
        nodes[1].replace(inputs=input_ports((PortRef("filter:r", "rows:r"),))),
        nodes[2],
        AiFilter(
            node_id="filter:again",
            inputs=input_ports((PortRef("label:r", "rows:r"),)),
            alias="r", arena_writes=True, question_token_ids=(QUESTION + 1,),
            written_position=2, question_tokens=1, preamble_tokens=0,
            selectivity=0.8, expected_rows=4.0),
        AiClassify(
            node_id="classify:kind",
            inputs=input_ports((PortRef("filter:again", "rows:r"),)),
            backend_name="quail", model="qwen3-4b-fp8",
            spec=spec.replace(name="kind", labels=("b", "a"),
                         prompt_token_parts=((), (CLASSIFY_FRAME + 1, CUE)))),
    ]
    graph = PhysicalGraph(tuple(nodes), PortRef("classify:kind", "rows:r"))

    assert [member.node_id for member in graph.pipelines()["filter:r"]] == [
        "filter:r", "classify:r", "label:r", "filter:again", "classify:kind"]
    state = {
        "gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                               execution=execution, spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "documents": documents,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    outputs, metrics = result["_outputs"], result["node_metrics"]
    passed = [d for d in range(num_documents) if filter_truth[d][0]]
    accepted = [d for d in passed if label_truth[d] == 0]
    asked_again = [d for d in accepted if filter_truth[d][1]]
    assert outputs[PortRef("label:r", "rows:r")].column("r").to_pylist() == accepted
    assert outputs[PortRef("filter:again", "rows:r")].column("r").to_pylist() \
        == asked_again
    assert outputs[PortRef("classify:kind", "rows:r")].column("r").to_pylist() \
        == asked_again
    # the second filter fed one question token per accepted document
    # over its resident KV, and every document was packed once
    assert metrics["filter:again"]["fresh_tokens"] == len(accepted)
    assert metrics["filter:again"]["evaluated_documents"] == len(accepted)
    assert metrics["classify:r"]["fresh_tokens"] == 2 * len(passed)
    assert metrics["classify:kind"]["fresh_tokens"] == 2 * len(asked_again)
    prefills = [entry["key"][1] for chunk in model.launched for entry in chunk
                if entry["prefix"] is not None]
    assert sorted(prefills) == list(range(num_documents))
    assert not arena.accounting.owned


# a score's tail and cue; the model answers every score suffix alike
SCORE_TAIL = 4800


class ScoringModel(LabelingModel):
    """Answer label and join suffixes from truth, and every score suffix."""

    def forward_chunk(self, chunk):
        if all(suffix[-1] == SCORE_TAIL + 1 for spec in chunk.specs
               for suffix in spec["suffixes"]):
            self.launched.append(chunk.specs)
            return [len(suffix) % 7 / 7 for spec in chunk.specs
                    for suffix in spec["suffixes"]]
        return super().forward_chunk(chunk)


def score_graph():
    """Score each document of r alone, then each pair of r and p."""
    def spec(name, aliases, parts):
        return ScoreSpec(name=name, aliases=aliases, query_template="",
                         arguments=tuple((alias, "text") for alias in aliases),
                         expected_rows=4, estimated_seconds=0.0,
                         prompt_token_parts=parts)

    tail = (SCORE_TAIL, SCORE_TAIL + 1)
    return PhysicalGraph((
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiScore(node_id="score:r", backend_name="quail", model="qwen3-4b-fp8",
                inputs=input_ports((PortRef("input:r", "rows:r"),)),
                spec=spec("fun", ("r",), ((), tail))),
        AiScore(node_id="score:rp", backend_name="quail", model="qwen3-4b-fp8",
                inputs=input_ports((PortRef("input:r", "rows:r"),
                                    PortRef("input:p", "rows:p"))),
                spec=spec("match", ("r", "p"), ((), (CLASSIFY_FRAME,), tail))),
    ), PortRef("score:rp", "rows"))


def classify_graph():
    """Classify each document of r alone."""
    nodes = fused_graph(classify_only=True).nodes
    (scan,) = [node for node in nodes if node.node_id == "input:r"]
    (classify,) = [node for node in nodes if node.node_id == "classify:r"]
    classify = classify.replace(inputs=input_ports((PortRef("input:r", "rows:r"),)))
    return PhysicalGraph((scan, classify), PortRef("classify:r", "rows:r"))


def filter_graph(num_filters):
    """Ask each document of r one filter question after another."""
    nodes = [Scan(node_id="input:r", alias="r", input_id="r")]
    source = PortRef("input:r", "rows:r")
    for index in range(num_filters):
        nodes.append(AiFilter(
            node_id=f"filter:{index}", inputs=input_ports((source,)), alias="r",
            arena_writes=num_filters > 1, question_token_ids=(QUESTION + index,),
            written_position=index, question_tokens=1, preamble_tokens=0,
            selectivity=0.7, expected_rows=10.0))
        source = PortRef(f"filter:{index}", "rows:r")
    return PhysicalGraph(tuple(nodes), source)


def chunk_compositions(launched) -> list:
    """Each packed chunk's entries: key, prefix length, and suffix tokens."""
    return [[(repr(entry["key"]),
              None if entry["prefix"] is None else len(entry["prefix"]),
              int(entry["f"]), int(entry.get("start", 0)),
              repr(entry.get("borrow_key")),
              tuple(tuple(int(token) for token in suffix)
                    for suffix in entry["suffixes"]))
             for entry in chunk] for chunk in launched]


# per region: chunks packed, a digest of their compositions, each
# node's fresh tokens, and a digest of the output tables, recorded on
# the executor before it streamed documents between operators
RECORDED_CHUNKS = {
    "filter": (3, "60593d47291287dce3509ac1afa1ef11b6892d5ddc5bb7f82a0d4406cc4ea7b3",
               {"filter:0": 336},
               "eea6f194bfc2d904e270bf21ef228f481b26fb3edeaa9642d253fca105c59441"),
    "filters": (5, "532361ce9c62647069dace66789bb277e036655251f73bdd6c0857d34f063089",
                {"filter:0": 336, "filter:1": 9},
                "cc574da4ec685bd89fe4b37a4ce345cc9bc94cbe69fae515053c64900202c3d5"),
    "fused": (8, "5569260ca2d19dc490f28f284efb652ce7037c2d98d196cfc0d246df7837e58c",
              {"classify:kind": 8, "classify:r": 18, "filter:r": 336, "group:0": 92},
              "d1a18ac0ec0b19778fd2714f6c9224c8ac60ce742537066800ac1a4e1e4b3051"),
    "joined": (12, "55f1c193ef72a23fbfc8dc00d047765ea31d1de12af325e43122dae6e6a66abd",
               {"classify:rp": 171, "group:0": 644, "group:1": 42},
               "fd3e330ee7483b08ce830d7f9c3f4311ec75f5fc84baa60c26f5b8d2586f8653"),
    "scores": (11, "0dabe20b32fbddbb6b3fe3b141a9b8b413413b2e7fbf79c72d1b56da5af9403b",
               {"score:r": 350, "score:rp": 756},
               "5b5d400f4a5835fc1c148132ab952e937fd79ebe2a8a9b43c8d8d1111b78a264"),
}


def _run_recorded_region(monkeypatch, region):
    """Run one region of RECORDED_CHUNKS; return its report and its model."""
    monkeypatch.setattr(chunk_mod, "pack_chunk", fake_pack)
    from quail.backends.quail.executor import label_stages as label_stages_module

    num_documents, num_partners = 14, 4
    documents = {
        "r": [[DOC + d] * (10 + 2 * d) for d in range(num_documents)],
        "p": [[PARTNER + i] * (4 + i) for i in range(num_partners)],
    }
    filter_truth = [[d % 3 != 0, d % 4 != 1] for d in range(num_documents)]
    label_truth = [d % 2 for d in range(num_documents)]
    join_truth = {("r", d): [(d + i) % 3 == 0 for i in range(num_partners)]
                  for d in range(num_documents)}
    joined_truth = {(d, i): (d + i) % 2 for d in range(num_documents)
                    for i in range(num_partners)}
    graph = {"fused": fused_graph, "scores": score_graph,
             "joined": joined_graph, "filter": lambda: filter_graph(1),
             "filters": lambda: filter_graph(2),
             "classify": classify_graph}[region]()
    if region == "joined":
        documents["q"] = [[PARTNER + num_partners] * 2]
        join_truth = {key: values + [1] for key, values in join_truth.items()}
        graph = joined_graph(two_joins=True)
    model = ScoringModel(filter_truth if region in ("filter", "filters", "fused")
                         else [], join_truth,
                         label_truth, joined_truth)
    torch = fake_torch()
    torch.nn = SimpleNamespace(functional=None)
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpus=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(torch=torch, async_answers=fake_readout(),
                         answer_rows=object(), chunk_tokens=120,
                         async_scores=fake_readout())

    class LabelReadout:
        trailing_rows = 1
        normalize = False

        def __init__(self, targets, rows):
            self.targets = np.asarray(targets)
            self.rows = rows
            self.dtype = np.dtype((np.float32, (rows, len(targets))))

        def submit(self, rows, rows_per_answer=None):
            if rows_per_answer is None:
                rows_per_answer = [1] * len(rows)
            padded = np.full((len(rows_per_answer), self.rows,
                              len(self.targets)), np.nan, np.float32)
            start = 0
            for answer, count in enumerate(rows_per_answer):
                for offset in range(count):
                    row = rows[start + offset]
                    padded[answer, offset] = (row if hasattr(row, "shape")
                                              else np.zeros(len(self.targets)))
                start += count
            return padded

        def result(self, rows):
            return rows

    monkeypatch.setattr(label_stages_module, "full_output_head",
                        lambda model: SimpleNamespace(shape=(1, 1), dtype="fake"))
    monkeypatch.setattr(
        label_stages_module, "AsyncLabelLogprobs",
        lambda torch, F, head, targets, rows, normalize: LabelReadout(
            targets, rows))
    execution.loaded_model.label_readout = LabelReadout(TARGETS, 1)
    state = {
        "gpu": SimpleNamespace(torch=torch, arena=arena, pipeline=pipeline,
                               execution=execution, spec=MODELS["qwen3-4b-fp8"]),
        "runtimes": built_in_registry().runtimes, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "documents": documents,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    return result, model


def _recorded(result, model) -> tuple:
    """The chunks, their digest, fresh tokens, and output digest of a run."""
    import hashlib
    import json

    compositions = chunk_compositions(model.launched)
    digest = hashlib.sha256(json.dumps(compositions).encode()).hexdigest()
    fresh = {name: metrics["fresh_tokens"]
             for name, metrics in sorted(result["node_metrics"].items())
             if metrics["fresh_tokens"]}
    rows = hashlib.sha256(json.dumps(
        [(repr(port), table.to_pydict())
         for port, table in sorted(result["_outputs"].items(), key=repr)],
        default=str).encode()).hexdigest()
    return len(compositions), digest, fresh, rows


@pytest.mark.parametrize("region", sorted(RECORDED_CHUNKS))
def test_each_region_packs_the_chunks_it_packed_before_streaming(
        monkeypatch, region):
    assert _recorded(*_run_recorded_region(monkeypatch, region)) \
        == RECORDED_CHUNKS[region]


def test_a_score_hands_on_each_row_once_its_request_is_answered(monkeypatch):
    reports = []

    class Taken:
        """Takes the rows a region reports, as the region's FinishedRows."""

        def wants(self, node, port):
            return False

        def report(self, table):
            metadata = table.schema.metadata
            reports.append((metadata[b"quail.node"].decode(),
                            list(zip(*(table.column(alias).to_pylist()
                                       for alias in ("r", "p")
                                       if alias in table.column_names)))))

    with reporting(Taken()):
        found = _recorded(*_run_recorded_region(monkeypatch, "scores"))
    # handing rows on changes no chunk
    assert found == RECORDED_CHUNKS["scores"]
    alone = [rows for node, rows in reports if node == "score:r"]
    pairs = [rows for node, rows in reports if node == "score:rp"]
    # each row is handed on once
    assert sorted(row for rows in alone for row in rows) == [(d,) for d in range(14)]
    assert sorted(row for rows in pairs for row in rows) == [
        (d, i) for d in range(14) for i in range(4)]
    # the documents under one prompt head are one anchor, handed on over
    # several chunks; some document's pairs, too, before it settles
    assert len(alone) > 1
    assert max(sum(any(row[0] == d for row in rows) for rows in pairs)
               for d in range(14)) > 1


def test_a_classification_alone_skips_the_documents_the_hint_drops(monkeypatch):
    class EvenDocuments:
        """A region's hint that keeps the even documents of r."""

        def generation(self):
            return 0

        def keep(self):
            return [d % 2 == 0 for d in range(14)]

    with reporting(None, EvenDocuments(), "r"):
        result, model = _run_recorded_region(monkeypatch, "classify")
    labeled = result["_outputs"][PortRef("classify:r", "rows:r")]
    assert labeled.column("r").to_pylist() == list(range(0, 14, 2))
    prefilled = [entry["key"][1] for chunk in model.launched for entry in chunk
                 if entry["prefix"] is not None]
    assert sorted(prefilled) == list(range(0, 14, 2))
