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
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    ClassifySpec,
    Filter,
    PhysicalGraph,
    PortRef,
    Scan,
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
