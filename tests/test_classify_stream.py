"""A classification's stages inside the chain and join of its table."""

from types import SimpleNamespace

import numpy as np
import pytest
from fakes import (
    DOC,
    FRAME,
    PARTNER,
    QUESTION,
    SETTINGS,
    FakeModel,
    cpu_arena,
    fake_pack,
    fake_pipeline,
    fake_torch,
)

from quail.backends.quail import QuailModelExecution
from quail.backends.quail.executor import loop
from quail.backends.quail.graph import execute_single_graph
from quail.builtins import built_in_registry
from quail.physical import (
    AiClassify,
    AiFilter,
    AiJoin,
    ClassifySpec,
    Filter,
    FilterStage,
    InList,
    JoinStage,
    PhysicalGraph,
    PortRef,
    Scan,
)
from quail.physical.base import input_ports
from quail.specs import DEVICES, MODELS

# the classification frame is written after the document like a join
# frame; the cue is the tail's last token
CLASSIFY_FRAME = 4500
CUE = 5000
# two-token labels sharing a first token: the trie path [CUE, 11] is
# read at both rows, beside filter and join rows read at one
LABEL_IDS = ((13,), (14,))
TARGETS = [13, 14]
# two-token labels a greedy decode follows one round at a time
DECODED_IDS = ((11, 13), (11, 14))
DECODED_TARGETS = [11, 13, 14]


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
                if head == PARTNER_LABEL:
                    # a joined row's block: the partner label, its document,
                    # the question, then the cue, whose row is read
                    partner = suffix[1] - PARTNER
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


def fused_graph():
    """The r chain, its classification, a filter on its label, and a join with p."""
    spec = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_inputs=14, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="letters", share_prefixes=True)
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            alias="r", arena_writes=True,
            stages=(FilterStage(0, 1, 0, 0.8, 14 * 0.8),),
            question_token_ids=((QUESTION,),)),
        AiClassify(
            node_id="classify:r",
            inputs=input_ports((PortRef("filter:r", "ids:r"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec),
        Filter(
            node_id="label:r",
            inputs=input_ports((PortRef("classify:r", "scores"),)),
            predicate=InList("topic", ("a",)), aliases=("r",),
            selectivity=0.5, written_pos=1),
        AiJoin(
            node_id="group:0", anchor="r", anchor_resident="filter",
            inputs=input_ports((PortRef("label:r", "ids:r"),
                                PortRef("input:p", "ids:p"))),
            stages=(JoinStage(
                written_pos=2, exec_idx=0, anchor="r", partners=("p",),
                semantics="full", selectivity=0.5, expected_tuples=1,
                anchor_frame_tokens=1, pair_tail_tokens=0,
                anchor_resident="filter", tuple_tokens=0, pairs_from="",
                frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
                tail_token_ids=()),)),
    ]
    return PhysicalGraph(tuple(nodes), PortRef("group:0", "ids:r"))


@pytest.mark.parametrize("shape", ["filter", "joined", "classify_first"])
def test_a_letters_read_on_a_canvas_model_runs_in_its_filter_chain_pipeline(
        monkeypatch, shape):
    """A filter, then a diffusion model's classification: one run.

    Each survivor's seeded canvas is read over its resident KV, in
    chunks that also hold other documents' filter rows. When ``joined``
    a filter on the label and a join follow, in the same pipeline. When
    ``classify_first`` the pipeline starts with the classification,
    whose plan settings carry no shared preamble, so the documents pack
    under the classification's own prompt head.
    """
    from dataclasses import replace

    from quail.backends.quail.executor import classify as classify_module
    from quail.execution.pipelines import build_pipelines
    from quail.specs.base import AnswerCanvas

    settings = AnswerCanvas(rows=4, turn_close_id=6, pad_id=0)
    first = shape == "classify_first"
    head = (80, 81) if first else ()
    spec = ClassifySpec(
        name="topic", aliases=("r",), query_template="", arguments=(),
        expected_inputs=4, estimated_seconds=0.0,
        prompt_token_parts=(head, (91, 92, 93)), labels=("a", "b"),
        label_token_ids=((1,), (2,)), scoring="letters")
    nodes = [Scan(node_id="input:r", alias="r", input_id="r")]
    if not first:
        nodes.append(AiFilter(
            node_id="filter:r",
            inputs=input_ports((PortRef("input:r", "ids:r"),)),
            alias="r", arena_writes=True,
            stages=(FilterStage(0, 1, 0, 0.8, 4 * 0.8),),
            question_token_ids=((QUESTION,),)))
    nodes.append(AiClassify(
        node_id="classify:r",
        inputs=input_ports((PortRef("input:r" if first else "filter:r",
                                    "ids:r"),)),
        backend_name="quail", model="diffusion-gemma-26b-a4b-fp8",
        spec=spec))
    root = PortRef("classify:r", "scores")
    if first:
        nodes.append(Filter(
            node_id="label:r",
            inputs=input_ports((PortRef("classify:r", "scores"),)),
            predicate=InList("topic", ("b",)), aliases=("r",),
            selectivity=0.5, written_pos=1))
        root = PortRef("label:r", "ids:r")
    if shape == "joined":
        nodes += [
            Scan(node_id="input:p", alias="p", input_id="p"),
            Filter(
                node_id="label:r",
                inputs=input_ports((PortRef("classify:r", "scores"),)),
                predicate=InList("topic", ("a",)), aliases=("r",),
                selectivity=0.5, written_pos=1),
            AiJoin(
                node_id="group:0", anchor="r", anchor_resident="filter",
                inputs=input_ports((PortRef("label:r", "ids:r"),
                                    PortRef("input:p", "ids:p"))),
                stages=(JoinStage(
                    written_pos=2, exec_idx=0, anchor="r", partners=("p",),
                    semantics="full", selectivity=0.5, expected_tuples=1,
                    anchor_frame_tokens=1, pair_tail_tokens=0,
                    anchor_resident="filter", tuple_tokens=0, pairs_from="",
                    frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
                    tail_token_ids=()),)),
        ]
        root = PortRef("group:0", "ids:r")
    graph = PhysicalGraph(tuple(nodes), root)
    chains = {pipeline.node_ids for pipeline in build_pipelines(graph).values()}
    assert chains == {{
        "filter": ("filter:r", "classify:r"),
        "joined": ("filter:r", "classify:r", "label:r", "group:0"),
        "classify_first": ("classify:r", "label:r")}[shape]}

    filter_truth = [[1], [0], [1], [1]]
    # the letter each survivor's first canvas row favors
    answers = {0: 1, 1: 1, 2: 2, 3: 2}
    vocab = 8
    packed = {}
    prefixes = {}

    def forward(chunk):
        rows = []
        for entry in chunk.specs:
            document = entry["key"][1]
            if entry["prefix"] is not None:
                prefixes[document] = list(entry["prefix"])
            for suffix in entry["suffixes"]:
                if entry.get("canvas") is not None:
                    packed.setdefault(document, []).append(list(entry["canvas"]))
                    assert entry["read_all_rows"]
                    # the frame and the cue pack as one entry before
                    # the canvas
                    assert list(suffix) == [91, 92, 93]
                    for position in range(settings.rows):
                        rows.append(np.asarray(
                            [-1.0 if position == 0 and token == answers[document]
                             else -5.0 for token in (1, 2)], np.float32))
                elif suffix[0] >= FRAME:
                    rows.append(0)
                elif suffix[0] >= PARTNER:
                    # document 0 pairs with partner 1 only
                    rows.append(float(document == 0 and suffix[0] == PARTNER + 1))
                else:
                    # a filter row on the fixed canvas
                    assert suffix[0] == QUESTION
                    rows.append(float(filter_truth[document][0]))
        return rows

    class FakeReadout:
        def __init__(self, targets, rows):
            self.targets = np.asarray(targets)
            self.rows = rows
            self.dtype = np.dtype((np.float32, (rows, len(targets))))

        def submit(self, rows, rows_per_answer=None):
            rows_per_answer = rows_per_answer or [1] * len(rows)
            padded = np.full((len(rows_per_answer), self.rows,
                              len(self.targets)), np.nan, np.float32)
            start = 0
            for answer, count in enumerate(rows_per_answer):
                for offset in range(count):
                    row = rows[start + offset]
                    padded[answer, offset] = (
                        row if hasattr(row, "shape")
                        else np.zeros(len(self.targets)))
                start += count
            return padded

        def result(self, rows):
            return rows

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    monkeypatch.setattr(classify_module, "full_output_head",
                        lambda model: SimpleNamespace(shape=(1, 1),
                                                      dtype="fake"))
    monkeypatch.setattr(
        classify_module, "AsyncLabelLogprobs",
        lambda torch, F, head, targets, rows, normalize: FakeReadout(
            targets, rows))
    torch = fake_torch()
    torch.nn = SimpleNamespace(functional=None)
    pipeline = fake_pipeline(forward_chunk=forward, canvas_ids=(7,),
                             tree_attention=False)
    arena = cpu_arena(64)
    model_spec = replace(MODELS["qwen3-4b-fp8"], name="tiny")
    execution = QuailModelExecution(SimpleNamespace(
        model=model_spec, gpu_index=0, gpu_count=1, device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(), chunk_tokens=64)
    execution._state.update(
        model_spec=SimpleNamespace(name="tiny", vocab=vocab,
                                   answer_canvas=settings))
    docs = {"r": [[DOC + d] * (3 + d) for d in range(4)],
            "p": [[PARTNER + d] for d in range(2)]}
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": model_spec, "device": DEVICES["h100-sxm"],
        "chunk_tokens": 64, "docs": docs,
    }
    # the settings' shared preamble is empty, as a classification plan's is
    assert SETTINGS["pre_ids"] == []
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    labels = outputs[PortRef("classify:r", "scores")]
    if first:
        assert labels.column("topic").to_pylist() == ["a", "a", "b", "b"]
        assert outputs[root].column("r").to_pylist() == [2, 3]
        assert prefixes == {d: list(head) + docs["r"][d] for d in range(4)}
        return
    assert outputs[PortRef("filter:r", "ids:r")].column("r").to_pylist() \
        == [0, 2, 3]
    assert labels.column("r").to_pylist() == [0, 2, 3]
    assert labels.column("topic").to_pylist() == ["a", "b", "b"]
    # each survivor packed one canvas: a random token, the turn close,
    # then padding
    assert sorted(packed) == [0, 2, 3]
    for document in (0, 2, 3):
        (canvas,) = packed[document]
        assert 0 <= canvas[0] < vocab and canvas[1:] == [6, 0, 0]
    # the classification packed a frame, the cue, and the canvas per
    # survivor over its resident KV: no second prefill
    assert metrics["classify:r"]["fresh_tokens"] == 3 * (2 + 1 + 4)
    assert metrics["classify:r"]["evaluated_documents"] == 3
    assert metrics["filter:r"]["fresh_tokens"] == sum(map(len, docs["r"])) + 4
    if shape == "joined":
        # the filter keeps document 0, whose partner suffixes
        # ran over its resident KV after its read
        assert outputs[PortRef("label:r", "ids:r")].column("r").to_pylist() \
            == [0]
        joined_ids = outputs[PortRef("group:0", "ids:r")]
        assert joined_ids.column("r").to_pylist() == [0]
        assert metrics["group:0"]["fresh_tokens"] == 1 + 2


def test_streamed_classification_labels_survivors_with_their_kv_resident(
        monkeypatch):
    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    n_docs, n_partners = 14, 4
    docs = {
        "r": [[DOC + d] * (10 + 2 * d) for d in range(n_docs)],
        "p": [[PARTNER + i] * 20 for i in range(n_partners)],
    }
    filter_truth = [[d % 3 != 0] for d in range(n_docs)]
    label_truth = [d % 2 for d in range(n_docs)]           # even: "a"
    join_truth = {("r", d): [(d + i) % 3 == 0 for i in range(n_partners)]
                  for d in range(n_docs)}
    model = LabelingModel(filter_truth, join_truth, label_truth)
    torch = fake_torch()
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpu_count=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(), chunk_tokens=120)
    def submit(rows, rows_per_answer=None):
        # (answers, rows, targets), padded; a frame entry's one row is
        # not read and carries no log probabilities
        rows_per_answer = rows_per_answer or [1] * len(rows)
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

    execution.state["label_readout"] = SimpleNamespace(
        targets=np.asarray(TARGETS), rows=1,
        dtype=np.dtype((np.float32, (1, len(TARGETS)))), submit=submit,
        result=lambda rows: rows)
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
    }
    result = execute_single_graph(state, SETTINGS, fused_graph())
    assert not arena.accounting.owned

    passed = [d for d in range(n_docs) if filter_truth[d][0]]
    accepted = [d for d in passed if label_truth[d] == 0]
    matched = [d for d in accepted if any(join_truth[("r", d)])]
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    labels = outputs[PortRef("classify:r", "scores")]
    assert labels.column("r").to_pylist() == passed
    assert labels.column("topic").to_pylist() == [
        ("a", "b")[label_truth[d]] for d in passed]
    assert outputs[PortRef("classify:r", "ids:r")].column("r").to_pylist() \
        == passed
    assert outputs[PortRef("label:r", "ids:r")].column("r").to_pylist() \
        == accepted
    assert outputs[PortRef("group:0", "ids:r")].column("r").to_pylist() \
        == matched
    # the chain packed every document once; the classification and the
    # join packed only frames and suffixes after resident KV
    prefill = sum(len(docs["r"][d]) for d in range(n_docs))
    assert metrics["filter:r"]["fresh_tokens"] == prefill + n_docs
    # a one-token frame and the cue per survivor
    assert metrics["classify:r"]["fresh_tokens"] == 2 * len(passed)
    assert metrics["classify:r"]["evaluated_documents"] == len(passed)
    assert metrics["classify:r"]["suffix_tokens"] == len(passed)
    # some chunk held documents at the chain's stage beside documents
    # at the classification's or the join's
    heads = [{suffix[0] for spec in specs for suffix in spec["suffixes"]}
             for specs in model.launched]
    assert any(QUESTION in chunk and (CUE in chunk or chunk & {FRAME})
               or CLASSIFY_FRAME in chunk and FRAME in chunk
               for chunk in heads)


def joined_graph():
    """A join of r with p, then a classification of the rows it keeps."""
    spec = ClassifySpec(
        name="stance", aliases=("r", "p"), query_template="", arguments=(),
        expected_inputs=3, estimated_seconds=0.0,
        prompt_token_parts=((), (CLASSIFY_FRAME, CUE)), labels=("a", "b"),
        label_token_ids=LABEL_IDS, scoring="letters",
        join_layout=((NOTE,), (PARTNER_LABEL,)))
    nodes = [
        Scan(node_id="input:r", alias="r", input_id="r"),
        Scan(node_id="input:p", alias="p", input_id="p"),
        AiJoin(
            node_id="group:0", anchor="r", anchor_resident="fresh",
            inputs=input_ports((PortRef("input:r", "ids:r"),
                                PortRef("input:p", "ids:p"))),
            stages=(JoinStage(
                written_pos=2, exec_idx=0, anchor="r", partners=("p",),
                semantics="full", selectivity=0.5, expected_tuples=3,
                anchor_frame_tokens=1, pair_tail_tokens=0,
                anchor_resident="fresh", tuple_tokens=0, pairs_from="",
                frame_token_ids=(FRAME,), label_token_ids=(("p", ()),),
                tail_token_ids=()),)),
        AiClassify(
            node_id="classify:rp",
            inputs=input_ports((PortRef("group:0", "join_answers:2"),)),
            backend_name="quail", model="qwen3-4b-fp8", spec=spec),
    ]
    return PhysicalGraph(tuple(nodes), PortRef("classify:rp", "scores"))


def test_a_classification_of_joined_rows_runs_after_its_join_on_the_anchors_kv(
        monkeypatch):
    from quail.execution.pipelines import build_pipelines

    graph = joined_graph()
    chains = {pipeline.node_ids for pipeline in build_pipelines(graph).values()}
    assert chains == {("group:0", "classify:rp")}

    monkeypatch.setattr(loop, "pack_chunk", fake_pack)
    docs = {"r": [[DOC + d] * (5 + d) for d in range(3)],
            "p": [[PARTNER + d, PARTNER + d] for d in range(2)]}
    # r0 pairs with both partners, r1 with p1 only, r2 with none
    join_truth = {("r", 0): [1, 1], ("r", 1): [0, 1], ("r", 2): [0, 0]}
    joined_truth = {(0, 0): 0, (0, 1): 1, (1, 1): 0}
    model = LabelingModel([], join_truth, [], joined_truth)
    torch = fake_torch()
    torch.nn = SimpleNamespace(functional=None)
    arena = cpu_arena(64)
    pipeline = fake_pipeline(forward_chunk=model.forward_chunk)
    execution = QuailModelExecution(SimpleNamespace(
        model=MODELS["qwen3-4b-fp8"], gpu_index=0, gpu_count=1,
        device=DEVICES["h100-sxm"]))
    execution.bind_loaded_model(model=object(), arena=arena, pipeline=pipeline)
    execution.bind_query(
        torch=torch,
        async_answers=SimpleNamespace(submit=lambda v: v, result=lambda v: v,
                                      dtype=None),
        answer_rows=object(), chunk_tokens=120)
    from quail.backends.quail.executor import classify as classify_module

    class FakeReadout:
        def __init__(self, targets, rows):
            self.targets = np.asarray(targets)
            self.rows = rows
            self.dtype = np.dtype((np.float32, (rows, len(targets))))

        def submit(self, rows, rows_per_answer=None):
            rows_per_answer = rows_per_answer or [1] * len(rows)
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

    monkeypatch.setattr(classify_module, "full_output_head",
                        lambda model: SimpleNamespace(shape=(1, 1),
                                                      dtype="fake"))
    monkeypatch.setattr(
        classify_module, "AsyncLabelLogprobs",
        lambda torch, F, head, targets, rows, normalize: FakeReadout(
            targets, rows))
    state = {
        "torch": torch, "arena": arena, "pipeline": pipeline,
        "model_execution": execution,
        "runtimes": built_in_registry().runtimes,
        "model_spec": MODELS["qwen3-4b-fp8"], "device": DEVICES["h100-sxm"],
        "chunk_tokens": 120, "docs": docs,
    }
    result = execute_single_graph(state, SETTINGS, graph)
    assert not arena.accounting.owned
    outputs = result["_outputs"]
    metrics = result["node_metrics"]
    assert outputs[PortRef("group:0", "ids:r")].column("r").to_pylist() \
        == [0, 1]
    labels = outputs[PortRef("classify:rp", "scores")]
    assert labels.column("r").to_pylist() == [0, 0, 1]
    assert labels.column("p").to_pylist() == [0, 1, 1]
    assert labels.column("stance").to_pylist() == ["a", "b", "a"]
    # the join packed the anchors once; the classification packed, per
    # anchor kept, its note and, per pair, the partner block, the
    # question, and the cue: no second prefill
    block = 1 + 2 + 1
    assert metrics["classify:rp"]["fresh_tokens"] == 2 * 1 + 3 * (block + 1)
    assert metrics["classify:rp"]["evaluated_document_pairs"] == 3
    # the join packed every anchor, its frame, and both partners' suffixes
    assert metrics["group:0"]["fresh_tokens"] == sum(map(len, docs["r"])) \
        + 3 * 1 + 6 * 2


