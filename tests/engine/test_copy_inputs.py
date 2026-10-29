"""Extraction branches and rewound stages see identical tokens and positions."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers.extract_fakes import make_vocabulary
from helpers.fakes import cpu_staging, fake_pipeline
from helpers.kv_checker import Setup, make_arena
from test_map_decode import TokenModel, events_torch

from quail.backends.quail.executor.cuda_graphs import PiecewiseGraphs
from quail.backends.quail.executor.extract.copy import CopyConstraint
from quail.backends.quail.executor.extract.stages import ExtractionStages
from quail.backends.quail.executor.extract.vocabulary import Vocabulary
from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.backends.quail.executor.operators.extract import ExtractPart
from quail.backends.quail.executor.readout import AsyncTopLogprobs
from quail.backends.quail.executor.stages import run_stages
from quail.catalog.tokens import GigatokenEncoder
from quail.logical import ColumnRef
from quail.logical.prompts import bind_extract_prompt
from quail.physical import AiExtract, ExtractSpec
from quail.specs import H100_SXM, QWEN3_4B_FP8

QUESTIONS = (
    "What is the term?",
    "What term applies after termination, including any exceptions?",
    "How long does the agreement last?",
)
TEXTS = (
    "Contract A. café 日本語. five days or " + "five " * 32 + "years.",
    "Contract B. " + "A long clause. " * 17 + "five days or " + "five " * 32 + "years.",
    "Contract C. " + "A clause. " * 43 + "five days or " + "five " * 32 + "years.",
)


def prompt(question):
    return bind_extract_prompt(
        (ColumnRef("c", "contracts", "text"),), question, QWEN3_4B_FP8.turn
    )


@pytest.fixture(scope="module", params=["words", "qwen"])
def vocabulary(request):
    if request.param == "words":
        parts = [
            text
            for question in QUESTIONS
            for text in (
                prompt(question).preamble,
                prompt(question).tail.removeprefix("{0}"),
            )
        ]
        return make_vocabulary(*TEXTS, *parts)
    path = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-FP8/snapshots"
        / "96b30dc13593a244a5e59e84687309f53c375cfa"
    )
    if not (path / "tokenizer.json").exists():
        pytest.skip("the real Qwen tokenizer is not cached")
    return Vocabulary(GigatokenEncoder(str(path)), QWEN3_4B_FP8.turn_suffix)


def specs_for(vocabulary, draft):
    specs = []
    for q, question in enumerate(QUESTIONS):
        prompt = bind_extract_prompt(
            (ColumnRef("c", "contracts", "text"),), question, QWEN3_4B_FP8.turn
        )
        head = tuple(vocabulary.encode_ids(prompt.preamble))
        tail = tuple(vocabulary.encode_ids(prompt.tail.removeprefix("{0}")))
        specs.append(
            ExtractSpec(
                name=f"field{q}",
                aliases=("c",),
                query_template="{0}",
                arguments=(("c", "text"),),
                expected_rows=3,
                estimated_seconds=0,
                question=question,
                prompt_token_parts=(head, tail),
                max_answer_tokens=13,
                draft_tokens=draft,
                draft_policy="fixed",
                top_tokens=4,
            )
        )
    return specs


def run(vocabulary, texts, old, mode, draft, budget, planned="unified"):
    specs = specs_for(vocabulary, draft)
    (word,) = vocabulary.encode_ids(" five")
    encoded = vocabulary.document_batch(texts, [None] * len(texts))
    setup = Setup(path=mode, pages=1000, budget=budget, page_tokens=16)
    arena = make_arena(setup)
    model = TokenModel(arena)
    torch_ = events_torch(0)
    columns = torch.arange(1024)
    columns = columns[columns != 1022]
    marker = torch.sparse_coo_tensor(
        torch.stack((torch.full((len(columns),), word), columns)),
        torch.ones(len(columns)),
        (len(vocabulary.vocab), 1024),
        check_invariants=True,
    ).coalesce()
    pipeline = fake_pipeline(
        forward_chunk=model.forward_chunk, canvas_ids=(), tree_attention=True
    )
    pipeline.engine = SimpleNamespace(arena=arena)
    pipeline.forward_hook = None
    graphs = PiecewiseGraphs.for_specs(pipeline, QWEN3_4B_FP8, H100_SXM)
    graphs.graphs = [object()]
    state = SimpleNamespace(
        torch=torch_,
        loaded_model=SimpleNamespace(
            arena=arena,
            vocabulary=vocabulary,
            model=SimpleNamespace(quail_full_head=marker),
            pipeline=pipeline,
        ),
    )
    if old:
        readout = AsyncTopLogprobs(
            torch_, torch_.nn.functional, marker, 4, [vocabulary.stop_ids], ragged=True
        )
        extraction = ExtractionStages(
            state,
            specs,
            vocabulary,
            range(3),
            encoded,
            texts,
            readout=readout,
            pipeline_head=specs[0].prompt_token_parts[0],
        )
        prefixes, keys, stages = extraction.prefixes, extraction.keys, extraction.stages
        decoder, tree, dependencies = None, None, None
        key_fields = None
    else:
        part = ExtractPart(
            state,
            [
                AiExtract(
                    node_id=f"e{q}",
                    backend_name="quail",
                    model=QWEN3_4B_FP8.name,
                    attention=planned,
                    spec=spec,
                )
                for q, spec in enumerate(specs)
            ],
            range(3),
            {"documents": {"c": encoded}, "texts": {"c": texts}},
        )
        extraction = part.extraction
        stages = part.stages
        prefixes, keys, tree, dependencies = part.branch_inputs(
            extraction.prefixes, extraction.keys, None, 16, 0
        )
        decoder = MapDecoder(
            torch_, marker, len(prefixes), part.answer_limits, vocabulary.stop_ids, 16,
            workspace_bytes=64 * 2**20,
        )
        part.decoder = decoder
        key_fields = {keys[a]: item for a, item in part.branch_of.items()}
    histories = Counter()
    expected_rows = {}
    graph_eligible = 0
    scatter_rows = 0
    original = model.forward_chunk

    def observe(chunk):
        nonlocal graph_eligible, scatter_rows
        graph_eligible += int(graphs.supports(chunk))
        if chunk.meta.get("unified") is not None:
            pools = [chunk.meta["unified"]]
            for pool in pools:
                dest = pool["dst"].numpy()
                assert len(dest) == len(np.unique(dest)), (
                    "duplicate scatter destinations"
                )
                if pool["tail_src"] is not None:
                    assert not set(pool["tail_src"].tolist()) & set(
                        pool["tail_dst"].tolist()
                    ), "overlapping tail copy"
                scatter_rows += len(dest)
        else:
            dest = chunk.meta["kv_dst"]
            if dest is not None:
                assert len(dest) == len(torch.unique(dest)), (
                    "duplicate tree scatter destinations"
                )
                scatter_rows += len(dest)
        result = original(chunk)
        # The unused head column identifies the selected normalized row.
        result[:, 1022] = torch.arange(1, len(result) + 1)
        final_rows = {row: i + 1 for i, row in enumerate(chunk.final_indices.tolist())}
        expected_rows.clear()
        finals = set(final_rows)
        work = []
        for key, start, _, end in chunk.row_spans:
            if old:
                d = int(key[1])
                q = next(
                    (
                        q
                        for q, a in enumerate(extraction.documents[d].answers)
                        if not a.done
                    ),
                    None,
                )
                if q is None:
                    continue
            else:
                q, d = key_fields[key]
            work.extend((r, key, q, d) for r in range(start, end) if r in finals)
        if chunk.decode_rows is not None:
            start = chunk.tokens - len(chunk.decode_rows)
            work.extend(
                (start + i, keys[int(slot)], *part.branch_of[int(slot)])
                for i, slot in enumerate(chunk.decode_rows.slots)
            )
        for row, key, q, d in work:
            position = int(chunk.positions[row])
            prompt = list(map(int, extraction.prefixes[d])) + list(
                specs[q].prompt_token_parts[1]
            )
            if position < len(prompt) - 1:
                continue
            table = np.asarray(arena.owned_pages(key))
            logical = np.arange(position + 1)
            physical = table[logical // 16] * 16 + logical % 16
            tokens = tuple(map(int, model.full.token[physical]))
            positions = tuple(map(int, model.full.position[physical]))
            assert positions == tuple(range(position + 1)), (
                old,
                mode,
                q,
                d,
                "position mismatch",
            )
            assert tokens[: len(prompt)] == tuple(prompt), (
                old,
                mode,
                q,
                d,
                "prompt mismatch",
            )
            index = position + 1 - len(prompt)
            histories[(d, q, index, tokens, positions)] += 1
            expected_rows[d, q, index] = final_rows[row]
        return result

    pipeline.forward_chunk = observe
    if decoder is not None:
        submit = decoder.submit

        def checked(normed, first_rows, first_documents, rows):
            for row, anchor in zip(first_rows, first_documents):
                q, d = part.branch_of[anchor]
                assert (
                    int(normed[row, 1022])
                    == expected_rows[d, q, int(decoder.fed[anchor])]
                )
            if rows is not None:
                start = len(normed) - len(rows)
                for offset, anchor in enumerate(rows.slots):
                    q, d = part.branch_of[int(anchor)]
                    index = int(rows.positions[offset] - decoder.origin[anchor] + 1)
                    assert (
                        int(normed[start + offset, 1022]) == expected_rows[d, q, index]
                    )
            return submit(normed, first_rows, first_documents, rows)

        decoder.submit = checked
    stats = {}
    run_stages(
        torch_,
        arena,
        pipeline,
        stages,
        prefixes,
        budget,
        anchor_keys=keys,
        prefix_tree=tree,
        dependencies=dependencies,
        decoder=decoder,
        attention_mode=mode,
        appended_tokens=(extraction.appended_tokens if old else None),
        stats=stats,
    )
    if old:
        extraction.finish()
        answers = [extraction.answers(q) for q in range(3)]
    else:
        answers = [part.answers(q) for q in range(3)]
    return (
        histories,
        answers,
        dict(
            eligible=graph_eligible,
            reads=sum(histories.values()),
            scatter=scatter_rows,
            shared_reads=model.shared_reads,
            read_tokens=None if decoder is None else decoder.read_tokens,
            stats=stats,
        ),
    )


@pytest.mark.parametrize("draft", [0, 3, 8])
@pytest.mark.parametrize("budget", [512, 1024])
def test_branches_and_rewinds_read_identical_prompt_and_draft_histories(
    monkeypatch, vocabulary, draft, budget
):
    cpu_staging(monkeypatch)
    texts = list(TEXTS)
    head = len(vocabulary.encode_ids(prompt(QUESTIONS[0]).preamble))
    padding = -(head + len(vocabulary.encode_ids(texts[0]))) % 16
    texts[0] += " five" * padding
    baseline, answers, info = run(vocabulary, texts, True, "tree", draft, budget)
    assert baseline and all(
        value is not None for values, _ in answers for value in values
    )
    assert info["eligible"] > 0
    for old, mode in [(True, "unified"), (False, "unified"), (False, "tree")]:
        current, got, other = run(vocabulary, texts, old, mode, draft, budget)
        assert got == answers
        assert current == baseline
        assert other["eligible"] > 0


@pytest.mark.parametrize("draft", [0, 3])
def test_planned_tree_extractions_read_shared_pages_once_without_changing_answers(
    monkeypatch, vocabulary, draft
):
    cpu_staging(monkeypatch)
    baseline, answers, unshared = run(
        vocabulary, list(TEXTS), False, "unified", draft, 512
    )
    current, got, shared = run(
        vocabulary, list(TEXTS), False, "unified", draft, 512, planned="tree"
    )
    assert current == baseline
    assert got == answers
    assert unshared["shared_reads"] == 0
    assert shared["shared_reads"] > 0
    reads = shared["read_tokens"]
    assert reads["shared"] > 0
    assert reads["alone"] + reads["own"] + reads["shared"] < reads["all"]


def test_prefill_launches_before_copy_setup_and_pending_first_tokens(
    monkeypatch, vocabulary
):
    cpu_staging(monkeypatch)
    _, expected, _ = run(vocabulary, list(TEXTS), True, "unified", 3, 512)
    num_passes = 0
    first_pending = None
    decoder = None
    launched_pending = []
    prepared = []
    submitted = {}
    submit = MapDecoder.submit
    forward = TokenModel.forward_chunk
    initialize = CopyConstraint.__init__
    prepare = CopyConstraint.prepare
    read = MapDecoder._read

    def launched(model, chunk):
        nonlocal num_passes
        num_passes += 1
        if decoder is not None and (decoder.copy_pending.any()
                                    or decoder.inflight.any()):
            launched_pending.append(num_passes)
            if chunk.decode_rows is not None:
                assert not decoder.copy_pending[chunk.decode_rows.slots].any()
        return forward(model, chunk)

    def configured(self, *args):
        assert num_passes > 0, "candidate setup must run under the first prefill"
        initialize(self, *args)

    def queued(self, *args):
        nonlocal decoder
        decoder = self
        submit(self, *args)
        for copy in self.copies:
            submitted.setdefault(id(copy), (copy, num_passes))

    def landed(self, copy):
        nonlocal first_pending
        read(self, copy)
        if first_pending is None and self.copy_pending.any():
            first_pending = submitted[id(copy)][1]
            assert num_passes > first_pending, "ready prefill must precede harvest"

    def ready(self, *args):
        if not prepared:
            assert first_pending is not None
            assert num_passes > first_pending, "ready prefill must launch first"
        prepared.append(num_passes)
        prepare(self, *args)

    monkeypatch.setattr(TokenModel, "forward_chunk", launched)
    monkeypatch.setattr(CopyConstraint, "__init__", configured)
    monkeypatch.setattr(CopyConstraint, "prepare", ready)
    monkeypatch.setattr(MapDecoder, "_read", landed)
    monkeypatch.setattr(MapDecoder, "submit", queued)
    _, answers, _ = run(vocabulary, list(TEXTS), False, "unified", 3, 512)
    assert answers == expected
    assert launched_pending and prepared
