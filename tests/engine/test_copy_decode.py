"""Copy decoding agrees with the recurrent extraction token rule on CPU."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from helpers.extract_fakes import make_spec
from test_map_decode import events_torch

from quail.backends.quail.executor.extract.decode import Decoder
from quail.backends.quail.executor.extract.stages import ExtractionInputs
from quail.backends.quail.executor.map_decoder import MapDecoder


@pytest.mark.parametrize("draft_tokens", [0, 4])
@pytest.mark.parametrize("draft_policy", ["fixed", "adaptive"])
@pytest.mark.parametrize("spellings", [False, True])
@pytest.mark.parametrize("answer_limit", [1, 8])
def test_copy_answers_and_spans_match_the_recurrent_token_rule(
    draft_tokens, draft_policy, spellings, answer_limit
):
    rng = np.random.default_rng(817)
    words = (
        ["foo", "foobarbaz", "barbaz", "Foo"]
        if spellings
        else ["five", "years", "days", "lease"]
    )
    plain = [" ".join(rng.choice(words, size=14)) for _ in range(9)]
    texts = [text.replace(" ", "  \n") if spellings else text for text in plain]
    pool_text = " ".join(words)
    spec, vocabulary = make_spec(*texts, *plain, pool_text, *words)
    spec = replace(
        spec,
        draft_tokens=draft_tokens,
        draft_policy=draft_policy,
        max_answer_tokens=answer_limit,
        top_tokens=4,
        first_window_chars=8,
        window_chars=12,
    )
    num_questions, num_documents = 3, len(texts)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(
        state, [spec] * num_questions, vocabulary, np.arange(num_documents), None, texts
    )
    num_tokens = len(vocabulary.vocab)
    logits = rng.normal(-8, 1, (num_questions, num_documents, 16, num_tokens))
    # Force repeated starts, alternate continuations, stops, and absent words.
    pool = vocabulary.encode_ids(pool_text)
    for q in range(num_questions):
        for document in range(num_documents):
            for at in range(16):
                token = int(rng.choice(pool))
                if at > rng.integers(1, 7):
                    token = vocabulary.stop_ids[0]
                logits[q, document, at, token] = 3
                if spellings:
                    logits[q, document, at, int(rng.choice(pool))] = 3
    logits = torch.tensor(logits, dtype=torch.float32)
    reference = Decoder(vocabulary, 4, 9, num_questions, spec)
    for document in documents.documents:
        for q in range(num_questions):
            answer = document.answers[q]
            fed = []
            while not answer.done:
                at = len(answer.written)
                block = logits[q, document.index, at : at + max(1, len(fed))]
                lp = block.log_softmax(dim=1)
                top, ids = lp.topk(4, dim=1)
                rows = torch.cat(
                    (
                        torch.logsumexp(lp[:, vocabulary.stop_ids], dim=1)[:, None],
                        ids.float(),
                        top,
                    ),
                    dim=1,
                ).numpy()
                reference.verify(document, q, rows, fed)
                if not answer.done:
                    fed = reference.draft(answer, document.rho, q)
    slots = np.arange(num_questions * num_documents).reshape(num_questions, -1)
    decoder = MapDecoder(
        events_torch(0),
        torch.eye(num_tokens),
        slots.size,
        [answer_limit] * num_questions,
        vocabulary.stop_ids,
        16,
    )
    decoder.configure_copy(0, slots, documents, [spec] * num_questions)
    for q in range(num_questions):
        for document in range(num_documents):
            decoder.start(int(slots[q, document]), q, 0, [0, 1])
    first = slots.flatten()
    decoder.submit(
        logits[:, :, 0].reshape(-1, num_tokens), np.arange(slots.size), first, None
    )
    decoder.harvest(wait=True)
    while decoder.active:
        rows = decoder.rows()
        assert rows is not None
        chunk = SimpleNamespace(
            decode_rows=rows,
            tokens=len(rows),
            input_ids=torch.empty(len(rows), dtype=torch.int64),
            meta={},
        )
        decoder.feed(chunk)
        questions, positions = np.divmod(rows.slots, num_documents)
        hidden = logits[questions, positions, rows.positions + 1]
        decoder.submit(hidden, [], [], rows)
        decoder.harvest(wait=True)
    for q in range(num_questions):
        expected = [document.answers[q] for document in documents.documents]
        values, spans = decoder.output_answers(q, slots[q])
        assert values == [answer.answer for answer in expected]
        assert spans == [answer.span for answer in expected]
        assert decoder.lengths[q][slots[q]].tolist() == [
            len(a.written) for a in expected
        ]


@pytest.mark.parametrize("draft_tokens", [0, 3])
@pytest.mark.parametrize("limit", [None, 2])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("lag", [0, 3])
@pytest.mark.parametrize("pages", [24, 300])
def test_copy_branches_match_reference_after_filters_and_limits(
    monkeypatch, draft_tokens, limit, filtered, lag, pages
):
    from helpers.fakes import cpu_staging, fake_pipeline
    from helpers.kv_checker import Setup, chain, make_arena
    from test_map_decode import Bits, TokenModel, next_token

    from quail.backends.quail.executor.operators.extract import ExtractPart
    from quail.backends.quail.executor.operators.filter import FilterPart
    from quail.backends.quail.executor.parts import gated_stages
    from quail.backends.quail.executor.stages import run_stages
    from quail.physical import AiExtract

    cpu_staging(monkeypatch)
    rng = np.random.default_rng(19)
    texts = [
        " ".join(rng.choice(["five", "years", "days", "lease"], 25)) for _ in range(12)
    ]
    spec, vocabulary = make_spec(*texts, "five years days lease")
    spec = replace(
        spec,
        draft_tokens=draft_tokens,
        max_answer_tokens=8,
        top_tokens=4,
        draft_policy="fixed",
    )
    nodes = [
        AiExtract(
            node_id=f"extract:{q}",
            backend_name="quail",
            model="qwen3-4b-fp8",
            spec=replace(
                spec,
                name=f"answer{q}",
                prompt_token_parts=(
                    spec.prompt_token_parts[0],
                    (*spec.prompt_token_parts[1], q + 10),
                ),
            ),
        )
        for q in range(3)
    ]
    setup = Setup(path="unified", pages=pages, page_tokens=4, budget=128)
    arena = make_arena(setup)
    torch_ = events_torch(lag)
    model = TokenModel(arena)
    state = SimpleNamespace(
        loaded_model=SimpleNamespace(arena=arena, vocabulary=vocabulary)
    )
    ids = list(range(len(texts)))
    inputs = {
        "documents": {
            "c": (
                None if lag else vocabulary.document_batch(texts, [None] * len(texts))
            )
        },
        "texts": {"c": texts},
    }
    part = ExtractPart(state, nodes, ids, inputs)
    parts = [part]
    question = [700, 701]
    if filtered:
        parts.insert(
            0, FilterPart(SimpleNamespace(question_token_ids=question), ids, Bits())
        )
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = part.branch_inputs(
        part.extraction.prefixes,
        part.extraction.keys,
        None,
        setup.page_tokens,
        len(stages) - 1,
    )
    pool = [*vocabulary.encode_ids("five years days lease"), *vocabulary.stop_ids]
    head = torch.zeros(len(vocabulary.vocab), 1024)
    head[rng.choice(pool, 1024), np.arange(1024)] = 1
    decoder = MapDecoder(
        torch_,
        head,
        len(prefixes),
        part.answer_limits,
        vocabulary.stop_ids,
        setup.page_tokens,
    )
    part.decoder = decoder
    survivors = []

    def settled(anchor, survived, row):
        if anchor < len(texts) and survived:
            survivors.append(anchor)
        arena.free_key(keys[anchor])

    every, timed_spans, fresh = run_stages(
        torch_,
        arena,
        fake_pipeline(forward_chunk=model.forward_chunk, canvas_ids=()),
        stages,
        prefixes,
        setup.budget,
        anchor_keys=keys,
        prefix_tree=tree,
        dependencies=dependencies,
        decoder=decoder,
        on_settled=settled,
        limit=limit,
        survival=1.0,
    )
    expected_survivors = [
        d
        for d, document in enumerate(part.extraction.documents)
        if not filtered
        or next_token(chain(list(map(int, document.prefix)) + question)) % 2
    ]
    assert set(survivors).issubset(expected_survivors)
    if limit is None:
        assert set(survivors) == set(expected_survivors)
    else:
        assert len(survivors) >= min(limit, len(expected_survivors))
    reference = Decoder(vocabulary, 4, 9, len(nodes), spec)
    for q, node in enumerate(nodes):
        assert set(np.flatnonzero(part.asked[q])) == set(survivors)
        for d in survivors:
            document = part.extraction.documents[d]
            answer = document.answers[q]
            prompt = list(map(int, document.prefix)) + list(
                node.spec.prompt_token_parts[1]
            )
            while not answer.done:
                hidden = head[:, next_token(chain(prompt + answer.written))]
                lp = hidden - torch.logsumexp(hidden, dim=0)
                top, tokens = lp.topk(4)
                row = torch.cat(
                    (
                        torch.logsumexp(lp[list(vocabulary.stop_ids)], dim=0)[None],
                        tokens.float(),
                        top,
                    )
                ).numpy()
                reference.choose(document, q, row)
        values, spans = part.answers(q, survivors)
        assert values == [
            part.extraction.documents[d].answers[q].answer for d in survivors
        ]
        assert spans == [
            part.extraction.documents[d].answers[q].span for d in survivors
        ]
    part.finish(every)
    part.complete(fresh, timed_spans, {}, len(stages) - 1)
    results = part.results()
    assert sum(result.metrics.fresh_tokens for result in results.values()) == fresh
    for q, node in enumerate(nodes):
        result = results[node.node_id]
        asked = np.flatnonzero(part.asked[q])
        assert result.outputs[node.rows_port].column(node.spec.name).to_pylist() == [
            part.extraction.documents[d].answers[q].answer for d in asked
        ]
        extension = result.metrics.extension
        assert extension["answer_tokens"] == sum(
            len(part.extraction.documents[d].answers[q].written) for d in asked
        )
        assert extension["suffix_tokens"] == (
            len(asked) * len(node.spec.prompt_token_parts[1]) + decoder.rows_fed[q]
        )
        assert extension["steps"] >= len(asked)
    assert not arena.accounting.owned


def test_map_and_copy_fields_can_reuse_a_slot_without_changing_saved_spans():
    spec, vocabulary = make_spec()
    spec = replace(spec, top_tokens=4, max_answer_tokens=2, draft_tokens=0)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(state, [spec], vocabulary, [0], None, ["five years"])
    size = len(vocabulary.vocab)
    decoder = MapDecoder(
        events_torch(0), torch.eye(size), 1, [2, 2, 2], vocabulary.stop_ids, 16
    )
    decoder.configure_copy(1, np.array([[0]]), documents, [spec])
    first = vocabulary.encode_ids("five")[0]
    stray = vocabulary.encode_ids('"')[0]
    for q, token in enumerate([stray, first, stray]):
        decoder.start(0, q, 0, [0])
        logits = torch.full((1, size), -10.0)
        logits[0, token] = 1
        decoder.submit(logits, [0], [0], None)
        decoder.harvest(wait=True)
        rows = decoder.rows()
        decoder.feed(
            SimpleNamespace(
                decode_rows=rows,
                tokens=len(rows),
                input_ids=torch.empty(len(rows), dtype=torch.int64),
                meta={},
            )
        )
        logits.fill_(-10)
        logits[0, vocabulary.stop_ids[0]] = 1
        decoder.submit(logits, [], [], rows)
        decoder.harvest(wait=True)
        assert not decoder.active
    assert decoder.answers(0, [0])[0].tolist() == [stray]
    assert decoder.answers(2, [0])[0].tolist() == [stray]
    assert decoder.output_answers(1, [0]) == (["five"], [(0, 4)])
    assert decoder.steps[1].tolist() == [2]


@pytest.mark.parametrize("map_first", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_maps_and_extracts_share_branches_without_changing_results(
    monkeypatch, map_first, gated
):
    from helpers.fakes import cpu_staging
    from helpers.kv_checker import Setup, chain, make_arena
    from test_map_decode import (
        TokenModel,
        batch_decode,
        map_node,
        map_state,
        next_token,
    )

    from quail.backends.quail.executor.operators.extract import ExtractPart
    from quail.backends.quail.executor.operators.filter import ConditionGate
    from quail.backends.quail.executor.operators.map import MapPart
    from quail.backends.quail.executor.parts import execute_parts
    from quail.logical import ColumnRef, RegularPredicate
    from quail.physical import AiExtract, Filter, PortRef
    from quail.physical.base import input_ports

    cpu_staging(monkeypatch)
    monkeypatch.setattr(
        "quail.backends.quail.executor.parts.input_staging", lambda state: None
    )
    texts = ["five days or five years", "five years", "lease for five days"]
    spec, vocabulary = make_spec(*texts, "five days years lease", "or", "for")
    vocabulary.tokenizer.batch_decode = batch_decode
    spec = replace(spec, max_answer_tokens=6, draft_tokens=3, top_tokens=4)
    setup = Setup(path="unified", pages=24, page_tokens=4, budget=128)
    arena = make_arena(setup)
    model = TokenModel(arena)
    state = map_state(setup, arena, model.forward_chunk)
    state.gpu_timing = False
    state.loaded_model.vocabulary = vocabulary
    head = torch.zeros(len(vocabulary.vocab), 1024)
    pool = [*vocabulary.encode_ids("five days years lease"), *vocabulary.stop_ids]
    head[np.array(pool)[np.arange(1024) % len(pool)], np.arange(1024)] = 1
    state.loaded_model.model.quail_full_head = head
    ids = np.arange(len(texts))
    tokens = vocabulary.document_batch(texts, [None] * len(texts))
    extracts = ExtractPart(
        state,
        [
            AiExtract(
                node_id=f"extract:{q}",
                backend_name="quail",
                model="qwen3-4b-fp8",
                spec=replace(spec, name=f"answer{q}"),
            )
            for q in range(2)
        ],
        ids,
        {"documents": {"c": tokens}, "texts": {"c": texts}},
        first=int(map_first),
    )

    def mapped(name, first):
        node = map_node(name, [10, 11], 6, "tree")
        node.node_id, node.rows_port, node.spec.aliases = name, "rows:c", ("c",)
        return MapPart(state, [node], ids, first)

    maps = mapped("summary", 0 if map_first else 2)
    parts = [maps, extracts] if map_first else [extracts, maps]
    after = mapped("after", 3)
    if gated:
        condition = Filter(
            node_id="condition",
            aliases=("c",),
            written_position=0,
            inputs=input_ports((PortRef("extract:1", "rows:c"),)),
            predicate=RegularPredicate(
                ColumnRef("c", "contracts", "answer1"), "is not null"
            ),
        )
        parts.extend([ConditionGate(condition, extracts, extracts), after])
    results = execute_parts(
        state,
        parts,
        extracts.extraction.prefixes,
        extracts.extraction.keys,
        keep_kv=False,
    )
    reference = Decoder(vocabulary, 4, 9, 2, spec)
    for d, document in enumerate(extracts.extraction.documents):
        prefix = list(map(int, document.prefix))
        expected = []
        for _ in range(6):
            token = int(
                head[:, next_token(chain(prefix + [10, 11] + expected))].argmax()
            )
            if token in vocabulary.stop_ids:
                break
            expected.append(token)
        assert (
            results["summary"].outputs["rows:c"].column("summary")[d].as_py()
            == batch_decode([expected])[0]
        )
        for q in range(2):
            answer = document.answers[q]
            prompt = prefix + list(spec.prompt_token_parts[1])
            while not answer.done:
                logits = head[:, next_token(chain(prompt + answer.written))]
                lp = logits - logits.logsumexp(dim=0)
                top, selected = lp.topk(4)
                row = torch.cat(
                    (
                        lp[list(vocabulary.stop_ids)].logsumexp(dim=0)[None],
                        selected.float(),
                        top,
                    )
                ).numpy()
                reference.choose(document, q, row)
            output = results[f"extract:{q}"].outputs["rows:c"]
            assert output.column(f"answer{q}")[d].as_py() == answer.answer
            assert output.column(f"answer{q}_span")[d].as_py() == (
                None
                if answer.span is None
                else dict(start=answer.span[0], end=answer.span[1])
            )
    if gated:
        expected = [
            document.answers[1].answer is not None
            for document in extracts.extraction.documents
        ]
        assert any(expected)
        assert after.asked[0].tolist() == expected
    assert len(set(maps.slots.flat) & set(extracts.slots.flat)) == 0
    assert not arena.accounting.owned


@pytest.mark.parametrize("gated", [False, True])
def test_later_questions_wait_only_for_the_answer_their_condition_reads(
    monkeypatch, gated
):
    from helpers.fakes import cpu_staging, fake_pipeline
    from helpers.kv_checker import Setup, chain, make_arena
    from test_map_decode import Bits, TokenModel, map_node, next_token

    from quail.backends.quail.executor.operators.extract import ExtractPart
    from quail.backends.quail.executor.operators.filter import ConditionGate, FilterPart
    from quail.backends.quail.executor.operators.map import MapPart, branch_parts
    from quail.backends.quail.executor.parts import gated_stages
    from quail.backends.quail.executor.stages import run_stages
    from quail.logical import ColumnRef, RegularPredicate
    from quail.physical import AiExtract

    cpu_staging(monkeypatch)
    text = " ".join(["five"] * 80)
    spec, vocabulary = make_spec(text)
    spec = replace(spec, max_answer_tokens=32, draft_tokens=0, top_tokens=4)
    setup = Setup(path="unified", pages=200, budget=512)
    arena = make_arena(setup)
    state = SimpleNamespace(
        loaded_model=SimpleNamespace(arena=arena, vocabulary=vocabulary)
    )
    nodes = [
        AiExtract(
            node_id=f"extract:{q}",
            backend_name="quail",
            model="qwen3-4b-fp8",
            spec=replace(
                spec,
                name=f"answer{q}",
                prompt_token_parts=(spec.prompt_token_parts[0], (600 + q, 610 + q)),
            ),
        )
        for q in range(3)
    ]
    extracts = ExtractPart(
        state, nodes, [0], {"documents": {"c": None}, "texts": {"c": [text]}}, first=1
    )
    parent = MapPart(state, [map_node("parent", [500, 501], 1)], [0], 0)
    parts = [parent, extracts]
    if gated:
        condition = SimpleNamespace(
            column="answer1",
            predicate=RegularPredicate(
                ColumnRef("c", "contracts", "answer1"), "is not null"
            ),
        )
        parts.append(ConditionGate(condition, extracts, extracts))
    question = [760, 761]
    parts.append(FilterPart(SimpleNamespace(question_token_ids=question), [0], Bits()))
    parts = branch_parts(parts)
    stages = gated_stages(parts)
    prefix = list(map(int, extracts.extraction.prefixes[0]))
    prefixes, keys, tree, dependencies = parts[0].branch_inputs(
        extracts.extraction.prefixes,
        extracts.extraction.keys,
        None,
        setup.page_tokens,
        0,
    )
    word = vocabulary.encode_ids(" five")[0]
    head = torch.zeros(len(vocabulary.vocab), 1024)
    head[word, :] = 1
    stop_column = next_token(chain(prefix + [601, 611] + [word] * 3))
    head[:, stop_column] = 0
    head[vocabulary.stop_ids[0], stop_column] = 1
    torch_ = events_torch(0)
    decoder = MapDecoder(
        torch_,
        head,
        len(prefixes),
        parts[0].answer_limits,
        vocabulary.stop_ids,
        setup.page_tokens,
    )
    parts[0].decoder = decoder
    model = TokenModel(arena)
    observed = []

    def forward(chunk):
        tokens = chunk.input_ids.tolist()
        if any(tokens[i : i + 2] == question for i in range(len(tokens) - 1)):
            observed.append(
                [int(decoder.lengths[q + 1][extracts.slots[q, 0]]) for q in range(3)]
            )
        return model.forward_chunk(chunk)

    run_stages(
        torch_,
        arena,
        fake_pipeline(forward_chunk=forward, canvas_ids=()),
        stages,
        prefixes,
        setup.budget,
        anchor_keys=keys,
        prefix_tree=tree,
        dependencies=dependencies,
        decoder=decoder,
    )
    assert len(observed) == 1
    assert observed[0] == [-1, 3 if gated else -1, -1]
    assert [int(decoder.lengths[q + 1][extracts.slots[q, 0]]) for q in range(3)] == [
        32,
        3,
        32,
    ]
