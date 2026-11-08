"""Copy decoding agrees with the host reference token rule on CPU."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch
from helpers.conditions import condition as sql_condition
from helpers.extract_fakes import make_spec
from helpers.extract_reference import Decoder, add_answers
from helpers.extract_reference import ReferenceInputs as ExtractionInputs
from test_map_decode import events_torch

from quail.backends.quail.executor.map_decoder import MapDecoder
from quail.execution.answer_stops import AnswerStop


# a draft policy matters only with draft tokens, and drafts only past one token
# a row budget below every draft trims drafts, never answers
@pytest.mark.parametrize(
    "draft_tokens,draft_policy,spellings,answer_limit,row_budget",
    [
        (0, "fixed", False, 1, None),
        (4, "fixed", True, 8, None),
        (4, "adaptive", False, 8, None),
        (4, "fixed", True, 8, 40),
        (4, "adaptive", False, 8, 40),
    ],
)
def test_copy_answers_and_spans_match_the_recurrent_token_rule(
    draft_tokens, draft_policy, spellings, answer_limit, row_budget
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
    spec = spec.replace(
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
    pool = vocabulary.tokenizer(pool_text)
    for q in range(num_questions):
        for document in range(num_documents):
            for at in range(16):
                token = int(rng.choice(pool))
                if at > rng.integers(1, 7):
                    token = vocabulary.tokenizer.stop_ids[0]
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
                        torch.logsumexp(
                            lp[:, vocabulary.tokenizer.stop_ids], dim=1)[:, None],
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
        vocabulary.tokenizer.stop_ids,
        16,
        workspace_bytes=64 * 2**20,
        row_budget=row_budget,
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
        if row_budget is not None:
            assert len(rows) <= max(row_budget, len(decoder.running()))
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


# every setting appears in some case
@pytest.mark.parametrize("draft_tokens,limit,filtered,lag,pages", [
    (3, 2, False, 0, 24),
    (0, None, True, 3, 300),
])
def test_copy_branches_match_reference_after_filters_and_limits(
    monkeypatch, draft_tokens, limit, filtered, lag, pages, state_slots=0
):
    from helpers.fakes import cpu_staging, fake_pipeline
    from helpers.kv_checker import Setup, chain, make_arena
    from test_map_decode import Bits, TokenModel, next_token

    from quail.backends.quail.executor.model_operator import gated_stages
    from quail.backends.quail.executor.operators.extract import ExtractOperator
    from quail.backends.quail.executor.operators.filter import FilterOperator
    from quail.backends.quail.executor.stages import run_stages
    from quail.physical import AiExtract

    cpu_staging(monkeypatch)
    rng = np.random.default_rng(19)
    texts = [
        " ".join(rng.choice(["five", "years", "days", "lease"], 25)) for _ in range(12)
    ]
    spec, vocabulary = make_spec(*texts, "five years days lease")
    spec = spec.replace(
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
            spec=spec.replace(
                name=f"answer{q}",
                prompt_token_parts=(
                    spec.prompt_token_parts[0],
                    (*spec.prompt_token_parts[1], q + 10),
                ),
            ),
        )
        for q in range(3)
    ]
    setup = Setup(path="unified", pages=pages, page_tokens=4, budget=128,
                  state_slots=state_slots)
    arena = make_arena(setup)
    torch_ = events_torch(lag)
    model = TokenModel(arena)
    state = SimpleNamespace(
        loaded_model=SimpleNamespace(arena=arena, vocabulary=vocabulary,
                                     tokenizer=vocabulary.tokenizer)
    )
    ids = list(range(len(texts)))
    inputs = {
        "documents": {
            "c": (
                None if lag else vocabulary.document_batch(
                    texts, [None] * len(texts))
            )
        },
        "texts": {"c": texts},
    }
    part = ExtractOperator(state, nodes, ids, inputs)
    parts = [part]
    question = [700, 701]
    if filtered:
        parts.insert(
            0, FilterOperator(SimpleNamespace(question_token_ids=question), ids, Bits())
        )
    stages = gated_stages(parts)
    prefixes, keys, tree, dependencies = part.branch_inputs(
        part.extraction.prefixes,
        part.extraction.keys,
        None,
        setup.page_tokens,
        len(stages) - 1,
    )
    pool = [*vocabulary.tokenizer("five years days lease"),
            *vocabulary.tokenizer.stop_ids]
    head = torch.zeros(len(vocabulary.vocab), 1024)
    head[rng.choice(pool, 1024), np.arange(1024)] = 1
    decoder = MapDecoder(
        torch_,
        head,
        len(prefixes),
        part.answer_limits,
        vocabulary.tokenizer.stop_ids,
        setup.page_tokens,
        workspace_bytes=64 * 2**20,
    )
    part.decoder = decoder
    survivors = []
    fed = []

    def forward_chunk(chunk):
        if state_slots and chunk.decode_rows is not None:
            rows = chunk.decode_rows
            tokens = chunk.input_ids[-len(rows):].tolist()
            fed.extend((int(decoder.map[slot]), int(slot),
                        int(position - decoder.origin[slot]), token)
                       for slot, position, token in zip(
                           rows.slots, rows.positions, tokens))
        return model.forward_chunk(chunk)

    def settled(anchor, survived, row):
        if anchor < len(texts) and survived:
            survivors.append(anchor)
        arena.free_key(keys[anchor])

    every, timed_spans, fresh = run_stages(
        torch_,
        arena,
        fake_pipeline(forward_chunk=forward_chunk, canvas_ids=()),
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
        attention_mode=setup.path if state_slots else None,
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
    add_answers(part.extraction)
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
                        torch.logsumexp(
                            lp[list(vocabulary.tokenizer.stop_ids)], dim=0)[None],
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
    part.read_answers(every)
    part.close(fresh, timed_spans, {}, len(stages) - 1)
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
    if state_slots:
        assert len(arena.accounting.free_slots) == state_slots
        if draft_tokens:
            assert any(position < decoder.lengths[q][slot] - 1
                       and token != decoder.tokens[q][slot, position]
                       for q, slot, position, token in fed), (
                           "the workload must continue after rejecting a draft")


@pytest.mark.parametrize("draft_tokens,pages,lag,filtered", [(3, 300, 3, True)])
def test_hybrid_copy_answers_match_reference_after_rejected_drafts(
        monkeypatch, draft_tokens, pages, lag, filtered):
    test_copy_branches_match_reference_after_filters_and_limits(
        monkeypatch, draft_tokens=draft_tokens, limit=None, filtered=filtered,
        lag=lag, pages=pages, state_slots=120)


# every setting appears in some case
@pytest.mark.parametrize(
    "map_first,tested,extract_attention,map_attention,pages",
    [
        (False, "extract", "tree", "tree", 300),
        (True, "map", "unified", "unified", 24),
        (False, None, "tree", "unified", 24),
    ],
)
def test_maps_and_extracts_share_branches_without_changing_results(
    monkeypatch, map_first, tested, extract_attention, map_attention, pages
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

    from quail.backends.quail.executor.operators.extract import ExtractOperator
    from quail.backends.quail.executor.operators.filter import ConditionGate
    from quail.backends.quail.executor.operators.map import MapOperator
    from quail.backends.quail.executor.region import RegionDocuments, RegionExecution
    from quail.physical import AiExtract, Filter, PortRef, input_ports

    cpu_staging(monkeypatch)
    monkeypatch.setattr(
        "quail.backends.quail.executor.region.input_staging", lambda state: None
    )
    texts = ["five days or five years", "five years", "lease for five days"]
    spec, vocabulary = make_spec(*texts, "five days years lease", "or", "for")
    vocabulary.tokenizer.decode_batch = batch_decode
    spec = spec.replace(max_answer_tokens=6, draft_tokens=3, top_tokens=4)
    setup = Setup(path="unified", pages=pages, page_tokens=4, budget=128)
    arena = make_arena(setup)
    model = TokenModel(arena)
    state = map_state(setup, arena, model.forward_chunk)
    state.gpu_timing = False
    state.loaded_model.vocabulary = vocabulary
    state.loaded_model.tokenizer = vocabulary.tokenizer
    head = torch.zeros(len(vocabulary.vocab), 1024)
    pool = [*vocabulary.tokenizer("five days years lease"),
            *vocabulary.tokenizer.stop_ids]
    head[np.array(pool)[np.arange(1024) % len(pool)], np.arange(1024)] = 1
    state.loaded_model.model.quail_full_head = head
    ids = np.arange(len(texts))
    tokens = vocabulary.document_batch(texts, [None] * len(texts))
    extracts = ExtractOperator(
        state,
        [
            AiExtract(
                node_id=f"extract:{q}",
                backend_name="quail",
                model="qwen3-4b-fp8",
                attention=extract_attention,
                spec=spec.replace(name=f"answer{q}"),
            )
            for q in range(2)
        ],
        ids,
        {"documents": {"c": tokens}, "texts": {"c": texts}},
        first=int(map_first),
    )

    def mapped(name, first):
        node = map_node(name, [10, 11], 6, map_attention)
        node.node_id, node.rows_port, node.spec.aliases = name, "rows:c", ("c",)
        return MapOperator(state, [node], ids, first)

    maps = mapped("summary", 0 if map_first else 2)
    parts = [maps, extracts] if map_first else [extracts, maps]
    after = mapped("after", 3)
    if tested:
        producer = extracts if tested == "extract" else maps
        column = "answer1" if tested == "extract" else "summary"
        source = "extract:1" if tested == "extract" else "summary"
        condition = Filter(
            node_id="condition",
            aliases=("c",),
            written_position=0,
            inputs=input_ports((PortRef(source, "rows:c"),)),
            predicate=sql_condition(
                f"{column} IS NOT NULL" if tested == "extract"
                else f"{column} <> ''", {column: pa.string()}, "c", "contracts"),
        )
        parts.extend([ConditionGate(condition, {column: producer}, producer, None),
                      after])
    region = RegionExecution(SimpleNamespace(query=state), SimpleNamespace(state={}))
    results = region.run_operators(
        parts, RegionDocuments(extracts.extraction.prefixes, extracts.extraction.keys))
    add_answers(extracts.extraction)
    reference = Decoder(vocabulary, 4, 9, 2, spec)
    for d, document in enumerate(extracts.extraction.documents):
        prefix = list(map(int, document.prefix))
        expected = []
        for _ in range(6):
            token = int(
                head[:, next_token(chain(prefix + [10, 11] + expected))].argmax()
            )
            if token in vocabulary.tokenizer.stop_ids:
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
                        lp[list(vocabulary.tokenizer.stop_ids)].logsumexp(dim=0)[None],
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
    if tested:
        expected = ([document.answers[1].answer is not None
                     for document in extracts.extraction.documents]
                    if tested == "extract" else [bool(value) for value in
                        results["summary"].outputs["rows:c"].column("summary").to_pylist()])
        assert any(expected)
        assert after.asked[0].tolist() == expected
    assert len(set(maps.slots.flat) & set(extracts.slots.flat)) == 0
    shared = "tree" in (extract_attention, map_attention)
    assert extracts.decoder.share == shared
    # Tight KV can serialize short answers before sibling branches decode.
    if pages == 300 or not shared:
        assert (model.shared_reads > 0) == shared
    reads = extracts.decoder.read_tokens
    assert (reads["alone"] + reads["own"] + reads["shared"] < reads["all"]) == (
        model.shared_reads > 0
    )
    assert not arena.accounting.owned


def test_copy_tables_match_each_documents_candidate_index():
    from quail.backends.quail.executor.extract.candidates import CandidateIndex
    from quail.backends.quail.executor.extract.copy import copy_tables

    texts = ["five years five days", "", "lease lease years", "days five foo"]
    spec, vocabulary = make_spec(*texts, "foobarbaz barbaz")
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(state, [spec], vocabulary, np.arange(len(texts)),
                                 None, texts)
    tables = copy_tables(documents)
    for d, document in enumerate(documents.documents):
        scan = document.scan
        index = CandidateIndex.of(scan, vocabulary)
        start, end = tables.position_offsets[d:d + 2]
        assert tables.position_local_keys[start:end].tolist() == index.inverse.tolist()
        keys = np.arange(tables.document_key_starts[d], tables.document_key_ends[d])
        assert tables.position_keys[start:end].tolist() == keys[index.inverse].tolist()
        rows = np.concatenate([
            np.arange(tables.key_option_starts[key],
                      tables.key_option_starts[key] + tables.key_option_counts[key])
            for key in keys] or [np.zeros(0, dtype=np.int64)])
        assert tables.option_tokens[rows].tolist() == index.token.tolist()
        assert tables.filler_chars[rows].tolist() == index.length.tolist()
        assert tables.filler_tokens[rows].tolist() == scan.tokens[
            index.anchor[index.segment]].tolist()
    assert tables.num_keys == max(
        len(CandidateIndex.of(document.scan, vocabulary).kids)
        for document in documents.documents)
    assert np.all(np.diff(tables.option_lookup) >= 0)


def test_an_extraction_goes_on_while_any_live_position_can_pass_its_test():
    # the first token " State" also begins "state courts", the first
    # position in document order; only the second can pass the test
    texts = ["Disputes go to the state courts. The laws of the State of California"
             " govern.",
             "Disputes go to the state courts. The laws of the State of New York"
             " govern."]
    spec, vocabulary = make_spec(*texts)
    spec = spec.replace(draft_tokens=0, max_answer_tokens=8, top_tokens=4)
    state = SimpleNamespace(loaded_model=SimpleNamespace(arena=SimpleNamespace()))
    documents = ExtractionInputs(
        state, [spec], vocabulary, np.arange(len(texts)), None, texts)
    num_tokens = len(vocabulary.vocab)
    logits = torch.full((len(texts), 8, num_tokens), -8.0)
    for document, answer in enumerate([" State of California", " State of New York"]):
        written = [*vocabulary.tokenizer(answer), vocabulary.tokenizer.stop_ids[0]]
        for at, token in enumerate(written):
            logits[document, at, token] = 8.0
    test = sql_condition("ends = 'State of California'", {"ends": pa.string()}, "c")
    decoder = MapDecoder(events_torch(0), torch.eye(num_tokens), len(texts), [8],
                         vocabulary.tokenizer.stop_ids, 16,
                         workspace_bytes=64 * 2**20)
    decoder.answer_stops[0] = AnswerStop("ends", (test,), (), False)
    slots = np.arange(len(texts))[None]
    decoder.configure_copy(0, slots, documents, [spec])
    for slot in slots[0]:
        decoder.start(int(slot), 0, 0, [0, 1])
    decoder.submit(logits[:, 0], np.arange(len(texts)), slots[0], None)
    decoder.harvest(wait=True)
    while decoder.active:
        rows = decoder.rows()
        decoder.feed(SimpleNamespace(
            decode_rows=rows, tokens=len(rows),
            input_ids=torch.empty(len(rows), dtype=torch.int64), meta={}))
        decoder.submit(logits[rows.slots, rows.positions + 1], [], [], rows)
        decoder.harvest(wait=True)
    values, _ = decoder.output_answers(0, slots[0])
    # neither position of New York's " State" can become California, so
    # it stops at once and holds the first position's text
    assert values == ["State of California", "state"]
    assert decoder.lengths[0].tolist() == [3, 1]
    assert decoder.stopped_early[0] == 1
