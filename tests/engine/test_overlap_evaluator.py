"""Differential evaluator for overlapped decode scheduling and contiguous GDN spans."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_kv_reads import cpu_staging  # noqa: E402
from test_map_decode import events_torch  # noqa: E402

from quail.backends.quail.executor.arena import KVArena  # noqa: E402
from quail.backends.quail.executor.gdn import plan_phases  # noqa: E402
from quail.backends.quail.executor.map_decoder import MapDecoder  # noqa: E402
from quail.backends.quail.executor.readout import AsyncAnswers  # noqa: E402
from quail.backends.quail.executor.stage_executor import StageExecutor  # noqa: E402
from quail.backends.quail.executor.stages import Stage  # noqa: E402


class _DeterministicMapPipeline:
    """Mock pipeline whose next token is a deterministic function of input ids.

    Each document starts with a unique tag token (1000 + document) followed by
    prompt tokens, then generates tokens derived from the document tag and
    exact input token fed at each step until reaching its target length. Any
    mismatched device-fed token or leaked KV state changes the output sequence.
    """

    needs_pages = True
    whole_sequences = False
    tree_attention = False
    canvas_ids = ()
    canvas_answer_row = 0
    piecewise_graphs = None
    images = None

    def __init__(self, target_lengths, stop_token, vocab_size, hidden_dim):
        self.target_lengths = target_lengths
        self.stop_token = stop_token
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.fed_tokens = []

    def forward_chunk(self, chunk):
        ids = chunk.input_ids.tolist()
        positions = chunk.positions.tolist()
        rows = len(ids)
        hidden = torch.zeros((rows, self.hidden_dim), dtype=torch.float32)
        # Recover which document owns each row from chunk.layout + decode_rows.
        decode = chunk.decode_rows
        decode_count = 0 if decode is None else len(decode)
        prefill_rows = rows - decode_count
        # Track document id per row for deterministic next-token emission.
        row_doc = [-1] * rows
        cursor = 0
        for key, num_suffixes in chunk.layout:
            if not num_suffixes:
                continue
            # Each prefill row belongs to key until the next layout entry.
            row_doc[cursor] = key
            cursor += 1
        # Map final prefill rows by walking chunk.row_spans.
        for key, row0, _prefix_end, row1 in chunk.row_spans:
            for row in range(row0, row1):
                row_doc[row] = key
        for offset in range(decode_count):
            slot = int(decode.slots[offset])
            row = prefill_rows + offset
            row_doc[row] = slot
            self.fed_tokens.append((slot, int(positions[row]), int(ids[row])))
        for row in range(rows):
            slot = row_doc[row]
            if slot < 0:
                continue
            step = positions[row]
            target = self.target_lengths[slot]
            # Prompt ends at position 3 (length 4): token 0 chosen at pos 3.
            answer_index = step - 3
            if answer_index >= target:
                next_id = self.stop_token
            else:
                # Mix slot, answer_index, and the actual fed input token.
                next_id = 10 + ((slot * 17 + answer_index * 7 + ids[row]) % 40)
            hidden[row, next_id] = 10.0
        return hidden.index_select(0, chunk.final_indices)


def _run_map_scenario(seed, num_documents, num_pages, chunk_budget, in_flight,
                      lag=0):
    rng = np.random.default_rng(seed)
    vocab_size = 64
    stop_token = 2
    limit = 10
    target_lengths = rng.integers(0, limit + 2, size=num_documents).tolist()
    prefixes = [[1000 + index, 5, 6] for index in range(num_documents)]
    suffix = [7]
    fake_torch = events_torch(lag)
    head = torch.eye(vocab_size, dtype=torch.float32)
    arena = KVArena(
        num_layers=1, num_pages=num_pages, page_tokens=4, num_kv_heads=1,
        d_head=4, dtype=torch.float32, device="cpu")
    arena.pinned = False
    pipeline = _DeterministicMapPipeline(
        target_lengths, stop_token, vocab_size, vocab_size)
    decoder = MapDecoder(
        fake_torch, head, num_documents, [limit], [stop_token],
        page_tokens=4, in_flight=in_flight)
    decoder.answer_slots[0] = np.arange(num_documents, dtype=np.int64)

    def _noop_submit(normed):
        return normed

    readout = SimpleNamespace(
        dtype=None, trailing_rows=1, normalize=False,
        submit=_noop_submit,
        result=lambda handle: [None] * handle.shape[0],
    )
    stage = Stage(suffixes=[suffix], readout=readout, single=True,
                  append=True, answer_tokens=limit, answer_index=0)
    initial_free = arena.free_pages
    executor = StageExecutor(
        fake_torch, arena, pipeline, [stage], prefixes, chunk_budget,
        decoder=decoder, summary=False)
    executor.run()
    # Invariant: zero leaked pages after run finishes.
    assert arena.free_pages == initial_free
    for key in range(num_documents):
        assert not arena.is_resident(key)
    answers = [
        decoder.tokens[0][index, :int(decoder.lengths[0, index])].tolist()
        for index in range(num_documents)
    ]
    return answers, pipeline.fed_tokens, decoder


def test_overlapped_map_decode_matches_synchronous_oracle(monkeypatch):
    """Verify overlapped in_flight=2,3 decode matches synchronous in_flight=0."""
    cpu_staging(monkeypatch)
    for seed, num_documents, num_pages, chunk_budget in (
        (1, 8, 64, 32),
        (2, 12, 48, 24),
        (3, 10, 18, 20),  # tight pages: forces evictions and replays
        (4, 6, 32, 8),    # tight chunk budget: small mixed and decode passes
        (5, 1, 16, 16),   # single document stopping immediately or mid-run
    ):
        oracle_answers, _oracle_fed, _ = _run_map_scenario(
            seed, num_documents, num_pages, chunk_budget, in_flight=0)
        for depth, lag in ((2, 0), (3, 0), (3, 2)):
            actual_answers, fed_tokens, decoder = _run_map_scenario(
                seed, num_documents, num_pages, chunk_budget, in_flight=depth,
                lag=lag)
            assert actual_answers == oracle_answers
            # Device-host agreement: every non-speculative fed token matches
            # the recorded token at the same answer index.
            for slot, position, token in fed_tokens:
                answer_index = position - 4
                length = int(decoder.lengths[0, slot])
                if 0 <= answer_index < length:
                    assert token == actual_answers[slot][answer_index]
            # Pinned buffer pool stays bounded by in_flight + 2.
            assert len(decoder._host_pool) <= depth + 2


def test_async_answers_reuses_host_buffer():
    fake_torch = events_torch(0)
    rows = SimpleNamespace(
        logits=lambda normed: (normed[:, 0], normed[:, 1]))
    readout = AsyncAnswers(fake_torch, rows)
    first = readout.result(readout.submit(torch.tensor([[2.0, 1.0], [0.0, 3.0]])))
    second = readout.result(readout.submit(torch.tensor([[0.0, 1.0], [4.0, 1.0]])))
    assert first == [1, 0]
    assert second == [0, 1]
    assert len(readout.available) == 1


def test_gdn_contiguous_span_matches_phase_rows():
    arena = SimpleNamespace(
        read_state=lambda keys: None,
        done_reading=lambda keys: None,
        has_checkpoint=lambda key: False,
        checkpoint_slot=lambda key: key,
        checkpoint_tokens=lambda key: 4,
        extension=lambda key: None,
        record_checkpoint=lambda key, tokens: None,
        record_extension=lambda key, tokens: None,
        extension_slot=lambda key: key + 10,
        accounting=SimpleNamespace(slots={0: 0, 1: 1, 2: 2}, slot_tokens={}),
    )
    groups = [
        dict(key=0, fresh=True, f=4, start=0, row0=0, prefix_end=4,
             suffix_spans=(np.array([]), np.array([]))),
        dict(key=1, fresh=False, f=4, start=0, row0=4, prefix_end=4,
             suffix_spans=(np.array([4]), np.array([5])), kept=1),
        dict(key=2, fresh=False, f=4, start=0, row0=5, prefix_end=5,
             suffix_spans=(np.array([5]), np.array([6])), kept=1),
    ]
    phases = plan_phases(
        arena, groups,
        lambda _name, vals, dtype: torch.as_tensor(vals, dtype=dtype))
    assert len(phases) == 2
    assert phases[0].contiguous_span == (0, 4)
    assert phases[1].contiguous_span == (4, 6)
    for phase in phases:
        low, high = phase.contiguous_span
        assert phase.rows.tolist() == list(range(low, high))

    # Validation: negative in_flight rejected by assert.
    with pytest.raises(AssertionError):
        MapDecoder(events_torch(0), torch.eye(8), 2, [4], [2], 4, in_flight=-1)
