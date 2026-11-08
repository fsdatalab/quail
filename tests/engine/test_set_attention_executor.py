"""Set-attention executor pieces: pass building, readouts, and the set mask."""

from types import SimpleNamespace

import numpy as np
import pytest

from quail.specs import D1_2B_BF16


def test_an_anchor_without_partners_at_the_next_join_settles_unmatched():
    from quail.backends.quail.executor.operators.join import settled_rows

    # anchor 0 matches at join 0, then has no partner to ask at join 1;
    # anchor 1 matches at both; anchor 2 matches nothing at join 0
    every = [{0: [True], 1: [True], 2: [False]}, {1: [False, True]}]
    asked = [[0, 1, 2], [0, 1]]

    assert settled_rows(every, asked) == {0: [], 1: [False, True], 2: [False]}


class _Pieces:
    """A tokenizer of fixed pieces, longest first: a mark joins a blank line."""

    pieces = {1: "a", 2: ".", 3: "\n\n", 4: ".\n\n", 5: "Q"}

    def __init__(self):
        self.calls = 0

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)

    def __call__(self, text):
        self.calls += 1
        ids = []
        while text:
            i, piece = max(((i, p) for i, p in self.pieces.items()
                            if text.startswith(p)), key=lambda e: len(e[1]))
            ids.append(i)
            text = text[len(piece):]
        return ids


def test_one_mixed_pass_decodes_a_yes_no_a_choice_and_a_read():
    from quail.backends.quail.executor.pass_groups import (
        SetQuestion,
        _decode,
        _layout,
    )

    yes_no = SetQuestion("yes_no", (), (), (1,), ())
    choice = SetQuestion("choice", (), (), (2,), (), spellings=((5,), (6,)))
    read = SetQuestion("read", (), (), (3,), (), slots=(-2, 0))
    targets = [5, 6, 20, 30]               # options A and B, then yes and no
    kinds, places, width, reads = _layout([yes_no, choice, read], [0, 1, 2],
                                          [20], [30], targets)
    # two documents: code logits of the two code slots, then the read
    values = np.array([[0, 0, 3, 1,   9, 1, 0, 0,   4, 6, 0.9],
                       [0, 0, 1, 3,   1, 9, 0, 0,   0, 0, 0.1]], dtype=float)

    margins, probabilities, spans = _decode(values, places, width, 4, 1.0)

    assert kinds == ["code", "code", "start", "end"] and reads == 1
    assert margins.tolist() == [2.0, -2.0]
    assert probabilities.argmax(-1).tolist() == [0, 1]
    assert spans.tolist() == [[4, 6, 0.9], [0, 0, 0.1]]


def test_a_pointer_read_takes_the_likeliest_span_or_nothing():
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.readout import pointer_spans

    width = 4
    state = torch.zeros(5, width)
    state[2, 0] = state[3, 1] = 1.0   # rows 2 and 3 answer the first read
    state[0, 2] = 1.0                 # row 0 stands for nothing
    slots = torch.zeros(4, width)
    slots[0, 0] = slots[1, 1] = 40.0  # read 1: start at row 2, end at row 3
    slots[2, 2] = slots[3, 2] = 40.0  # read 2: both slots on row 0
    # one chunk: the same answer twice, then one of four document rows
    rows = torch.cat((state, slots, state, slots, state[:4], slots))
    firsts, states = np.array([0, 9, 18]), np.array([5, 5, 4])
    slot_rows = (firsts + states)[:, None] + np.arange(4)

    spans = pointer_spans(torch, torch.nn.functional, rows,
                          rows[torch.from_numpy(slot_rows.reshape(-1))],
                          np.repeat(firsts, 2), np.repeat(states, 2), 512)

    assert spans[0, :2].tolist() == [2.0, 3.0] and spans[0, 2] > 0.99
    assert spans[1, :2].tolist() == [0.0, 0.0] and spans[1, 2] < 0.01
    assert torch.equal(spans[2:4], spans[0:2])
    assert spans[4, :2].tolist() == [2.0, 3.0]


def _varlen_attention(torch):
    """FlashAttention's varlen call, in plain torch, returning (out, lse)."""
    def fa(q, k, v, cu_q, cu_k, max_q, max_k, causal, window=None):
        outs, lses = [], []
        groups = q.shape[1] // k.shape[1]
        for i in range(len(cu_q) - 1):
            qi = q[cu_q[i]:cu_q[i + 1]].float()
            ki = k[cu_k[i]:cu_k[i + 1]].float().repeat_interleave(groups, 1)
            vi = v[cu_k[i]:cu_k[i + 1]].float().repeat_interleave(groups, 1)
            scores = torch.einsum("thd,shd->hts", qi, ki) / qi.shape[-1] ** 0.5
            outs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), vi))
            lses.append(scores.logsumexp(-1))
        return torch.cat(outs), torch.cat(lses, dim=1)
    return fa


def merge_reference(out, rows, over_state, lse_state, own, lse_own):
    """lfm3_kernels.merge_questions in plain torch."""
    a, b = lse_state.transpose(0, 1)[:, :, None], lse_own.transpose(0, 1)[:, :, None]
    top = a.maximum(b)
    wa, wb = (a - top).exp(), (b - top).exp()
    merged = (over_state.float() * wa + own.float() * wb) / (wa + wb)
    out.index_copy_(0, rows, merged.to(out.dtype))
    return out


def test_set_attention_lets_the_state_read_every_question_and_each_question_itself(
        monkeypatch):
    torch = pytest.importorskip("torch")
    from quail.backends.quail.executor.models import lfm3_kernels
    from quail.backends.quail.executor.models.lfm3 import Lfm3Pipeline

    # the Triton merge needs a GPU; tests/gpu checks it against this one
    monkeypatch.setattr(lfm3_kernels, "merge_questions", merge_reference)

    torch.manual_seed(0)
    # pass 0: rows 0-4 state, questions of 2, 3, and 1 rows; pass 1: 4
    # rows, one question of 2 rows
    passes = ((0, 5, (2, 3, 1)), (11, 13, (2,)))
    group = [-1] * 5 + [1, 1, 2, 2, 2, 3] + [-1] * 2 + [1, 1]
    sequence = [0] * 11 + [1] * 4
    rows, heads, kv_heads, dim = 15, 4, 2, 8
    q = torch.randn(rows, heads, dim)
    k, v = torch.randn(rows, kv_heads, dim), torch.randn(rows, kv_heads, dim)
    pipeline = Lfm3Pipeline.__new__(Lfm3Pipeline)
    pipeline.engine = SimpleNamespace(torch=torch, _fa=_varlen_attention(torch))
    meta = dict(passes=passes, cu_a=torch.tensor([0, 11, 15]), max_a=11)
    chunk = SimpleNamespace(input_ids=torch.zeros(rows, dtype=torch.long))
    meta["set_rows"] = pipeline._plan_set_rows(chunk, meta)

    out = pipeline._set_attention(q, k, v, meta)

    sees = torch.tensor([[sequence[i] == sequence[j] and (
        group[i] == -1 or group[j] == -1 or group[i] == group[j])
        for j in range(rows)] for i in range(rows)])
    scores = torch.einsum("thd,shd->hts", q, k.repeat_interleave(2, 1)) / dim ** 0.5
    weights = scores.masked_fill(~sees, float("-inf")).softmax(-1)
    expected = torch.einsum("hts,shd->thd", weights, v.repeat_interleave(2, 1))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class _OneGroup:
    """Asks a pass group by handing back answers given in advance."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def group(self, pass_group, kept):
        self.asked.append(len(kept))
        return self.answers


def test_a_label_packed_with_a_filter_answers_only_the_rows_the_filter_keeps():
    from quail.backends.quail.executor import pipeline
    from quail.backends.quail.executor.pass_groups import GroupAnswers
    from quail.physical import AiClassify, AiFilter, ClassifySpec

    spec = ClassifySpec(query_template="", arguments=(), expected_rows=4,
                        estimated_seconds=0.0, name="mood", aliases=("r",),
                        labels=("happy", "sad"), prompt_token_parts=((), ()),
                        scoring="set_choice", label_token_ids=((5,), (6,)))
    nodes = (AiFilter(node_id="f", alias="r", pass_group=0),
             AiClassify(node_id="c", spec=spec, pass_group=0))
    # the filter passes rows 0 and 2; the label is answered for all four
    ask = _OneGroup(GroupAnswers(np.arange(4), [
        np.array([1.0, -1.0, 2.0, -3.0]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])],
        [4, 4], tokens=100))
    state = SimpleNamespace(loaded_model=SimpleNamespace(model_spec=D1_2B_BF16))

    results = pipeline.set_steps(state, nodes, [10, 11, 12, 13], [[0]] * 4, ask)

    labels = results["c"].outputs["rows:r"]
    assert ask.asked == [4]
    assert results["f"].outputs["rows:r"].ids["r"].to_pylist() == [10, 12]
    assert labels.ids["r"].to_pylist() == [10, 12]
    assert labels.column("mood").to_pylist() == ["happy", "sad"]
    assert (results["f"].metrics.fresh_tokens, results["c"].metrics.fresh_tokens) \
        == (100, 0)


def test_a_score_packed_with_a_label_reads_its_answers_for_every_row():
    from quail.backends.quail.executor import pipeline
    from quail.backends.quail.executor.pass_groups import GroupAnswers
    from quail.physical import AiClassify, AiScore, ClassifySpec, ScoreSpec

    score = ScoreSpec(query_template="", arguments=(), expected_rows=3,
                      estimated_seconds=0.0, name="fun", aliases=("r",),
                      prompt_token_parts=((), ()), state_tail_tokens=1)
    label = ClassifySpec(query_template="", arguments=(), expected_rows=3,
                         estimated_seconds=0.0, name="mood", aliases=("r",),
                         labels=("happy", "sad"), prompt_token_parts=((), ()),
                         scoring="set_choice", label_token_ids=((5,), (6,)))
    nodes = (AiScore(node_id="s", spec=score, pass_group=0),
             AiClassify(node_id="c", spec=label, pass_group=0))

    ask = _OneGroup(GroupAnswers(np.arange(3), [
        np.array([0.9, 0.2, 0.3]),
        np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])], [3, 3], tokens=100))
    state = SimpleNamespace(loaded_model=SimpleNamespace(model_spec=D1_2B_BF16))

    results = pipeline.set_steps(state, nodes, [10, 11, 12], [[0]] * 3, ask)

    scored = results["s"].outputs["rows:r"]
    assert scored.ids["r"].to_pylist() == [10, 11, 12]
    assert scored.column("fun").to_pylist() == pytest.approx([0.9, 0.2, 0.3])
    assert results["c"].outputs["rows:r"].column("mood").to_pylist() \
        == ["happy", "sad", "happy"]
    assert (results["s"].metrics.fresh_tokens, results["c"].metrics.fresh_tokens) \
        == (100, 0)


def test_a_later_step_asks_only_the_documents_the_step_before_kept(monkeypatch):
    from helpers.fakes import (
        cpu_arena,
        fake_pack,
        fake_pipeline,
        fake_readout,
        fake_torch,
    )

    from quail.backends.quail.executor import chunk
    from quail.backends.quail.executor.set_passes import (
        Passes,
        own_stage,
        run_passes,
    )
    from quail.backends.quail.executor.stages import Stage, StageRequests

    monkeypatch.setattr(chunk, "pack_chunk", fake_pack)

    def forward_chunk(packed):
        # a pass's answer: whether its document token is even, then, at
        # the second step, whether it is a multiple of four
        return [int(spec["prefix"][0]) % (2 if suffix[0] == 7 else 4) == 0
                for spec in packed.specs for suffix in spec["suffixes"]]

    state = SimpleNamespace(
        torch=fake_torch(), chunk_tokens=8, gpu_timing=False,
        loaded_model=SimpleNamespace(
            arena=cpu_arena(64), input_staging=None,
            pipeline=fake_pipeline(whole_sequences=True,
                                   forward_chunk=forward_chunk)))
    readout = fake_readout()
    documents = [[10], [11], [12], [13], [16]]

    first = run_passes(state, [Stage(suffixes=[[7]], readout=readout)], Passes(
        stages=[0] * 5, states=documents, suffix_tokens=[1] * 5,
        owners=list(range(5))), label="first step")
    kept = [index for index, value in zip(first.owners, first.values) if value]
    second = Passes(stages=[0] * len(kept), states=[documents[i] for i in kept],
                    suffix_tokens=[2] * len(kept), owners=kept,
                    requests=[StageRequests([[8, 9]]) for _ in kept])
    run_passes(state, [own_stage(state, readout, "second step", second)], second,
               label="second step")

    assert {index: bool(value) for index, value in
            zip(second.owners, second.values)} == {0: False, 2: True, 4: True}
    # each of the three kept documents' passes: one state token, two asked
    assert second.tokens == 3 * (1 + 2)
