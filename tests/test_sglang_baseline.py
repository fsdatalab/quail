from baselines.stock_sglang.run import MAX_NUM_SEQS, StockSGLangClient
from baselines.stock_vllm.run import _run_filter_chain


class _WaveRecordingEngine:
    """Answers by a fixed (document, stage) truth table.

    Documents are identified by their body length, stages by the
    question length, so the recorded waves can be decoded.
    """

    def __init__(self, verdicts, body_lengths, question_lengths):
        self.verdicts = verdicts
        self.body_lengths = body_lengths
        self.question_lengths = question_lengths
        self.waves = []

    def generate(self, input_ids=None, sampling_params=None):
        wave = []
        outs = []
        for ids in input_ids:
            found = None
            for index, body in enumerate(self.body_lengths):
                for stage, tail in enumerate(self.question_lengths):
                    if len(ids) == body + tail:
                        found = (index, stage)
            wave.append(found)
            bit = 7 if self.verdicts[found] else 8
            outs.append({"output_ids": [bit],
                         "meta_info": {"prompt_tokens": len(ids),
                                       "cached_tokens": 0}})
        self.waves.append(wave)
        return outs

    def flush_cache(self):
        return True


def _client(engine, budget_tokens):
    client = StockSGLangClient.__new__(StockSGLangClient)
    client.engine = engine
    client.filter_budget_tokens = budget_tokens
    return client


def test_chain_advances_documents_in_waves_under_the_cap():
    # Bodies 10/20/30 tokens, questions 3/4 tokens: every (doc, stage)
    # pair has a distinct prompt length.
    verdicts = {(0, 0): True, (0, 1): True,
                (1, 0): False,
                (2, 0): True, (2, 1): False}
    engine = _WaveRecordingEngine(verdicts, [10, 20, 30], [3, 4])
    # mean request = 20 + 4 + 1 = 25; budget 50 -> two admission slots
    client = _client(engine, 50)

    result = client.run_pipelined_filter_chain(
        {"temperature": 0.0}, [[1] * 10, [2] * 20, [3] * 30],
        [[4] * 3, [5] * 4], {7}, tag="test")

    assert result["doc_cap"] == 2
    # Wave 1: documents 0 and 1 on stage 1. Wave 2: document 0 has
    # advanced to stage 2 while document 2 takes the freed slot -- the
    # pipelining property, one request per alive document.
    assert engine.waves[0] == [(0, 0), (1, 0)]
    assert engine.waves[1] == [(0, 1), (2, 0)]
    assert engine.waves[2] == [(2, 1)]
    assert result["survivors"] == [0]
    assert result["answers"] == {(0, 1): 1, (0, 2): 1,
                                 (1, 1): 0,
                                 (2, 1): 1, (2, 2): 0}
    assert result["requests"] == 5
    assert result["max_num_seqs"] == MAX_NUM_SEQS


def test_pipelined_dispatch_prefers_the_client_chain():
    class ChainOnlyClient:
        def run_pipelined_filter_chain(self, sp, body_ids, question_ids,
                                       true_ids, tag=""):
            assert len(body_ids) == 2
            assert len(question_ids) == 1
            return dict(wall=0.5, survivors=[1],
                        answers={(0, 1): 0, (1, 1): 1},
                        requests=2, prompt_tokens=40, cached_tokens=10,
                        doc_cap=2)

    class CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    result = _run_filter_chain(
        ChainOnlyClient(), object(), {1}, ["Is {0} good?"],
        ["one", "two"], CharTokenizer(), "pipelined",
        capacity={"kv_cache_size_tokens": 1}, tag="t")

    assert result["survivors"] == [1]
    assert result["stages"] == [dict(stage=1, n_in=2, n_out=1)]
    assert result["fresh_tokens"] == 30
