import asyncio

from baselines.stock_sglang.run import MAX_NUM_SEQS, StockSGLangClient
from baselines.stock_vllm.run import _run_filter_chain


class _FakeAsyncEngine:
    """Answers TRUE for every request; doc 1 stage 1 finishes only
    after doc 0 stage 2 has been submitted."""

    def __init__(self, true_id):
        self.loop = asyncio.new_event_loop()
        self.true_id = true_id
        self.events = []
        self.doc0_stage2_submitted = None

    async def async_generate(self, input_ids=None, sampling_params=None,
                             rid=""):
        if self.doc0_stage2_submitted is None:
            self.doc0_stage2_submitted = asyncio.Event()
        self.events.append(("submit", rid))
        if rid.endswith("-0-1"):
            self.doc0_stage2_submitted.set()
        if rid.endswith("-1-0"):
            await self.doc0_stage2_submitted.wait()
        self.events.append(("finish", rid))
        return {"output_ids": [self.true_id],
                "meta_info": {"prompt_tokens": len(input_ids),
                              "cached_tokens": 0}}


def test_chain_submits_next_stage_before_other_documents_finish():
    engine = _FakeAsyncEngine(true_id=7)
    client = StockSGLangClient.__new__(StockSGLangClient)
    client.engine = engine
    client.filter_budget_tokens = 100

    result = client.run_pipelined_filter_chain(
        {"temperature": 0.0}, [[1] * 5, [2] * 17], [[3] * 3, [4] * 3],
        {7}, tag="test")

    assert result["doc_cap"] == 100 // ((5 + 17) // 2 + 3 + 1)
    assert engine.events.index(("submit", "test-0-1")) < \
        engine.events.index(("finish", "test-1-0"))
    assert result["survivors"] == [0, 1]
    assert result["answers"] == {(0, 1): 1, (0, 2): 1,
                                 (1, 1): 1, (1, 2): 1}
    assert result["requests"] == 4
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
