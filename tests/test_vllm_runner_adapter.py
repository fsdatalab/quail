from types import SimpleNamespace

from docengine.runtime import vllm_runner
from docengine.runtime.batching import BatchLimits, VariableLengthBatchPacker
from docengine.runtime.custom import DocEngineRuntime
from docengine.runtime.kv import KVPageAllocator
from docengine.runtime.protocol import FilterQuery
from docengine.runtime.vllm_runner import VLLMModelRunner


class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CachedRecord(Record):
    @classmethod
    def make_empty(cls):
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


class EncoderStats:
    pass


class FakeExecutor:
    def __init__(self):
        self.outputs = []

    def execute_model(self, scheduler_output):
        self.outputs.append(scheduler_output)
        sampled_ids = []
        req_ids = []
        for request in scheduler_output.scheduled_new_reqs:
            if request.sampling_params is not None:
                req_ids.append(request.req_id)
                sampled_ids.append([99])
        return SimpleNamespace(
            req_ids=req_ids,
            sampled_token_ids=sampled_ids,
        )


def fake_types():
    return CachedRecord, Record, EncoderStats, Record


def test_adapter_builds_scheduler_output_without_vllm_scheduler(monkeypatch):
    monkeypatch.setattr(vllm_runner, "_scheduler_types", fake_types)
    query = FilterQuery.from_sequences(
        body_token_ids=[[1, 2, 3, 4]],
        question_token_ids=[[5, 6]],
        yes_token_ids=[99],
    )
    kv = KVPageAllocator(
        total_pages=8,
        page_size_tokens=4,
        bytes_per_token=8,
    )
    executor = FakeExecutor()
    sampling = object()
    runner = VLLMModelRunner(
        query=query,
        kv=kv,
        model_executor=executor,
        sampling_params=sampling,
    )
    runtime = DocEngineRuntime(
        query=query,
        runner=runner,
        packer=VariableLengthBatchPacker(BatchLimits(
            max_new_tokens=16,
            max_sequences=4,
            max_temporary_bytes=1024,
        )),
        kv=kv,
    )
    result = runtime.run()
    assert result.survivors == (0,)
    assert result.answers == {(0, 1): 1}
    assert len(executor.outputs) == 2
    prefill, filter_step = executor.outputs
    assert prefill.total_num_scheduled_tokens == 4
    assert prefill.scheduled_new_reqs[0].sampling_params is None
    assert filter_step.total_num_scheduled_tokens == 2
    assert filter_step.scheduled_new_reqs[0].sampling_params is sampling
    assert filter_step.scheduled_new_reqs[0].num_computed_tokens == 4
    assert filter_step.finished_req_ids == {"prefill-0"}


def test_adapter_builds_shared_prefix_cascade_batch(monkeypatch):
    monkeypatch.setattr(vllm_runner, "_scheduler_types", fake_types)
    query = FilterQuery.from_sequences(
        body_token_ids=[[1, 2, 3, 4]],
        question_token_ids=[[5, 6], [7, 8]],
        yes_token_ids=[99],
    )
    kv = KVPageAllocator(
        total_pages=8,
        page_size_tokens=4,
        bytes_per_token=8,
    )
    executor = FakeExecutor()
    runner = VLLMModelRunner(
        query=query,
        kv=kv,
        model_executor=executor,
        sampling_params=object(),
    )
    runtime = DocEngineRuntime(
        query=query,
        runner=runner,
        packer=VariableLengthBatchPacker(BatchLimits(
            max_new_tokens=16,
            max_sequences=4,
            max_temporary_bytes=1024,
        )),
        kv=kv,
        speculation_k=2,
    )
    result = runtime.run()
    assert result.survivors == (0,)
    assert result.answers == {(0, 1): 1, (0, 2): 1}
    cascade = executor.outputs[1]
    assert cascade.num_common_prefix_blocks == [1]
    assert len(cascade.scheduled_new_reqs) == 2
    first, second = cascade.scheduled_new_reqs
    assert first.block_ids[0][0] == second.block_ids[0][0]
    assert first.block_ids[0][1] != second.block_ids[0][1]
