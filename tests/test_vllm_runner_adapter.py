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
    return Record, Record, EncoderStats, Record


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
