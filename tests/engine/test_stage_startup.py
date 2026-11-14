"""CPU tests of how a region starts on its GPU workers and what a failure leaves."""

import json
import sys
import threading

import _quail_native
import pyarrow as pa
import pytest

from quail.physical import PhysicalGraph, PortRef, Scan

RAN = json.dumps({"metrics": {}})
# one stage that reads the scan whole, as a region on one GPU has
STAGES = json.dumps([{
    "key": "", "nodes": ["node"], "outputs": [],
    "inputs": [{"node_id": "scan:r", "port": "rows:r", "stage": None,
                "hashed": False}]}])

# a worker process that says where it read the region's table from
WORKER_TEXT = """
import json, os, sys
import _quail_native

end = _quail_native.WorkerEnd(sys.argv[1])
found = {"gpu": os.environ["CUDA_VISIBLE_DEVICES"], "index": sys.argv[2]}
while (message := end.next()) is not None:
    if message[0] == "region":
        table = message[2]["documents:r"]
        address = table.column("id").chunk(0).buffers()[1].address
        with open("/proc/self/maps") as maps:
            mapped = [line.split()[0].split("-") for line in maps
                      if "/region" in line]
        found["in_file"] = any(
            int(start, 16) <= address < int(end_, 16) for start, end_ in mapped)
        found["total"] = sum(table.column("id").to_pylist())
        end.ready()
    elif message[0] == "end":
        end.ran(json.dumps(found), [])
"""


def graph():
    return PhysicalGraph(
        (Scan(node_id="scan:r", alias="r", input_id="r", num_documents=1),),
        PortRef("scan:r", "rows:r"),
    )


def _worker(path, seen, failing):
    """Answer Quail as a worker does; fail the stages of the regions in failing."""
    end = _quail_native.WorkerEnd(path)
    regions = 0
    while (message := end.next()) is not None:
        seen.append(message[0])
        if message[0] == "region":
            regions += 1
            end.ready()
        elif message[0] == "stage" and regions in failing:
            end.fail("the stage went wrong")
        elif message[0] == "stage":
            end.outputs([])
        elif message[0] == "end":
            end.ran(RAN, [])


@pytest.fixture
def workers():
    """One worker on a thread: (the workers, what it saw, the regions it fails)."""
    found = _quail_native.Workers(1)
    seen, failing = [], set()
    thread = threading.Thread(
        target=_worker, args=(str(found.paths[0]), seen, failing), daemon=True)
    thread.start()
    found.connect()
    yield found, seen, failing
    found.close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def _run(workers, stages=STAGES):
    return workers.run_region("{}", [], graph().to_json(), stages,
                              lambda table: None, lambda question: "null")


def test_a_region_that_does_not_decode_reaches_no_worker(workers):
    found, seen, _ = workers
    for stages in ["null", '"not stages"', "{}", '[{"key": "r"}]']:
        with pytest.raises(ValueError):
            _run(found, stages)
    assert seen == []


def test_a_failed_stage_is_the_regions_error_and_the_workers_run_the_next(workers):
    found, seen, failing = workers
    failing.add(1)
    with pytest.raises(RuntimeError, match="the stage went wrong"):
        _run(found)
    # the worker was told the region ended, and serves the next one
    assert seen == ["region", "stage", "end"]
    assert [json.loads(ran) for ran, _ in _run(found)] == [json.loads(RAN)]
    assert seen == ["region", "stage", "end"] * 2


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc/self/maps")
def test_a_worker_process_reads_a_regions_table_over_the_files_mapped_pages(
        tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER_TEXT)
    found = _quail_native.Workers(2)
    try:
        found.spawn([sys.executable, str(script)])
        found.connect()
        assert found.alive()
        table = pa.table({"id": pa.array(range(1000), pa.int32())})
        ran = found.run_region(
            "{}", [("documents:r", table)], graph().to_json(), "[]",
            lambda table: None, lambda question: "null")
    finally:
        found.close()
    # each worker sees its own GPU and maps the one file the table is in
    assert [json.loads(text) for text, _ in ran] == [
        {"gpu": str(gpu), "index": str(gpu), "in_file": True, "total": 499500}
        for gpu in range(2)]
    assert not found.alive()
