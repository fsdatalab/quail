"""A Session on the Apple GPU: the device spec, support checks, and queries.

A tiny Decision 2.0 package with random weights runs filters, a join, a
classification, and scores through `Session` on `device="apple-gpu"`.
Its scores are compared with mlx-lm running each request's whole prompt.
The session runs bf16, so the yardstick is mlx-lm's own bf16 error
against float32.

Runs on Apple silicon with mlx, vllm-metal, and mlx-lm installed;
skipped elsewhere.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("vllm_metal")
pytest.importorskip("mlx_lm")

from test_mlx_device import (  # noqa: E402
    CONFIG,
    head_weights,
    reference_model,
    reference_scores,
    write_package,
)

import quail  # noqa: E402
from quail.backends.quail import backend as backend_module  # noqa: E402
from quail.backends.quail import worker  # noqa: E402
from quail.backends.quail.executor.mlx_device import (  # noqa: E402
    implementation as implementation_module,
)
from quail.backends.quail.executor.mlx_device.implementation import (  # noqa: E402
    MlxImplementation,
)
from quail.backends.quail.executor.mlx_device.kernels import (  # noqa: E402
    INSTALL_TEXT,
)
from quail.cost import budgets  # noqa: E402
from quail.execution import execute  # noqa: E402
from quail.execution.session import RefusalError  # noqa: E402
from quail.logical import bind_prompt, render_filter_prompt_ids  # noqa: E402
from quail.planner.plan import EngineConfig  # noqa: E402
from quail.specs import (  # noqa: E402
    APPLE_GPU,
    DECISION_2_KAI_0_6B_BF16,
    H100_SXM,
    QWEN3_4B_FP8,
    local_apple_gpu,
)

DOCUMENTS = [
    "Please refund this order, it arrived broken.",
    "Thanks, the package came early and works well.",
    "I want my money back for the damaged lamp.",
    "Where is my parcel? It has been two weeks.",
    "Great service, I will order again.",
    "The charger stopped working after a day, refund it.",
]
QUERIES = ["refund request", "shipping delay"]
REFUND = "Does the customer ask for a refund? {0}"
LATE = "Is the customer waiting for a delivery? {0}"


def logit(probability):
    return np.log(probability) - np.log1p(-probability)


def current(setter):
    """An MLX limit, read by setting it and setting it back."""
    value = setter(0)
    setter(value)
    return value

def test_the_apple_gpu_spec_is_built_from_this_machine():
    spec = local_apple_gpu()
    recommended = mx.device_info()["max_recommended_working_set_size"]
    assert spec.name == APPLE_GPU == "apple-gpu"
    assert spec.implementation == "mlx" and H100_SXM.implementation == "cuda"
    assert spec.mem_bytes == recommended / 2
    assert local_apple_gpu(0.25).mem_bytes == recommended / 4
    assert local_apple_gpu(1).mem_bytes == recommended
    for fraction in (0, -0.5, 1.5):
        with pytest.raises(ValueError, match="memory_fraction"):
            local_apple_gpu(fraction)
    assert spec.usd_per_hour == 0.0
    # the device caps the chunk below the model's own cap
    assert budgets.chunk_budget(DECISION_2_KAI_0_6B_BF16, spec) == 8192
    assert budgets.chunk_budget(DECISION_2_KAI_0_6B_BF16, H100_SXM) == 65_536
    # the weights, two chunks of activations, and some KV fit the budget
    assert budgets.arena_tokens(DECISION_2_KAI_0_6B_BF16, spec) > 8192


def test_quail_runs_only_the_decision_model_on_one_apple_gpu():
    registry = quail.ExtensionRegistry.with_built_ins()
    backend = registry.backend("quail")
    spec = local_apple_gpu()
    assert backend.supports(DECISION_2_KAI_0_6B_BF16, spec, 1).supported
    other = backend.supports(QWEN3_4B_FP8, spec, 1)
    assert not other.supported
    assert other.reason == ("Quail on apple-gpu runs decision-2.0-kai-0.6b-bf16, "
                            "not 'qwen3-4b-fp8'")
    two = backend.supports(DECISION_2_KAI_0_6B_BF16, spec, 2)
    assert two.reason == "Quail on apple-gpu runs one model copy, not 2"
    # the CUDA checks are as they were
    assert backend.supports(QWEN3_4B_FP8, H100_SXM, 2).supported
    assert not backend.supports(QWEN3_4B_FP8, H100_SXM, 3).supported
    assert not registry.backend("stock_vllm").supports(
        DECISION_2_KAI_0_6B_BF16, spec, 1).supported

    with pytest.raises(RefusalError, match="not 'qwen3-4b-fp8'"):
        quail.Session(EngineConfig(model="qwen3-4b-fp8", device="apple-gpu"),
                      tokenizer=str.split)
    with pytest.raises(RefusalError, match="one model copy"):
        quail.Session(EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name,
                                   device="apple-gpu", gpus=2),
                      tokenizer=str.split)
    # a session registers the device it built
    session = quail.Session(
        EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name, device="apple-gpu"),
        tokenizer=str.split, registry=registry)
    assert registry.device("apple-gpu") is session.device
    session.close()

    # a session may take another share of the memory, with its own registry
    quarter = quail.Session(
        EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name, device="apple-gpu",
                     memory_fraction=0.25), tokenizer=str.split)
    assert quarter.device.mem_bytes == session.device.mem_bytes / 2
    quarter.close()
    with pytest.raises(ValueError, match="another memory fraction"):
        quail.Session(
            EngineConfig(model=DECISION_2_KAI_0_6B_BF16.name, device="apple-gpu",
                         memory_fraction=0.25),
            tokenizer=str.split, registry=registry)
    with pytest.raises(ValueError, match="applies to apple-gpu, not h100-sxm"):
        quail.Session(EngineConfig(model="qwen3-4b-fp8", device="h100-sxm",
                                   memory_fraction=0.5), tokenizer=str.split)


def test_the_device_check_asks_the_device_implementation(monkeypatch):
    spec = local_apple_gpu()
    assert execute.gpu_problem(spec) is None
    assert MlxImplementation.problem() is None

    def missing():
        raise RuntimeError(INSTALL_TEXT)

    monkeypatch.setattr(implementation_module, "metal_ops", missing)
    assert execute.gpu_problem(spec) == INSTALL_TEXT
    assert "pip install" in INSTALL_TEXT and "vllm_metal-0.30.0" in INSTALL_TEXT


def test_chunk_events_wait_for_the_device_only_when_timed():
    implementation = MlxImplementation()
    waits = []
    implementation.mx = type("FakeMx", (), {
        "synchronize": staticmethod(lambda: waits.append(1))})
    implementation.record_event()
    assert not waits
    implementation.time_chunks(True)
    start = implementation.record_event()
    assert waits == [1]
    assert implementation.elapsed_ms(start, implementation.record_event()) >= 0
    implementation.time_chunks(False)
    implementation.record_event()
    assert len(waits) == 2


def test_the_implementation_holds_memory_in_a_budget_and_gives_the_limits_back():
    wired, kept = current(mx.set_wired_limit), current(mx.set_cache_limit)
    implementation = MlxImplementation()
    implementation.release_limits()        # nothing held yet
    live = mx.zeros((1 << 20,), dtype=mx.float32)
    mx.eval(live)
    in_use = mx.get_active_memory()
    assert in_use >= live.nbytes
    implementation.hold_within(in_use + 1_000_000_000)
    assert current(mx.set_wired_limit) == in_use + 1_000_000_000
    # a few bytes of MLX's own arrays come and go between the two reads
    assert abs(current(mx.set_cache_limit) - 1_000_000_000) < 1_000_000
    # a smaller budget leaves less for the buffers MLX keeps
    implementation.hold_within(in_use + 250_000_000)
    assert abs(current(mx.set_cache_limit) - 250_000_000) < 1_000_000
    implementation.hold_within(in_use // 2)
    assert current(mx.set_cache_limit) == 0
    implementation.release_limits()
    assert current(mx.set_wired_limit) == wired
    assert current(mx.set_cache_limit) == kept


# ---- queries through a Session on a tiny decision model

TINY = dict(CONFIG, vocab_size=320)


def write_tokenizer(directory):
    """A byte-level tokenizer: one token per byte, as test_decision_model's."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    alphabet = [chr(byte) for byte in visible] + [
        chr(256 + index) for index in range(len(missing))]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False,
                  "trim_offsets": True, "use_regex": True}
    (directory / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "added_tokens": [], "normalizer": None,
        "pre_tokenizer": byte_level, "post_processor": None,
        "decoder": byte_level, "model": {"type": "BPE", "merges": [],
        "vocab": {char: index for index, char in enumerate(alphabet)}}}))


@pytest.fixture()
def tiny(tmp_path, monkeypatch):
    """A tiny decision package on disk, its spec, and its reference model."""
    package = tmp_path / "package"
    package.mkdir()
    model = reference_model(TINY)
    head = head_weights(np.random.default_rng(8))
    write_package(package, model, {k: mx.array(v) for k, v in head.items()},
                  config=TINY)
    write_tokenizer(package)
    spec = replace(
        DECISION_2_KAI_0_6B_BF16, name="tiny-decision", hf_name=str(package),
        revision="", layers=2, hidden=64, n_q=4, n_kv=2, d_head=64,
        ffn_width=256, params=2 * 64 * 64 * 12, vocab=320, w_mem_bytes=0.0)
    monkeypatch.setattr(backend_module, "MLX_MODELS", frozenset({spec.name}))
    # the tiny model takes the place of any model an earlier test loaded
    monkeypatch.setattr(execute, "_BACKEND_STATE", {})
    for name, column, values in (("documents", "body", DOCUMENTS),
                                 ("queries", "text", QUERIES)):
        pq.write_table(pa.table({"id": list(range(len(values))), column: values}),
                       tmp_path / f"{name}.parquet")

    def session(**config):
        # a registry holds one apple-gpu spec, and sessions here differ
        # in memory fraction
        registry = quail.ExtensionRegistry.with_built_ins()
        registry.register_model(spec)
        value = quail.Session(
            EngineConfig(model=spec.name, device="apple-gpu", **config),
            registry=registry)
        for name in ("documents", "queries"):
            value.register(name, quail.DocumentProvider.from_parquet(
                str(tmp_path / f"{name}.parquet"), id_col="id"))
        return value

    half = reference_model(TINY)
    half.set_dtype(mx.bfloat16)

    def margins(template, session):
        """mlx-lm's Yes minus No score of each document's whole prompt.

        Returns:
            The margins in float32, and how far mlx-lm in bf16 is from them.
        """
        ids = session.tokenizer
        prompt = bind_prompt(template, ("body",), ids, turn=spec.turn,
                             layout=spec.prompt_layout)
        offsets = worker.decision_offsets(spec, str(package))
        out = {exact: [] for exact in (True, False)}
        for document in DOCUMENTS:
            tokens = render_filter_prompt_ids(prompt, ids(document), ids)
            for exact in out:
                hidden = np.array((model if exact else half).model(
                    mx.array([tokens]))[0].astype(mx.float32))
                rows = hidden[[len(tokens) - 1 - offset for offset in offsets]]
                no, yes = reference_scores(
                    head, rows[None, :-1], rows[None, -1])[0]
                out[exact].append(yes - no)
        exact, rounded = np.array(out[True]), np.array(out[False])
        return exact, float(np.abs(rounded - exact).max())

    return session, margins


def test_a_session_runs_scores_filters_a_join_and_a_classification(tiny):
    open_session, margins = tiny
    with open_session() as session:
        assert session.device.implementation == "mlx"
        refund, refund_rounding = margins(REFUND, session)
        late, late_rounding = margins(LATE, session)

        scored = session.sql(
            f"SELECT d.id, AI.SCORE(PROMPT('{REFUND}', d.body)) AS s "
            "FROM documents d").run()
        got = dict(scored.to_rows())
        assert sorted(got) == list(range(len(DOCUMENTS)))
        ours = np.abs(logit(np.array([got[d] for d in range(len(DOCUMENTS))]))
                      - refund).max()
        assert refund_rounding > 0
        assert ours < 2 * refund_rounding + 0.05, (ours, refund_rounding)
        assert scored.report["boot_kind"] == "cold"
        assert scored.report["boot"]["warm_tier"] == "touch"
        assert scored.report["fresh_tokens"] > 0
        assert scored.report["peak_gib"] > 0

        # the second filter reads each document's KV the first one kept
        chain = session.sql(
            "SELECT d.id FROM documents d "
            f"WHERE AI_FILTER(PROMPT('{REFUND}', d.body)) "
            f"AND AI_FILTER(PROMPT('{LATE}', d.body))").run()
        survivors = {row[0] for row in chain.to_rows()}
        # a margin within the rounding error of zero may go either way
        decided = [d for d in range(len(DOCUMENTS))
                   if abs(refund[d]) > 3 * refund_rounding
                   and abs(late[d]) > 3 * late_rounding]
        assert len(decided) >= len(DOCUMENTS) // 2
        for d in decided:
            assert (d in survivors) == (refund[d] > 0 and late[d] > 0), d
        assert chain.report["boot_kind"] == "warm"

        joined = session.sql(
            "SELECT q.id, d.id FROM queries q JOIN documents d "
            "ON AI_FILTER(PROMPT('Does {1} match the topic {0}?', "
            "q.text, d.body))").run()
        pairs = joined.to_rows()
        assert len(set(pairs)) == len(pairs)
        assert all(q in range(len(QUERIES)) and d in range(len(DOCUMENTS))
                   for q, d in pairs)
        assert joined.report["fresh_tokens"] > 0

        labeled = session.sql(
            "SELECT d.id, AI.CLASSIFY(d.body, ARRAY['refund', 'delivery', "
            "'praise']) AS topic FROM documents d").run()
        labels = dict(labeled.to_rows())
        assert sorted(labels) == list(range(len(DOCUMENTS)))
        assert set(labels.values()) <= {"refund", "delivery", "praise"}

    # another session finds the model loaded, and exact chunk times
    # add up to at most the query time
    with open_session(gpu_timing=True) as session:
        timed = session.sql(
            "SELECT d.id FROM documents d "
            f"WHERE AI_FILTER(PROMPT('{REFUND}', d.body)) "
            f"AND AI_FILTER(PROMPT('{LATE}', d.body))").run()
        assert timed.report["boot_kind"] == "warm"
        assert timed.report["chunks"] >= 1
        assert 0 < timed.report["gpu_s"] <= timed.report["wall_s"] + 0.01
        assert {row[0] for row in timed.to_rows()} == survivors


def test_a_later_session_gives_a_loaded_model_its_own_memory_budget(tiny):
    open_session, _ = tiny
    sql = (f"SELECT d.id FROM documents d "
           f"WHERE AI_FILTER(PROMPT('{REFUND}', d.body))")
    rows, boots = [], []
    for fraction in (0.04, 0.02):
        with open_session(memory_fraction=fraction) as session:
            result = session.sql(sql).run()
            rows.append(sorted(result.to_rows()))
            boots.append(result.report["boot_kind"])
            budget = int(session.device.mem_bytes)
            mx.synchronize()
            held = mx.get_active_memory() + mx.get_cache_memory()
            assert current(mx.set_wired_limit) == budget
            assert current(mx.set_cache_limit) < budget
            assert 0.5 * budget < held <= 1.01 * budget, (fraction, held, budget)
    assert boots == ["cold", "warm"]
    assert rows[0] == rows[1]


def test_a_loaded_mlx_model_is_released_without_torch(tiny, monkeypatch):
    open_session, _ = tiny
    with open_session() as session:
        session.sql("SELECT d.id FROM documents d "
                    f"WHERE AI_FILTER(PROMPT('{REFUND}', d.body))").run()
    (loaded,) = execute._BACKEND_STATE.values()
    assert isinstance(loaded, worker.LoadedMlx) and loaded.model is not None
    # a process that runs MLX models has no torch
    monkeypatch.setitem(sys.modules, "torch", None)
    released = worker.release_booted_models(execute._BACKEND_STATE)
    assert released["models_released"] == 1
    assert released["cuda_allocated_bytes"] == 0
    assert loaded.implementation._limits_before is None
    assert execute._BACKEND_STATE == {}
    assert loaded.model is None and loaded.arena is None
