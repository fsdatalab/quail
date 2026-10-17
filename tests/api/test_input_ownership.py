"""Failed input preparation releases its files before a retry."""

from pathlib import Path

import pyarrow as pa
import pytest
from helpers.sessions import CONFIG

import quail
from quail.catalog import tokens


def _track_files(monkeypatch):
    handles = []
    for name in ("OSFile", "memory_map"):
        original = getattr(pa, name)

        def opened(*args, _original=original, **kwargs):
            handle = _original(*args, **kwargs)
            handles.append(handle)
            return handle

        monkeypatch.setattr(pa, name, opened)
    return handles


def _fail_second(monkeypatch, owner, name, error):
    original = getattr(owner, name)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise error("injected input failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, fail)


@pytest.mark.parametrize("state", ["cold", "warm", "offsets"])
@pytest.mark.parametrize("phase", [
    "writer_open", "write", "cancel", "finish", "read_open",
])
def test_input_failure_rolls_back_only_its_own_files(monkeypatch, phase, state):
    table = pa.table({"id": [1, 2], "body": ["hello", "world"], "rating": [3, 4]})

    def tokenizer(text):
        return list(text.encode())

    tokenizer.token_bytes = lambda: {i: bytes([i]) for i in range(256)}
    with quail.Session(CONFIG, tokenizer=tokenizer) as session:
        session.register("docs", quail.DocumentProvider.from_table(table, id_col="id"))
        previous = (session.inputs.tokenize("docs", "body", ("id",))
                    if state != "cold" else None)
        old_tokens = dict(session.inputs._token_stores)
        old_columns = dict(session.inputs._column_stores)
        old_paths = {Path(store.path) for store in (*old_tokens.values(),
                                                   *old_columns.values())}
        error = KeyboardInterrupt if phase == "cancel" else OSError
        with monkeypatch.context() as patch:
            handles = _track_files(patch)
            owner, name = {
                "writer_open": (pa.ipc, "new_file"),
                "write": (tokens.ColumnStoreWriter, "write_batch"),
                "cancel": (tokens.ColumnStoreWriter, "write_batch"),
                "finish": (tokens.ColumnStoreWriter, "finish"),
                "read_open": (pa.ipc, "open_file"),
            }[phase]
            _fail_second(patch, owner, name, error)
            with pytest.raises(error, match="injected input failure") as caught:
                session.inputs.tokenize("docs", "body", ("id", "body", "rating"),
                                        with_offsets=state == "offsets")
        assert caught.value.__traceback__ is not None
        assert handles and all(handle.closed for handle in handles)
        assert session.inputs._token_stores == old_tokens
        assert session.inputs._column_stores == old_columns
        directory = Path(session.inputs._token_directory.name)
        assert set(directory.iterdir()) == old_paths
        assert not session.inputs._replaced_token_stores
        retry = session.inputs.tokenize("docs", "body", ("id", "body", "rating"),
                                        with_offsets=state == "offsets")
        assert retry.column("rating").to_pylist() == [3, 4]
        assert [list(doc) for doc in retry.tokens] == [list(b"hello"), list(b"world")]
        if state == "offsets":
            assert retry.tokens.has_offsets and not previous.tokens.has_offsets
            assert session.inputs._replaced_token_stores == [previous.tokens]
            assert list(previous.tokens[0]) == list(b"hello")
        elif previous is not None:
            assert retry.tokens is previous.tokens


@pytest.mark.parametrize("kind", ["tokens", "column"])
@pytest.mark.parametrize("phase", ["cancel", "read_open"])
def test_store_write_owns_files_through_finalization(
    tmp_path, monkeypatch, kind, phase,
):
    path = tmp_path / "store.arrow"
    table = pa.table({"body": ["hello", "world"]})
    error = KeyboardInterrupt if phase == "cancel" else OSError

    def batches():
        yield from table.to_batches()
        if phase == "cancel":
            raise error("injected input failure")

    def fail(*_args, **_kwargs):
        raise error("injected input failure")

    with monkeypatch.context() as patch:
        handles = _track_files(patch)
        if phase == "read_open":
            patch.setattr(pa.ipc, "open_file", fail)
        with pytest.raises(error, match="injected input failure") as caught:
            if kind == "tokens":
                tokens.TokenStore.write(str(path), batches(), document_column="body",
                                        tokenizer=lambda text: list(text.encode()),
                                        token_type=pa.int32())
            else:
                tokens.ColumnStore.write(str(path), batches(), table.schema.field(0))
    assert caught.value.__traceback__ is not None
    assert handles and all(handle.closed for handle in handles)
    assert not path.exists()
