"""Input caches follow the registered dataset's rows and their order."""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from helpers.sessions import CONFIG

import quail


@pytest.mark.parametrize("kind", ["memory", "file_order", "filtered", "union"])
def test_datasets_with_the_same_schema_keep_distinct_cached_inputs(tmp_path, kind):
    first = pa.table({"id": [1, 2], "body": ["a", "long"]})
    second = pa.table({"id": [3, 4], "body": ["other", "text"]})
    if kind in ("file_order", "filtered"):
        paths = [str(tmp_path / f"part{i}.parquet") for i in range(2)]
        for table, path in zip((first, second), paths):
            pq.write_table(table, path)
        left = ds.dataset(paths, format="parquet")
        if kind == "file_order":
            right = ds.dataset(paths[::-1], format="parquet")
            expected = [pa.concat_tables([first, second]),
                        pa.concat_tables([second, first])]
        else:
            right = left.filter(ds.field("id") > 2)
            left = left.filter(ds.field("id") <= 2)
            expected = [first, second]
    elif kind == "union":
        left = ds.dataset([ds.dataset(first), ds.dataset(second)])
        right = ds.dataset([ds.dataset(second), ds.dataset(first)])
        expected = [pa.concat_tables([first, second]),
                    pa.concat_tables([second, first])]
    else:
        left, right = ds.dataset(first), ds.dataset(second)
        expected = [first, second]
    with quail.Session(CONFIG, tokenizer=lambda text: list(text.encode())) as session:
        for name, dataset in (("left", left), ("right", right), ("again", left)):
            session.register(name, quail.DocumentProvider.from_dataset(dataset, "id"))
        stores = []
        for name, table in zip(("left", "right"), expected):
            assert session.inputs.estimate_lengths(name, "body") == [
                len(text.encode()) for text in table["body"].to_pylist()]
            store = session.inputs.tokenize(name, "body", ("id", "body"))
            stores.append(store)
            assert store.column("id").to_pylist() == table["id"].to_pylist()
            assert [list(tokens) for tokens in store.tokens] == [
                list(text.encode()) for text in table["body"].to_pylist()]
        again = session.inputs.tokenize("again", "body", ("id", "body"))
        assert again.tokens is stores[0].tokens
