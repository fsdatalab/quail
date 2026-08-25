"""DocumentProvider and the catalog.

A provider is a named source of rows, where each row has an id and one
or more text columns. There is no text_col at registration: the
provider does not decide which column is "the document" - the query
does, through the column its PROMPT references.

Registration is cheap on purpose: it binds the name and reads the
schema (parquet metadata / dataset features), never the data.
Tokenization is not registration work; it happens at the first
DocScan of a column and is cached under a content hash.
"""

from dataclasses import dataclass, field

from quail.logical import CompileError


@dataclass(frozen=True)
class DocumentProvider:
    kind: str                 # "parquet" | "hf"
    source: str               # file path or dataset name
    id_col: str
    columns: tuple            # schema: column names, in schema order
    hf_split: str = "train"
    hf_config: str = ""

    @classmethod
    def from_parquet(cls, path: str, id_col: str) -> "DocumentProvider":
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)     # metadata only, no data scan
        cols = tuple(schema.names)
        if id_col not in cols:
            raise CompileError(
                f"id column {id_col!r} not in parquet schema {cols}")
        return cls(kind="parquet", source=path, id_col=id_col,
                   columns=cols)

    @classmethod
    def from_hf(cls, dataset: str, id_col: str, split: str = "train",
                config: str = "") -> "DocumentProvider":
        # `datasets` is a GPU-image dependency; lazy so CPU installs
        # without it still import this module.
        from datasets import load_dataset_builder
        builder = load_dataset_builder(dataset, config or None)
        cols = tuple(builder.info.features.keys())
        if id_col not in cols:
            raise CompileError(
                f"id column {id_col!r} not in dataset features {cols}")
        return cls(kind="hf", source=dataset, id_col=id_col,
                   columns=cols, hf_split=split, hf_config=config)

    def read_column(self, column: str) -> tuple[list, list]:
        """(ids, texts) for one column. The executor's DocScan calls
        this; registration never does."""
        if column not in self.columns:
            raise CompileError(
                f"column {column!r} not in schema {self.columns}")
        if self.kind == "parquet":
            import pyarrow.parquet as pq
            cols = [self.id_col] if column == self.id_col \
                else [self.id_col, column]
            table = pq.read_table(self.source, columns=cols)
            return (table.column(self.id_col).to_pylist(),
                    table.column(column).to_pylist())
        from datasets import load_dataset
        ds = load_dataset(self.source, self.hf_config or None,
                          split=self.hf_split)
        return list(ds[self.id_col]), list(ds[column])


@dataclass
class Catalog:
    providers: dict = field(default_factory=dict)

    def register(self, name: str, provider: DocumentProvider) -> None:
        self.providers[name] = provider

    def get(self, name: str) -> DocumentProvider:
        if name not in self.providers:
            raise CompileError(
                f"unknown provider {name!r}; registered: "
                f"{sorted(self.providers)}")
        return self.providers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.providers
