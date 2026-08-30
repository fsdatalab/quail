"""DocumentProvider and the catalog.

A provider is a named source of rows with an id and one or more text
columns. Registration reads the schema only, never the data.
"""

from dataclasses import dataclass, field

from quail.logical import CompileError


@dataclass(frozen=True)
class DocumentProvider:
    kind: str                 # "dataset" | "hf"
    source: object            # Arrow Dataset or Hugging Face dataset name
    id_col: str
    columns: tuple            # schema: column names, in schema order
    hf_split: str = "train"
    hf_config: str = ""

    @classmethod
    def from_dataset(cls, dataset, id_col: str) -> "DocumentProvider":
        """Create a provider for an Arrow Dataset."""
        import pyarrow.dataset as ds

        if not isinstance(dataset, ds.Dataset):
            raise TypeError(
                "dataset must be a pyarrow.dataset.Dataset, got "
                f"{type(dataset).__name__}")
        cols = tuple(dataset.schema.names)
        if id_col not in cols:
            raise CompileError(
                f"id column {id_col!r} not in dataset schema {cols}")
        return cls(kind="dataset", source=dataset, id_col=id_col,
                   columns=cols)

    @classmethod
    def from_parquet(cls, path: str, id_col: str) -> "DocumentProvider":
        """Create a provider for a Parquet file or directory."""
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        return cls.from_dataset(dataset, id_col=id_col)

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

    def _dataset(self):
        """Return the provider source as an Arrow Dataset."""
        if self.kind == "dataset":
            return self.source
        if self.kind != "hf":
            raise ValueError(f"unknown document provider kind {self.kind!r}")

        import pyarrow.dataset as ds
        from datasets import load_dataset

        hf_dataset = load_dataset(
            self.source, self.hf_config or None, split=self.hf_split)
        return ds.dataset(hf_dataset.data.table)

    def scan_batches(self, columns, batch_rows: int = 1_024):
        """Yield bounded Arrow batches for the requested columns.

        Args:
            columns: Document columns to read alongside the ID column.
            batch_rows: Maximum rows requested from the Arrow scanner.

        Yields:
            Arrow record batches with the ID column first.
        """
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        requested = [self.id_col]
        for column in columns:
            if column not in self.columns:
                raise CompileError(
                    f"column {column!r} not in schema {self.columns}")
            if column not in requested:
                requested.append(column)
        scanner = self._dataset().scanner(
            columns=requested, batch_size=batch_rows)
        yield from scanner.to_batches()

    def arrow_schema(self):
        """Return the Arrow schema for the provider source."""
        return self._dataset().schema

    def read_column(self, column: str):
        """Return an Arrow table containing the ID and requested column."""
        if column not in self.columns:
            raise CompileError(
                f"column {column!r} not in schema {self.columns}")
        return self._dataset().to_table(
            columns=[self.id_col] if column == self.id_col
            else [self.id_col, column])


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
