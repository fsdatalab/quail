"""Turn table providers into inputs a remote service can read.

A remote query must not depend on the client's memory, files, or
provider objects. A supported provider becomes either an uploaded
snapshot (one Arrow IPC file named by the hash of its bytes) or an
immutable reference (a Hugging Face dataset pinned to a commit).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from quail.catalog import (
    ArrowDatasetProvider,
    DocumentProvider,
    HuggingFaceProvider,
    MemoryTableProvider,
    ScanRequest,
    TableProvider,
)
from quail.service.artifacts import write_ipc_file
from quail.service.records import InvalidRequestError

SUPPORTED = (
    "DocumentProvider.from_table, from_parquet, from_ipc, from_dataset, "
    "or from_hf"
)


@dataclass(frozen=True)
class PreparedInput:
    """A provider described for the service, with its upload if any."""

    spec: dict
    upload_path: Path | None = None

    @property
    def content_id(self) -> str | None:
        return self.spec.get("content_id")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hf_head_revision(dataset_name: str) -> str:
    """Return the current commit hash of a Hugging Face dataset."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_name).sha


def describe(provider: TableProvider, workdir: str | Path, *,
             resolve_revision=hf_head_revision) -> PreparedInput:
    """Describe one provider for submission.

    Snapshots are written to ``workdir`` as Arrow IPC files. Raises
    TypeError for a provider that cannot be sent to a service.

    Args:
        provider: The registered provider.
        workdir: Where snapshot files are written before upload.
        resolve_revision: Called with a dataset name when a Hugging Face
            provider has no revision; returns the commit hash to pin.
    """
    workdir = Path(workdir)
    if isinstance(provider, HuggingFaceProvider):
        revision = provider.revision or resolve_revision(provider.dataset_name)
        return PreparedInput({
            "kind": "hf",
            "dataset": provider.dataset_name,
            "config": provider.config,
            "split": provider.split,
            "revision": revision,
            "id_col": provider.id_col,
        })
    if isinstance(provider, MemoryTableProvider):
        source = provider.table
    elif isinstance(provider, ArrowDatasetProvider):
        source = provider.scan(ScanRequest(columns=provider.columns))
    else:
        raise TypeError(
            f"{type(provider).__name__} cannot be sent to a Quail service; "
            f"register {SUPPORTED}")
    workdir.mkdir(parents=True, exist_ok=True)
    staging = workdir / f"snapshot-{id(provider)}.arrow"
    try:
        write_ipc_file(staging, source)
    finally:
        if not isinstance(source, pa.Table):
            source.close()
    content_id = file_digest(staging)
    path = workdir / f"{content_id}.arrow"
    staging.replace(path)
    return PreparedInput({
        "kind": "snapshot",
        "content_id": content_id,
        "id_col": provider.id_col,
        "columns": list(provider.columns),
    }, path)


def resolve(spec: dict, snapshot_paths: dict[str, str | Path]) -> TableProvider:
    """Build the provider a service reads for one described input.

    Args:
        spec: The input description saved with the record.
        snapshot_paths: Uploaded snapshot files by content id.
    """
    kind = spec.get("kind")
    if kind == "hf":
        if not spec.get("revision"):
            raise InvalidRequestError("a Hugging Face input needs a revision")
        return DocumentProvider.from_hf(
            spec["dataset"], id_col=spec["id_col"], split=spec["split"],
            config=spec.get("config", ""), revision=spec["revision"])
    if kind == "snapshot":
        path = snapshot_paths.get(spec.get("content_id"))
        if path is None:
            raise InvalidRequestError(
                f"input snapshot {spec.get('content_id')!r} was not uploaded")
        return DocumentProvider.from_ipc(str(path), id_col=spec["id_col"])
    raise InvalidRequestError(f"unknown input kind {kind!r}")
