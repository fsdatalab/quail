"""Immutable experiment artifact writing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid
from typing import Any, Mapping


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    phase: str
    created_at_utc: str
    git_commit: str
    command: tuple[str, ...]
    python_version: str
    dependency_versions: Mapping[str, str | None]
    config: Mapping[str, Any]
    seeds: Mapping[str, int]
    model_revision: str
    dataset_revision: str
    gpu: Mapping[str, Any]
    cache_reset_confirmed: bool | None

    @classmethod
    def create(
        cls,
        *,
        phase: str,
        config: Mapping[str, Any],
        seeds: Mapping[str, int],
        model_revision: str,
        dataset_revision: str,
        gpu: Mapping[str, Any] | None = None,
        cache_reset_confirmed: bool | None = None,
        command: tuple[str, ...] | None = None,
        repo: str | Path = ".",
    ) -> "RunMetadata":
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        safe_phase = "".join(
            char if char.isalnum() or char in "-_" else "-"
            for char in phase
        ).strip("-")
        run_id = f"{stamp}-{safe_phase}-{uuid.uuid4().hex[:8]}"
        return cls(
            run_id=run_id,
            phase=phase,
            created_at_utc=now.isoformat(),
            git_commit=_git_commit(Path(repo).resolve()),
            command=tuple(command or ()),
            python_version=platform.python_version(),
            dependency_versions={
                "modal": _package_version("modal"),
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
                "vllm": _package_version("vllm"),
                "flashinfer-python": _package_version("flashinfer-python"),
            },
            config=dict(config),
            seeds={key: int(value) for key, value in seeds.items()},
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            gpu=dict(gpu or {}),
            cache_reset_confirmed=cache_reset_confirmed,
        )


def _write_json_atomic(path: Path, value: Any, *, compress: bool) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    opener = gzip.open if compress else open
    with opener(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
    os.replace(temporary, path)


def write_run_artifact(
    root: str | Path,
    metadata: RunMetadata,
    result: Mapping[str, Any],
    *,
    compress_result: bool = True,
) -> Path:
    directory = Path(root) / metadata.run_id
    directory.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(
        directory / "metadata.json",
        asdict(metadata),
        compress=False,
    )
    result_name = "result.json.gz" if compress_result else "result.json"
    _write_json_atomic(
        directory / result_name,
        dict(result),
        compress=compress_result,
    )
    digest = hashlib.sha256(
        (directory / result_name).read_bytes()
    ).hexdigest()
    _write_json_atomic(
        directory / "checksums.json",
        {result_name: digest},
        compress=False,
    )
    return directory
