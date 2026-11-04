"""Read model and device specs from TOML files.

Each file in ``quail/specs/data/models`` and ``quail/specs/data/devices``
holds one spec: its fields as keys, named as the ``ModelSpec`` or
``DeviceSpec`` fields. A model file may name ``base``, another model,
and then lists only the fields that differ from it. Rust reads the same
files.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields, is_dataclass, replace
from functools import cache
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .base import DeviceSpec, ModelSpec

DATA = Path(__file__).parent / "data"


def _convert(annotation, value):
    """Return a TOML value as the type a spec field is annotated with."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        members = [member for member in get_args(annotation) if member is not NoneType]
        return _convert(members[0], value) if len(members) == 1 else value
    if origin is tuple:
        items = get_args(annotation)
        if len(items) == 2 and items[1] is Ellipsis:
            return tuple(_convert(items[0], item) for item in value)
        return tuple(_convert(kind, item) for kind, item in zip(items, value))
    if origin is Literal:
        return value
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(**{key: _convert(hints[key], item)
                             for key, item in value.items()})
    if annotation is float:
        return float(value)
    return value


def _build(kind: type, entries: dict[str, Any], base=None):
    hints = get_type_hints(kind)
    known = {field.name for field in fields(kind)}
    unknown = set(entries) - known
    if unknown:
        raise ValueError(f"{kind.__name__} has no fields {sorted(unknown)}")
    values = {key: _convert(hints[key], value) for key, value in entries.items()}
    if base is not None:
        return replace(base, **values)
    return kind(**values)


def _read(directory: Path) -> dict[str, dict]:
    tables = {}
    for path in sorted(directory.glob("*.toml")):
        with path.open("rb") as file:
            table = tomllib.load(file)
        if table.get("name") != path.stem:
            raise ValueError(f"{path} names spec {table.get('name')!r}")
        tables[path.stem] = table
    return tables


@cache
def read_models(directory: Path = DATA / "models") -> dict[str, ModelSpec]:
    """Return every model spec in a directory, by name.

    Raises:
        ValueError: A file's name differs from its spec's name, a key
            is not a ModelSpec field, or a ``base`` is missing or forms
            a cycle.
    """
    tables = _read(directory)
    models: dict[str, ModelSpec] = {}

    def build(name: str, seen: tuple[str, ...] = ()) -> ModelSpec:
        if name in models:
            return models[name]
        if name in seen:
            raise ValueError(f"model bases form a cycle: {' -> '.join(seen)}")
        if name not in tables:
            raise ValueError(f"no model spec named {name!r}")
        entries = dict(tables[name])
        base_name = entries.pop("base", None)
        base = build(base_name, (*seen, name)) if base_name else None
        models[name] = _build(ModelSpec, entries, base)
        return models[name]

    for name in tables:
        build(name)
    return {name: models[name] for name in tables}


@cache
def read_devices(directory: Path = DATA / "devices") -> dict[str, DeviceSpec]:
    """Return every device spec in a directory, by name.

    Raises:
        ValueError: A file's name differs from its spec's name, or a
            key is not a DeviceSpec field.
    """
    return {name: _build(DeviceSpec, table)
            for name, table in _read(directory).items()}
