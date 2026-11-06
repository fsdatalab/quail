"""Read model and device specs from TOML files, with Rust's reader.

Each file in ``quail/specs/data/models`` and ``quail/specs/data/devices``
holds one spec: its fields as keys, named as the ``ModelSpec`` or
``DeviceSpec`` fields. A model file may name ``base``, another model,
and then lists only the fields that differ from it. Rust reads the same
files.
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from functools import cache
from pathlib import Path
from types import NoneType, UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

import _quail_native

from .base import DeviceSpec, ModelSpec

DATA = Path(__file__).parent / "data"


def _convert(annotation, value):
    """Return a spec value as the type its field is annotated with."""
    if value is None:
        return None
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


def _read(kind: type, name: str, directory: Path) -> dict:
    hints = get_type_hints(kind)
    found = json.loads(_quail_native.read_specs(name, str(directory)))
    return {spec: kind(**{key: _convert(hints[key], value)
                          for key, value in values.items()})
            for spec, values in found.items()}


@cache
def read_models(directory: Path = DATA / "models") -> dict[str, ModelSpec]:
    """Return every model spec in a directory, by name.

    Raises:
        ValueError: A file's name differs from its spec's name, a key
            is not a ModelSpec field, or a ``base`` is missing or forms
            a cycle.
    """
    return _read(ModelSpec, "models", directory)


@cache
def read_devices(directory: Path = DATA / "devices") -> dict[str, DeviceSpec]:
    """Return every device spec in a directory, by name.

    Raises:
        ValueError: A file's name differs from its spec's name, or a
            key is not a DeviceSpec field.
    """
    return _read(DeviceSpec, "devices", directory)
