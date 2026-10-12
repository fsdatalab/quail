"""Saved AI.PARSE results, so a later query reads a row instead of parsing it.

A row's text is saved under a key made of its file's path, size, and
modification time, its pages, the parser and the parser's version, and
the mode. A file that changes, or another parser, mode, or version,
misses. Texts live in one file each under QUAIL_PARSE_DIR (default
~/.cache/quail/parses).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

DEFAULT_DIRECTORY = "~/.cache/quail/parses"


class SavedParses:
    """Read and write saved parse results in one directory.

    Args:
        directory: The directory; QUAIL_PARSE_DIR or DEFAULT_DIRECTORY
            when None.
    """

    def __init__(self, directory: str | None = None):
        self.directory = Path(os.path.expanduser(
            directory or os.environ.get("QUAIL_PARSE_DIR", DEFAULT_DIRECTORY)))

    @staticmethod
    def key(path_file, pages, parser: str, mode: str) -> str:
        """The key of one row's parse."""
        from quail.parsers import parser_spec

        value = repr((path_file.path, path_file.size, path_file.mtime_ns,
                      tuple(pages), parser, parser_spec(parser).version, mode))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.txt"

    def get(self, key: str) -> str | None:
        """The saved text, or None."""
        try:
            return self._path(key).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def put(self, key: str, text: str) -> None:
        """Save a text; a reader never sees a partly written file."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
