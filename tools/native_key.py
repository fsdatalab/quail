"""Print the key of the Rust sources that _quail_native is built from.

Two checkouts with the same key build the same module, so a module one
of them built serves the other. The key is a hash of the files Cargo
compiles: the crates, the manifests, the lock file, the pinned
toolchain, and Cargo's settings. A file Git ignores does not count.

    uv run python tools/native_key.py
"""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
           ".cargo")


def native_key() -> str:
    """Return the first 16 hex digits of the hash of the Rust sources."""
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "--", *SOURCES],
        cwd=ROOT, check=True, capture_output=True, text=True).stdout
    digest = hashlib.sha256()
    for name in sorted(set(listed.splitlines())):
        path = ROOT / name
        if not path.is_file():
            continue
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


if __name__ == "__main__":
    sys.stdout.write(native_key() + "\n")
