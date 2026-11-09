#!/usr/bin/env bash
# Build the editable Python extension with the repository's pinned Rust.
# Usage: tools/build.sh [uv sync options]
set -euo pipefail

cd "$(dirname "$0")/.."
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
# Prefer this Cargo home even when a different installation is on PATH.
if [ -d "$CARGO_HOME/bin" ]; then
  export PATH="$CARGO_HOME/bin:$PATH"
fi
export MATURIN_NO_INSTALL_RUST=1

for tool in rustup cargo rustc uv; do
  if ! command -v "$tool" >/dev/null; then
    echo "Missing $tool. Install rustup (https://rustup.rs) and uv (https://docs.astral.sh/uv/), then rerun this script." >&2
    exit 1
  fi
done
if [ "$(uname -sm)" = "Linux x86_64" ]; then
  for tool in clang ld.lld; do
    if ! command -v "$tool" >/dev/null; then
      echo "Missing $tool: install clang and lld, as required by .cargo/config.toml." >&2
      exit 1
    fi
  done
fi

# rustup selects rust-toolchain.toml; it downloads that toolchain if missing.
rustc_version="$(rustc -V)"
echo "cargo:      $(command -v cargo)"
echo "compiler:   $(rustup which rustc)"
echo "rustc:      $rustc_version"
echo "CARGO_HOME: $CARGO_HOME"
echo "target:     ${CARGO_TARGET_DIR:-$PWD/target}"
exec uv sync --frozen "$@"
