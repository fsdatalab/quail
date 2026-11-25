#!/usr/bin/env bash
# Install Quail for development: the Python package from this checkout,
# the Rust module _quail_native, and the program .venv/bin/quail-server.
#
# The script first looks for a prebuilt wheel for this checkout's Rust
# sources (tools/native_key.py), which CI uploads for the main branch.
# With one, no Rust compiler is needed. Without one, or with --compile,
# it builds the module with the repository's pinned Rust.
#
# Usage: tools/build.sh [--compile] [uv sync options]
set -euo pipefail

cd "$(dirname "$0")/.."

compile=""
if [ "${1:-}" = "--compile" ]; then
  compile=1
  shift
fi
case "$(uname -sm)" in
  "Linux x86_64") platform="linux-x86_64" ;;
  "Linux aarch64") platform="linux-aarch64" ;;
  "Darwin arm64") platform="macos-arm64" ;;
  *) platform="" ;;
esac
wheels="${QUAIL_NATIVE_WHEELS_URL:-https://github.com/fsdatalab/quail/releases/download/native-wheels}"
if [ -z "$compile" ] && [ -n "$platform" ] && command -v uv >/dev/null; then
  key="$(uv run --no-project python tools/native_key.py)"
  wheel="$(mktemp)"
  trap 'rm -f "$wheel"' EXIT
  if curl -fsSL -o "$wheel" "$wheels/quail_native-$key-$platform.whl"; then
    echo "prebuilt:   _quail_native for sources $key ($platform)"
    uv sync --frozen --no-install-project "$@"
    # the module beside the environment's packages, and this checkout on
    # the path, as an editable install leaves them
    .venv/bin/python - "$wheel" "$PWD" <<'PY'
import shutil, sys, sysconfig, zipfile
from pathlib import Path

site = Path(sysconfig.get_paths()["platlib"])
shutil.rmtree(site / "_quail_native", ignore_errors=True)
with zipfile.ZipFile(sys.argv[1]) as wheel:
    wheel.extractall(site, [name for name in wheel.namelist()
                            if name.startswith("_quail_native/")])
# the program an editable install writes from [project.scripts]
program = Path(sys.executable).parent / "quail-server"
program.unlink(missing_ok=True)
program.write_text(f"#!{sys.executable}\nfrom quail.remote.server import main\nmain()\n")
program.chmod(0o755)
(site / "quail_engine.pth").write_text(sys.argv[2])
PY
    .venv/bin/python -c "import _quail_native, quail"
    echo "Run tools with 'uv run --no-sync', or set UV_NO_SYNC=1: a plain"
    echo "'uv run' compiles the module."
    exit 0
  fi
  echo "prebuilt:   none for sources $key ($platform); compiling"
fi

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
