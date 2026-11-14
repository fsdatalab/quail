#!/usr/bin/env bash
# Build quail-server where a wheel build picks it up: maturin installs
# the files in _quail_native.data/scripts as programs beside Python.
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build --release --locked -p quail-server
mkdir -p _quail_native.data/scripts
cp target/release/quail-server _quail_native.data/scripts/
