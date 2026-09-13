#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: tools/release.sh VERSION" >&2
    exit 2
fi

version="$1"
tag="v${version}"
root="$(git rev-parse --show-toplevel)"
cd "$root"

if [[ "$(uv version --short)" != "$version" ]]; then
    echo "pyproject.toml version does not match ${version}" >&2
    exit 1
fi

if [[ -n "$(git status --short)" ]]; then
    echo "the worktree must be clean" >&2
    exit 1
fi

uv lock --check
uv sync --frozen
uv run --frozen ruff check quail tests experiments reports tools
uv run --frozen python tools/check_long_strings.py
uv run --frozen vulture
uv run --frozen pytest -q

rm -f dist/*.whl dist/*.tar.gz
uv build
uvx --from twine==7.0.0 twine check dist/*
sha256sum dist/*

if [[ "${PUBLISH:-0}" != "1" ]]; then
    echo "release files are ready in dist/"
    echo "publish with: PUBLISH=1 tools/release.sh ${version}"
    exit 0
fi

if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "publishing requires the main branch" >&2
    exit 1
fi

git fetch origin main
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "main must match origin/main before publishing" >&2
    exit 1
fi

if git rev-parse "$tag" >/dev/null 2>&1; then
    echo "tag ${tag} already exists" >&2
    exit 1
fi

git tag -a "$tag" -m "Quail ${version}"
git push origin "$tag"
echo "pushed ${tag}; GitHub Actions will publish the checked release files"
