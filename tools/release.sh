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

if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "releases must start from the main branch" >&2
    exit 1
fi

git fetch origin main
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "main must match origin/main before publishing" >&2
    exit 1
fi

if git rev-parse --verify "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "tag ${tag} already exists" >&2
    exit 1
fi
if git ls-remote --exit-code --tags origin "refs/tags/${tag}" \
    >/dev/null 2>&1; then
    echo "tag ${tag} already exists on origin" >&2
    exit 1
fi

git tag -a "$tag" -m "Quail ${version}"
git push origin "$tag"
echo "pushed ${tag}; GitHub Actions will build, test, and publish the release"
