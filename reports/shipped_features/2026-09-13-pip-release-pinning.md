# Pip release dependency pins

## What changed

Quail's runtime, development, and build dependencies now use exact versions.
The versions come from the existing `uv.lock` resolution. The lock keeps the
full transitive dependency graph fixed for repository installs.

The runnable Modal quickstart and the documentation's Modal example install
the same exact Linux runtime requirements as the package.

The PyPI distribution is named `quail-engine`, while its Python import remains
`quail`. This avoids the unrelated package that already owns the `quail`
distribution name. The project now uses the MIT license and publishes its
repository and issue URLs in package metadata.

`tools/release.sh` checks the version and repository state, then pushes the
version tag. Following uv's official guide, the tag starts separate build and
publish jobs. They use `uv build --no-sources` and upload with `uv publish`
through PyPI trusted publishing. The workflow actions, `uv`, and Python
dependencies all use exact versions. These changes do not publish a release
by themselves.

## Prediction and validation

Prediction before validation: `uv` will accept the lock without changing its
resolved packages, the CPU checks will pass, and the built wheel will list
only exact direct requirements.

The lock check resolved 234 packages without changing the lock. Ruff, the
long-string check, and Vulture passed. `uv build` produced the sdist and wheel,
named `quail_engine-0.1.0`. The wheel lists all nine platform-specific direct
requirements with exact versions.

The CPU tests could not start in this environment because dependency setup
could not read the private `fsdatalab/quail-bench` repository. GitHub CI used
its separate `QUAILB_TOKEN` and passed the full existing test suite.

Prediction before release automation validation: the script will reject a
version that differs from `pyproject.toml`, the workflow will parse, and the
built package will carry the MIT license metadata and file.

The Bash parser accepted the script. It rejected version `9.9.9` and refused
to release from a feature branch. Actionlint 1.7.7 accepted the publishing
workflow. `uv build --no-sources` built both package formats. The wheel reports
`License-Expression: MIT` and includes `dist-info/licenses/LICENSE`.

No model run was needed because this change does not alter query execution.
There is therefore no Modal function call id or `quail-results` volume path.

## Work still required before release

1. Decide how pip users get QUAIL-B. It is currently a development dependency
   fetched from a pinned Git commit, while the quickstart imports it.
2. Create the protected `pypi` GitHub environment and register the pending
   trusted publisher on PyPI.
3. Add author metadata if the project wants a named person or organization
   on PyPI.
4. Merge this change, then run `tools/release.sh 0.1.0` from `main`.
