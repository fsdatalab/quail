# Pip release dependency pins

## What changed

Quail's runtime, development, and build dependencies now use exact versions.
The versions come from the existing `uv.lock` resolution. The lock keeps the
full transitive dependency graph fixed for repository installs.

The runnable Modal quickstart and the documentation's Modal example install
the same exact Linux runtime requirements as the package. A test checks that
the package and quickstart lists remain equal.

This prepares a wheel for repeatable validation. It does not publish one.

## Prediction and validation

Prediction before validation: `uv` will accept the lock without changing its
resolved packages, the CPU checks will pass, and the built wheel will list
only exact direct requirements.

The lock check resolved 234 packages without changing the lock. Ruff, the
long-string check, and Vulture passed. `uv build` produced the sdist and wheel,
and Twine accepted both. The wheel contains 75 files and lists all nine
platform-specific direct requirements with exact versions.

The CPU tests could not start in this environment. Dependency setup could not
read the private `fsdatalab/quail-bench` repository. CI has the separate
`QUAILB_TOKEN` needed to run them. This is an authentication limit of this
agent, not a test failure, but the prediction that all CPU tests pass remains
unconfirmed until CI finishes.

No model run was needed because this change does not alter query execution.
There is therefore no Modal function call id or `quail-results` volume path.

## Work still required before release

1. Choose a PyPI distribution name. `quail` is owned by the unrelated
   Contextual Dynamics Lab project, whose current PyPI release is 0.2.2.
   The Python import can remain `quail` under a different distribution name.
2. Choose a license. The repository has no `LICENSE` file or package license
   metadata.
3. Add the chosen distribution name, license, authors, repository URL, and
   issue URL to `pyproject.toml`.
4. Decide how pip users get QUAIL-B. It is currently a development dependency
   fetched from a pinned Git commit, while the quickstart imports it.
5. Add a trusted PyPI publishing workflow, configure the PyPI project, and
   protect the release environment.
6. Pick the first public version, build an sdist and wheel, inspect both, and
   install the wheel in a clean Python 3.12 environment.
7. Publish to TestPyPI first, test that install, then create the production
   tag and publish the same artifacts to PyPI.
