# Stable benchmark predicate keys

## What changed

`PredicateSpec` now has one identifier named `key`. The judge pass writes the
key into each label manifest. The SoL report and migration scripts also use the
key when they load or report predicate results.

The old short code field was removed from the active benchmark code. Existing
label sets still work because their manifests already contain the stable key.

Vulture and Ruff are now development dependencies. The Vulture settings in
`pyproject.toml` check `quail/` at 70 percent confidence, so `uv run vulture`
is the complete command.

## Why

The short code duplicated the stable key and required every benchmark script
to maintain a second identifier. The stable key is already used for label set
identities and collection manifests.

Declaring the code checks in the development dependencies makes a clean
checkout use the same versions.

## Checks

The full CPU suite passes with 234 tests. Ruff, Vulture, and the Git whitespace
check also pass.
