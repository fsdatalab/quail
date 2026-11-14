"""Check that the Python module and the workspace build one set of dependencies.

`uv sync` builds the package quail-python alone, and `cargo build
--workspace` builds every package. Cargo compiles a dependency again,
and everything that uses it, when its features differ between the two
builds. The script prints each such dependency and exits 1.
"""

import collections
import subprocess
import sys


def features(*selection: str) -> dict:
    """Return each package's features in the build of a selection."""
    tree = subprocess.run(
        ["cargo", "tree", "--locked", *selection, "-e", "normal",
         "-f", "{p}|{f}", "--prefix", "none"],
        check=True, capture_output=True, text=True).stdout
    found = collections.defaultdict(set)
    for line in tree.splitlines():
        package, _, names = line.replace(" (*)", "").partition("|")
        found[" ".join(package.split()[:2])].update(filter(None, names.split(",")))
    return found


def main() -> int:
    """Print the dependencies whose features differ; return 1 if any do."""
    module, workspace = features("-p", "quail-python"), features("--workspace")
    different = {package: sorted(workspace[package] - names)
                 for package, names in module.items() if names != workspace[package]}
    for package, missing in sorted(different.items()):
        print(f"{package}: the workspace adds {', '.join(missing)}; "
              "add them in crates/quail-python/Cargo.toml")
    return 1 if different else 0


if __name__ == "__main__":
    sys.exit(main())
