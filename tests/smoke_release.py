"""Check the installed release package and its metadata."""

from importlib.metadata import metadata

import quail

package = metadata("quail-engine")

assert package["Name"] == "quail-engine"
assert package["License-Expression"] == "MIT"
assert quail.Session
