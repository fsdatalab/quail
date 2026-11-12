"""The GPU worker process: ``python -m quail.worker <socket> <gpu index>``.

Quail starts one per GPU (``quail_python::workers``). The worker loads
the model onto its GPU and serves the stages Quail sends it.
"""

import sys

from quail.backends.quail.worker import main
from quail.builtins import built_in_registry


def worker_registry(extensions):
    """Return the built in registry with a session's extension modules loaded.

    Args:
        extensions: The names of the session's extension modules.
    """
    registry = built_in_registry()
    for module in extensions:
        registry.load_extension(module)
    return registry


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), worker_registry)
