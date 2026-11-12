"""The GPU worker process: ``python -m quail.worker <socket> <gpu index>``.

Quail starts one per GPU (``quail_python::workers``). The worker loads
the model onto its GPU and serves the stages Quail sends it. It also
describes a configuration to Quail, which plans in Rust and reads the
models' definitions from here.
"""

import json
import sys

from quail.api.engine import Engine
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


def describe(config: dict) -> dict:
    """Return what Quail plans a configuration's regions with.

    Args:
        config: The model, device, gpus, and backend of a session.

    Returns:
        The model's chat turn and prompt layout, the logical rules, the
        tokenizer file, and the planning settings; the settings are
        None when the backend refuses the configuration.
    """
    engine = Engine.open(config["model"], config["device"], config["gpus"],
                         config["backend"])
    try:
        settings = engine.settings
        return {
            "turn": list(engine.turn), "layout": engine.layout,
            "rules": engine.rules, "tokenizer_file": engine.tokenizer_file,
            "settings": None if settings is None else json.loads(settings),
        }
    finally:
        engine.close()


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), worker_registry, describe)
