"""Start Quail Server: ``python -m quail.server`` or ``quail-server``.

On a machine with one supported GPU, ``quail-server`` with no options
starts a server that runs every registered model on the GPU it finds,
keeps its data under ``~/.quail/server``, and listens on
``127.0.0.1:8642``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from quail.server.scheduler import DEFAULT_TIMEOUT_S

DEFAULT_DATA_DIR = Path("~/.quail/server")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642

# a substring of the name CUDA reports for the GPU, and the registered
# device it is. The H100 spec is the SXM part with HBM3; the PCIe and
# NVL parts have other memory bandwidth and are not listed.
GPU_NAMES = (
    ("H100 80GB HBM3", "h100-sxm"),
    ("RTX PRO 6000 Blackwell", "rtx-pro-6000-blackwell-server"),
)


def cuda_device_name() -> str | None:
    """The name CUDA reports for GPU 0, or None without a usable GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def detect_device(gpu_name: str | None) -> str:
    """Map the GPU's CUDA name to a registered device name.

    Raises SystemExit with a message that lists the choices when there
    is no GPU or its name is not one this server knows.
    """
    choices = ", ".join(device for _, device in GPU_NAMES)
    if gpu_name is None:
        raise SystemExit(
            "quail-server: no CUDA GPU found; pass --device with one of "
            f"{choices}")
    for pattern, device in GPU_NAMES:
        if pattern in gpu_name:
            return device
    raise SystemExit(
        f"quail-server: GPU {gpu_name!r} is not a device this server knows; "
        f"pass --device with one of {choices}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quail-server",
        description="Run Quail Server, the optional query server, on this host.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory for the SQLite file, inputs, and "
                             f"results; default {DEFAULT_DATA_DIR}")
    parser.add_argument("--model", action="append", dest="models",
                        metavar="MODEL",
                        help="a model this server runs; repeat for more. "
                             "Default: every registered model")
    parser.add_argument("--device",
                        help="the device name every query must ask for. "
                             "Default: the GPU this machine has")
    parser.add_argument("--gpus", type=int, action="append", metavar="N",
                        help="a GPU count this server accepts; default 1")
    parser.add_argument("--backend", action="append", dest="backends",
                        metavar="BACKEND", help="an accepted backend; default quail")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"address to listen on; default {DEFAULT_HOST}. "
                             "Use 0.0.0.0 to accept other machines")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to listen on; default {DEFAULT_PORT}")
    parser.add_argument("--default-timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        metavar="SECONDS",
                        help="execution timeout when a submission names none; "
                             f"default {DEFAULT_TIMEOUT_S:g}")
    parser.add_argument("--max-timeout", type=float, default=4 * 3600.0,
                        metavar="SECONDS",
                        help="largest timeout a submission may ask for; "
                             "default 14400 (4 hours)")
    parser.add_argument("--max-upload-gib", type=float, default=8.0,
                        help="largest input snapshot accepted; default 8")
    parser.add_argument("--token", default=os.environ.get("QUAIL_SERVER_TOKEN"),
                        help="bearer token every /v1 request must carry; "
                             "default QUAIL_SERVER_TOKEN, none when unset")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace, *,
                       gpu_name=cuda_device_name):
    """Build ServerSettings from parsed options.

    Args:
        args: The parsed command line.
        gpu_name: Called for the GPU's CUDA name when ``--device`` is
            not given.
    """
    from quail.builtins import built_in_registry
    from quail.server.app import ServerSettings

    registry = built_in_registry()
    models = tuple(args.models or sorted(registry.models))
    for model in models:
        registry.model(model)
    device = args.device or detect_device(gpu_name())
    registry.device(device)
    return ServerSettings(
        data_dir=args.data_dir.expanduser(),
        models=models,
        device=device,
        gpus=tuple(args.gpus or (1,)),
        backends=tuple(args.backends or ("quail",)),
        default_timeout_s=args.default_timeout,
        max_timeout_s=args.max_timeout,
        max_upload_bytes=int(args.max_upload_gib * (1 << 30)),
        token=args.token or None,
    )


def describe(settings, host: str, port: int) -> str:
    """The lines printed at startup: what runs where, and how to connect."""
    address = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return "\n".join([
        f"Quail Server on {settings.device}: {', '.join(settings.models)}",
        f"data: {settings.data_dir}",
        f"token: {'required' if settings.token else 'none'}",
        f"connect: quail.Session(config, endpoint=\"http://{address}:{port}\")",
        "the first query loads its model; later queries reuse it",
    ])


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = settings_from_args(args)
    import uvicorn

    from quail.server.app import create_app

    print(describe(settings, args.host, args.port), flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
