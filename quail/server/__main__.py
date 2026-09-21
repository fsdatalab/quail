"""Start Quail Server: ``python -m quail.server`` or ``quail-server``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from quail.server.scheduler import DEFAULT_TIMEOUT_S


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quail-server",
        description="Run Quail Server, the optional query server, on this host.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="directory for the SQLite file, inputs, and results")
    parser.add_argument("--model", action="append", required=True,
                        dest="models", metavar="MODEL",
                        help="a model this server runs; repeat for more")
    parser.add_argument("--device", required=True,
                        help="the device name every query must ask for")
    parser.add_argument("--gpus", type=int, action="append", metavar="N",
                        help="a GPU count this server accepts; default 1")
    parser.add_argument("--backend", action="append", dest="backends",
                        metavar="BACKEND", help="an accepted backend; default quail")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--default-timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        metavar="SECONDS",
                        help="execution timeout when a submission names none")
    parser.add_argument("--max-timeout", type=float, default=4 * 3600.0,
                        metavar="SECONDS",
                        help="largest timeout a submission may ask for")
    parser.add_argument("--max-upload-gib", type=float, default=8.0,
                        help="largest input snapshot accepted")
    parser.add_argument("--token", default=os.environ.get("QUAIL_SERVER_TOKEN"),
                        help="bearer token every /v1 request must carry; "
                             "default QUAIL_SERVER_TOKEN, none when unset")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace):
    from quail.builtins import built_in_registry
    from quail.server.app import ServerSettings

    registry = built_in_registry()
    for model in args.models:
        registry.model(model)
    registry.device(args.device)
    return ServerSettings(
        data_dir=args.data_dir,
        models=tuple(args.models),
        device=args.device,
        gpus=tuple(args.gpus or (1,)),
        backends=tuple(args.backends or ("quail",)),
        default_timeout_s=args.default_timeout,
        max_timeout_s=args.max_timeout,
        max_upload_bytes=int(args.max_upload_gib * (1 << 30)),
        token=args.token or None,
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = settings_from_args(args)
    import uvicorn

    from quail.server.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
