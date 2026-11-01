"""Serve saved Perfetto traces from the results volume.

Deploys one web endpoint under the quail-milestone1 app name. It serves
only the gzipped traces profile_query.py saves under
/results/ablations/traces, with CORS for ui.perfetto.dev, so a link opens
a trace in the Perfetto UI:

    uv run modal deploy experiments/cells/trace_server.py
    https://ui.perfetto.dev/#!/?url=<endpoint>/<trace file name>
"""

import modal

from quail.deploy.images import cpu_image

app = modal.App("quail-milestone1")
results_vol = modal.Volume.from_name("quail-results")
TRACE_DIR = "/results/ablations/traces"
PERFETTO_ORIGIN = "https://ui.perfetto.dev"


@app.function(image=cpu_image(), volumes={"/results": results_vol},
              max_containers=1)
@modal.asgi_app()
def traces():
    """Return the app that serves one trace file per request."""
    from pathlib import Path

    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import FileResponse, Response
    from starlette.routing import Route

    root = Path(TRACE_DIR)

    async def trace(request):
        name = request.path_params["name"]
        # names only: no directories, nothing outside the trace folder
        if "/" in name or not name.endswith(".json.gz"):
            return Response(status_code=404)
        if not (root / name).is_file():
            # a trace saved after this container started
            results_vol.reload()
        if not (root / name).is_file():
            return Response(status_code=404)
        return FileResponse(root / name, media_type="application/gzip")

    return Starlette(
        routes=[Route("/{name}", trace)],
        middleware=[Middleware(CORSMiddleware, allow_origins=[PERFETTO_ORIGIN],
                               allow_methods=["GET"])])
