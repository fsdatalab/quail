"""The quail-server command: defaults, GPU detection, and the start banner."""

from pathlib import Path

from quail.server.__main__ import (
    describe,
    parse_args,
    settings_from_args,
)

H100 = "NVIDIA H100 80GB HBM3"


def test_no_options_means_every_model_on_the_gpu_found_and_says_how_to_connect():
    args = parse_args([])
    settings = settings_from_args(args, gpu_name=lambda: H100)
    assert settings.device == "h100-sxm"
    assert "qwen3-4b-fp8" in settings.models
    assert "diffusion-gemma-26b-a4b-fp8" in settings.models
    assert settings.data_dir == Path("~/.quail/server").expanduser()
    assert settings.gpus == (1,) and settings.backends == ("quail",)
    assert settings.token is None
    assert (args.host, args.port) == ("127.0.0.1", 8642)

    text = describe(settings, "0.0.0.0", 8642)
    assert "Quail Server on h100-sxm: " in text
    assert f"data: {settings.data_dir}" in text
    assert 'endpoint="http://127.0.0.1:8642"' in text
    assert "token: none" in text
    assert 'endpoint="http://gpu-host:1"' in describe(settings, "gpu-host", 1)
