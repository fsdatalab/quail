"""The quail-server command: defaults, GPU detection, and the start banner."""

import importlib.util
import sys
from pathlib import Path

import pytest

from quail.server.__main__ import (
    describe,
    detect_device,
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


def test_options_override_detection_and_defaults(tmp_path):
    args = parse_args([
        "--data-dir", str(tmp_path), "--model", "qwen3-4b-fp8",
        "--device", "rtx-pro-6000-blackwell-server", "--gpus", "1",
        "--gpus", "2", "--host", "0.0.0.0", "--port", "9000",
        "--token", "secret", "--max-upload-gib", "0.5"])

    def no_gpu():
        raise AssertionError("--device given, the GPU is not asked")

    settings = settings_from_args(args, gpu_name=no_gpu)
    assert settings.models == ("qwen3-4b-fp8",)
    assert settings.device == "rtx-pro-6000-blackwell-server"
    assert settings.gpus == (1, 2) and settings.token == "secret"
    assert settings.max_upload_bytes == 1 << 29
    assert "token: required" in describe(settings, "0.0.0.0", 9000)
    with pytest.raises(ValueError, match="unknown model 'gpt-9'"):
        settings_from_args(parse_args(["--model", "gpt-9"]), gpu_name=lambda: H100)

    assert detect_device(H100) == "h100-sxm"
    assert detect_device("NVIDIA L40S") == "l40s"
    assert detect_device("NVIDIA B200") == "b200"
    assert detect_device(
        "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    ) == "rtx-pro-6000-blackwell-server"
    with pytest.raises(SystemExit, match="no CUDA GPU found.*h100-sxm"):
        detect_device(None)
    with pytest.raises(SystemExit, match="'NVIDIA H100 PCIe' is not"):
        detect_device("NVIDIA H100 PCIe")


@pytest.mark.parametrize("device,gpu", [
    ("h100-sxm", "H100!"), ("l40s", "L40S"), ("b200", "B200"),
])
def test_modal_preserves_the_deployed_device_on_remote_import(monkeypatch, device, gpu):
    import modal

    deployment = {}
    decorate = modal.App.cls

    def capture(app, **options):
        deployment.update(options)
        return decorate(app, **options)

    def load():
        spec = importlib.util.find_spec("quail.server.modal_app")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setattr(modal.App, "cls", capture)
    monkeypatch.delenv("QUAIL_SERVER_DEVICE", raising=False)
    args = [] if device == "h100-sxm" else ["--device", device]
    monkeypatch.setattr(sys, "argv", ["quail.server.modal_app", *args])
    assert load().DEVICE == device
    assert deployment["gpu"] == gpu
    assert deployment["env"] == {"QUAIL_SERVER_DEVICE": device}

    monkeypatch.setenv("QUAIL_SERVER_DEVICE", deployment["env"]["QUAIL_SERVER_DEVICE"])
    monkeypatch.setattr(sys, "argv", ["modal._container_entrypoint"])
    assert load().DEVICE == device
    assert deployment["gpu"] == gpu
