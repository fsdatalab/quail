"""The TOML spec files: every built-in loads, and a variant reads its base."""

import pytest

from quail.specs import MODELS
from quail.specs.loading import read_models


def _write(directory, name, text):
    (directory / f"{name}.toml").write_text(text)


SMALL = """\
name = "small"
params = 1e9
layers = 2
hidden = 8
num_q_heads = 1
num_kv_heads = 1
d_head = 8
ffn_width = 16
w_bytes = 1
"""


def test_a_variant_keeps_its_base_fields_and_changes_its_own(tmp_path):
    _write(tmp_path, "small", SMALL)
    _write(tmp_path, "small-bf16", 'base = "small"\nname = "small-bf16"\nw_bytes = 2\n')
    models = read_models(tmp_path)
    assert models["small-bf16"].layers == 2
    assert models["small-bf16"].w_bytes == 2.0
    assert models["small"].w_bytes == 1.0


def test_every_built_in_model_loads_with_its_file_name():
    assert all(name == spec.name for name, spec in MODELS.items())
    assert "qwen3-4b-fp8" in MODELS


def test_a_misspelled_field_is_refused(tmp_path):
    _write(tmp_path, "typo", 'name = "typo"\nlayerz = 3\n')
    with pytest.raises(ValueError, match="layerz"):
        read_models(tmp_path)


def test_a_file_named_for_another_spec_is_refused(tmp_path):
    _write(tmp_path, "one", 'name = "two"\n')
    with pytest.raises(ValueError, match="names another spec"):
        read_models(tmp_path)
