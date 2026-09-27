"""Measured H100 settings for large DiffusionGemma expert batches."""

import json
from pathlib import Path

# vLLM selects files by device and expert shape. Other devices use its defaults.
# Measurements: /results/ablations/diffusion_gemma_moe_tiles*.json.
TUNED = {
    "E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json": {
        "32768": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
        "65536": {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
    }
}


def write_configs(folder, base_folder) -> Path:
    """Extend the installed vLLM tables with Quail's measured batch sizes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, overrides in TUNED.items():
        source = Path(base_folder) / name
        if not source.is_file():
            (folder / name).unlink(missing_ok=True)
            continue
        table = json.loads(source.read_text())
        table.update(overrides)
        (folder / name).write_text(json.dumps(table, indent=1))
    return folder
