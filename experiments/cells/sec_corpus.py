"""Build the SEC 8-K sample of quail-bench PR #50 on the results volume.

The quail-bench commit is the PR's head; the pin in pyproject.toml does
not have the SEC workload yet. The builder reads only the input columns
of the source's 217 parquet files with HTTP range reads, samples 100,000
filings, and writes sec_filings.parquet and manifest.json. The table is
also copied beside the other sf 1.0 tables so the extract_draft cell's
--table mode can read it. CPU only; the call runs detached.

    uv run modal run --detach experiments/cells/sec_corpus.py::build
"""

import shutil
from pathlib import Path

from bench.quailb_parallel import DATA_DIR, VOLUMES, app, results_vol
from quail.deploy.images import gpu_image

QUAIL_BENCH_PR50 = "e7255c214b41cb08b443c052f54f9715f85aa713"
image = gpu_image(packages=(
    "quail-b @ git+https://github.com/fsdatalab/quail-bench.git@" + QUAIL_BENCH_PR50,))


@app.function(image=image, timeout=4 * 3600, cpu=8, memory=32768, volumes=VOLUMES)
def build_sec_sample() -> dict:
    """Build the sample and return its manifest and text-length summary."""
    import json

    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from quail_b.sec import build_sec_corpus

    out = Path(DATA_DIR) / "sec"
    directory = build_sec_corpus(out, source_dir=Path("/tmp/sec-source"))
    table = pq.read_table(directory / "sec_filings.parquet")
    chars = np.asarray(pc.utf8_length(table.column("text")).to_pylist())
    shutil.copy(directory / "sec_filings.parquet",
                Path(DATA_DIR) / "sf1.0" / "sec_filings.parquet")
    results_vol.commit()
    return {
        "rows": table.num_rows,
        "text_chars_percentiles_10_50_90_99": np.percentile(
            chars, [10, 50, 90, 99]).astype(int).tolist(),
        "manifest": json.loads((directory / "manifest.json").read_text()),
        "saved": str(directory),
    }


@app.local_entrypoint()
def build():
    call = build_sec_sample.spawn()
    print(f"function call id: {call.object_id} (sec corpus)", flush=True)
