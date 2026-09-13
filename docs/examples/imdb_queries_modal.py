"""Reproduce the documentation's filter and join outputs on real IMDB reviews.

Reads eight reviews from the pinned stanfordnlp/imdb revision QUAIL-B
uses, registers them with the twelve QUAIL-B movie aspects, and runs
the two quickstart queries on Modal. The printed output is what the
user guide shows.

    uv run modal run docs/examples/imdb_queries_modal.py \
        2>&1 | tee results/docs_quickstart.log
"""

import json
import uuid
from pathlib import Path

import modal
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import quail

app = modal.App("quail-engine")
IMDB_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"
ASPECTS = (
    "the acting",
    "the plot",
    "the directing",
    "the cinematography",
    "the soundtrack",
    "the pacing",
    "the ending",
    "the dialogue",
    "the special effects",
    "the character development",
    "the screenplay",
    "the editing",
)
F1 = (
    "Judge strictly from the review above whether it mentions at least one "
    "positive aspect of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
    "review mentions at least one positive aspect of the movie, FALSE "
    "otherwise."
)
DISCUSS_ASPECT = (
    "Does the review in DOCUMENT {0} discuss the movie aspect in "
    "DOCUMENT {1}?"
)
IMAGE_REQUIREMENTS = (
    "sqlglot==30.17.0",
    "transformers==5.15.0",
    "huggingface-hub==1.27.0",
    "pyarrow==25.0.1",
    "numpy==2.3.5",
    "bpe-qwen==0.1.5",
    "datasets==5.0.1",
    "vllm==0.26.0",
)


def load_reviews(count: int = 8) -> pa.Table:
    """Return short, tag-free reviews from the pinned IMDB train split."""
    path = hf_hub_download(
        "stanfordnlp/imdb",
        "plain_text/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=IMDB_REVISION,
    )
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    picked = [t for t in texts if 250 < len(t) < 600 and "<br" not in t]
    picked = picked[:count]
    return pa.table({
        "id": [f"rv{i}" for i in range(len(picked))],
        "body": picked,
    })


def section(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install(*IMAGE_REQUIREMENTS)
    .env({
        "QUAIL_CACHE_DIR": "/root/.cache/kernels",
        "VLLM_CACHE_ROOT": "/root/.cache/kernels/vllm",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DG_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "DG_JIT_CACHE_DIR": "/root/.cache/kernels/deep_gemm",
        "TRITON_CACHE_DIR": "/root/.cache/kernels/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/kernels/torchinductor",
    })
    .add_local_python_source("quail")
)
results_vol = modal.Volume.from_name("quail-results", create_if_missing=True)
kernel_cache = modal.Volume.from_name("quail-kernel-cache", create_if_missing=True)
volumes = {
    "/results": results_vol,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": kernel_cache,
}


@app.function(
    image=image,
    gpu="H100!", memory=98304, volumes=volumes, timeout=1200,
)
def run_queries() -> None:
    destination = Path("/results/docs-quickstart") / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=True)
    reviews = load_reviews()
    aspects = pa.table({
        "id": [f"as{i}" for i in range(len(ASPECTS))],
        "aspect": list(ASPECTS),
    })
    section("reviews")
    for row in reviews.to_pylist():
        print(f"{row['id']}: {row['body']}")
    section("aspects")
    print(aspects.to_pydict())

    config = quail.EngineConfig(
        gpus=1,
        model="qwen3-4b-fp8",
        backend="quail",
        device="h100-sxm",
    )
    with quail.Session(config) as session:
        session.register("reviews", quail.DocumentProvider.from_table(
            reviews, id_col="id"))
        session.register("aspects", quail.DocumentProvider.from_table(
            aspects, id_col="id"))

        filter_sql = f"""
            SELECT r.id
            FROM reviews r
            WHERE AI_FILTER(PROMPT('{F1}', r.body))
        """
        section("filter sql")
        print(filter_sql)
        query = session.sql(filter_sql)
        section("filter explain")
        print(query.explain())
        section("filter run")
        result = query.run()
        path = destination / "filter.json"
        result.report["result_volume_path"] = str(path)
        path.write_text(json.dumps(result.report))
        results_vol.commit()
        print(result.to_rows())
        section("filter report")
        print(json.dumps(result.report, indent=2, default=str))

        join_sql = f"""
            SELECT r.id, a.aspect
            FROM reviews r
            JOIN aspects a
              ON AI_FILTER(PROMPT('{DISCUSS_ASPECT}', r.body, a.aspect),
                           {{'selectivity': 0.15}})
            WHERE AI_FILTER(PROMPT('{F1}', r.body), {{'selectivity': 0.6}})
        """
        section("join sql")
        print(join_sql)
        join = session.sql(join_sql)
        section("join explain")
        print(join.explain())
        section("join run")
        pairs = join.run()
        path = destination / "join.json"
        pairs.report["result_volume_path"] = str(path)
        path.write_text(json.dumps(pairs.report))
        results_vol.commit()
        print(pairs.collect().to_pandas().to_string(index=False))
        section("join report")
        print(json.dumps(pairs.report, indent=2, default=str))
    kernel_cache.commit()


@app.local_entrypoint()
def main() -> None:
    call = run_queries.spawn()
    print(f"function call id: {call.object_id}", flush=True)
    call.get()
