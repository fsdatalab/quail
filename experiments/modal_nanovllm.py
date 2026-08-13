"""Batch-size sweep with nano-vLLM on the Quail filter workload.

nano-vLLM does not support the FP8 checkpoint used by the main
experiment, so this runs Qwen3-4B in BF16. The absolute throughput is
not comparable with the vLLM sweep. The shape of the curve can show
whether the B=1024 slowdown is specific to vLLM.

Run:
    modal run experiments/modal_nanovllm.py::batchsweep
"""

import modal

from workload import (IMAGE_BASE, IMAGE_STAMP, build_corpus, hf_cache,
                      results_vol)


NANO_COMMIT = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
NANO_MODEL = "Qwen/Qwen3-4B"
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3.post1/"
    "flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-"
    "cp312-cp312-linux_x86_64.whl"
)

nano_image = (
    modal.Image.from_registry(IMAGE_BASE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "python -m pip install 'torch==2.9.1' "
        "--index-url https://download.pytorch.org/whl/cu130"
    )
    .pip_install("transformers>=4.51.0", "huggingface_hub", "pandas",
                 "pyarrow", "numpy", "einops", "xxhash")
    .pip_install(FLASH_ATTN_WHEEL)
    .pip_install(
        f"git+https://github.com/GeeeekExplorer/nano-vllm.git@{NANO_COMMIT}",
        extra_options="--no-deps",
    )
    .add_local_python_source("workload")
)

app = modal.App("quail-nanovllm")


@app.function(image=nano_image, gpu="H100!", timeout=7200, memory=65536,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/results": results_vol})
def batchsweep(n_docs: int = 10000,
               batch_sizes: str = "512,1024,2048,4096,8192,16384,25305"
               ) -> str:
    import atexit
    import gc
    import json
    import time

    import torch
    from huggingface_hub import snapshot_download
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_path = snapshot_download(NANO_MODEL)
    tok = AutoTokenizer.from_pretrained(model_path)
    body_ids, q_ids, _flags = build_corpus(tok, n_docs)
    prompts = [body_ids[i] + q_ids[0] for i in range(n_docs)]
    prompt_tokens = sum(len(p) for p in prompts)
    sp = SamplingParams(temperature=1e-5, max_tokens=1)

    results = []
    for b in (int(x) for x in batch_sizes.split(",")):
        seqs = min(4096, b)
        print(f"\n[nano sweep] B={b}, max_num_seqs={seqs}", flush=True)
        llm = LLM(model_path, max_model_len=4608,
                  max_num_seqs=seqs, max_num_batched_tokens=b,
                  gpu_memory_utilization=0.88)
        t0 = time.time()
        llm.generate(prompts, sp, use_tqdm=False)
        wall = time.time() - t0
        row = dict(B=b, max_num_seqs=seqs, wall=round(wall, 3),
                   prompt_tokens=prompt_tokens,
                   rate=round(prompt_tokens / wall, 1))
        results.append(row)
        print(f"[nano sweep B={b}] wall {wall:.2f}s, "
              f"rate {prompt_tokens / wall:,.0f} tok/s", flush=True)

        atexit.unregister(llm.exit)
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    payload = dict(engine="nano-vllm", nano_commit=NANO_COMMIT,
                   model=NANO_MODEL, dtype="bfloat16",
                   n_docs=n_docs, image=IMAGE_STAMP, results=results)
    outpath = "/results/batchsweep_nanovllm.json"
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    text = json.dumps(payload, indent=2)
    print(text, flush=True)
    return text
