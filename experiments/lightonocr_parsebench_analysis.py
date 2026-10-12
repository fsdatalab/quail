"""Compare LightOnOCR's vLLM runs on ParseBench with their speed of light.

Reads the files experiments/cells/lightonocr_parsebench_vllm.py saves
and prints, per run: measured pages per second, output tokens per page
by folder, an H100 speed-of-light estimate for the same pages, and how
many decode steps a draft taken from each page's stored text would
save under greedy draft-and-check.

The first argument is the directory the files were pulled into:

    modal volume get quail-results \
      ablations/lightonocr_parsebench/pages.parquet $W/
    modal volume get quail-results \
      ablations/lightonocr_parsebench/vllm_run_256_<run id>.parquet $W/
    modal volume get quail-results \
      ablations/lightonocr_parsebench/vllm_run_256_<run id>.json $W/
    uv run python experiments/lightonocr_parsebench_analysis.py $W

The speed of light assumes one H100 at its HBM bandwidth and dense bf16
peak, every page's image tokens prefilled once, decode reading the
weights once per step for a full batch of max_num_seqs sequences, and
each sequence reading its KV and reading and writing its Gated DeltaNet
state once per step. It leaves out the Gated DeltaNet prefill and
sampling, so it is a lower bound on time.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from quail.specs import H100_SXM

MODEL = "lightonai/LightOnOCR-3-0.8B"
MODEL_REVISION = "36d8636925327c671bd8e479c517a004f7566bd9"
DRAFT_LENGTHS = (4, 8)
MATCH_TOKENS = 3


def geometry() -> dict:
    """Read the parameter and state sizes from the checkpoint's config."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL, "config.json", revision=MODEL_REVISION)
    config = json.loads(Path(path).read_text())
    text, vision = config["text_config"], config["vision_config"]
    hidden, ffn = text["hidden_size"], text["intermediate_size"]
    head = text["head_dim"]
    full_layers = text["layer_types"].count("full_attention")
    linear_layers = text["num_hidden_layers"] - full_layers
    q_width = text["num_attention_heads"] * head
    kv_width = text["num_key_value_heads"] * head
    full = hidden * (2 * q_width + 2 * kv_width) + q_width * hidden
    key_width = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_width = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    linear = (hidden * (2 * key_width + 2 * value_width)
              + value_width * hidden)
    mlp = 3 * hidden * ffn
    body = full_layers * (full + mlp) + linear_layers * (linear + mlp)
    embedding = text["vocab_size"] * hidden
    v_hidden = vision["hidden_size"]
    vision_layer = 4 * v_hidden * v_hidden + 2 * v_hidden * vision["intermediate_size"]
    merged = v_hidden * vision["spatial_merge_size"] ** 2
    merger = merged * merged + merged * vision["out_hidden_size"]
    state = (linear_layers * text["linear_num_value_heads"]
             * text["linear_key_head_dim"] * text["linear_value_head_dim"] * 4)
    conv = (linear_layers * (2 * key_width + value_width)
            * (text["linear_conv_kernel_dim"] - 1) * 2)
    return dict(
        body_params=body, embedding_params=embedding,
        weight_bytes=2 * (body + embedding),
        kv_bytes_per_token=full_layers * 2 * kv_width * 2,
        state_bytes=state, conv_bytes=conv, full_layers=full_layers,
        q_width=q_width, vision_layer_params=vision_layer,
        vision_layers=vision["depth"], vision_hidden=v_hidden,
        merger_params=merger, patch=vision["patch_size"],
        merge=vision["spatial_merge_size"])


def page_flops(g: dict, width: int, height: int, prompt_tokens: int) -> float:
    """Return the vision encoder and prefill FLOPs for one page."""
    unit = g["patch"] * g["merge"]
    patches = (round(height / unit) * unit // g["patch"]) * (
        round(width / unit) * unit // g["patch"])
    vision = (2 * g["vision_layers"] * g["vision_layer_params"] * patches
              + 4 * g["vision_layers"] * patches ** 2 * g["vision_hidden"]
              + 2 * g["merger_params"] * patches / g["merge"] ** 2)
    prefill = (2 * g["body_params"] * prompt_tokens
               + 2 * g["full_layers"] * prompt_tokens ** 2 * g["q_width"])
    return vision + prefill


def speed_of_light(g: dict, records: list[dict], batch: int) -> dict:
    """Return the speed-of-light seconds for decode and for prefill."""
    decode_bytes, tokens, flops = 0.0, 0, 0.0
    per_token = 2 * g["state_bytes"] + 2 * g["conv_bytes"]
    for record in records:
        prompt, out = record["prompt_tokens"], record["output_tokens"]
        context = out * prompt + out * (out + 1) / 2
        decode_bytes += out * per_token + context * g["kv_bytes_per_token"]
        tokens += out
        flops += page_flops(g, record["width"], record["height"], prompt)
    decode_bytes += tokens / batch * g["weight_bytes"]
    decode_s = decode_bytes / H100_SXM.hbm_bw
    compute_s = flops / H100_SXM.bf16_flops
    return dict(decode_s=decode_s, compute_s=compute_s,
                overlapped_s=max(decode_s, compute_s),
                serial_s=decode_s + compute_s)


def draft_steps(output: list[int], stored: list[int], length: int) -> int:
    """Count greedy decode steps when drafts come from the stored text.

    A step checks the drafted tokens and keeps the longest prefix the
    model would have produced, plus the model's own next token. A draft
    is the `length` stored tokens after the latest place, past the last
    match, where the stored text repeats the output's last MATCH_TOKENS
    tokens; with no such place the step produces one token.
    """
    places = defaultdict(list)
    for i in range(len(stored) - MATCH_TOKENS + 1):
        places[tuple(stored[i:i + MATCH_TOKENS])].append(i + MATCH_TOKENS)
    steps, done, last = 0, 0, 0
    while done < len(output):
        accepted = 0
        if done >= MATCH_TOKENS:
            starts = places.get(tuple(output[done - MATCH_TOKENS:done]), ())
            start = next((s for s in starts if s >= last),
                         starts[0] if starts else None)
            if start is not None:
                draft = stored[start:start + length]
                while (accepted < len(draft) and done + accepted < len(output)
                       and draft[accepted] == output[done + accepted]):
                    accepted += 1
                last = start + accepted
        done += min(accepted + 1, len(output) - done)
        steps += 1
    return steps


def percentile(values: list[int], q: float) -> float:
    """Return the q-th percentile of `values` by nearest rank."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main(directory: str) -> None:
    """Print the analysis for every run summary in `directory`."""
    from transformers import AutoTokenizer

    root = Path(directory)
    stored = {(r["pdf"], r["page"]): r["stored_text"]
              for r in pq.read_table(root / "pages.parquet",
                                     columns=["pdf", "page", "stored_text"]
                                     ).to_pylist()}
    g = geometry()
    print(f"weights {g['weight_bytes'] / 1e9:.2f} GB per step, KV "
          f"{g['kv_bytes_per_token']} B per token, state "
          f"{g['state_bytes'] / 1e6:.1f} MB per sequence")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    for summary_path in sorted(root.glob("vllm_*.json")):
        summary = json.loads(summary_path.read_text())
        records = pq.read_table(summary_path.with_suffix(".parquet")).to_pylist()
        print(f"\n== {summary_path.name}: {summary['pages']} pages, "
              f"max_num_seqs {summary['max_num_seqs']}")
        print(f"measured: {summary['wall_s']:.1f} s, "
              f"{summary['pages_per_s']:.2f} pages/s, "
              f"{summary['output_tokens_per_s']:.0f} output tokens/s, "
              f"GPU utilization {summary['gpu_utilization_mean']:.0f}%, "
              f"{summary['pages_at_max_tokens']} pages at the output limit")
        sol = speed_of_light(g, records, summary["max_num_seqs"])
        wall = summary["wall_s"]
        print(f"speed of light: decode {sol['decode_s']:.1f} s, vision and "
              f"prefill {sol['compute_s']:.1f} s; overlapped "
              f"{sol['overlapped_s']:.1f} s (measured is "
              f"{wall / sol['overlapped_s']:.1f}x), serial "
              f"{sol['serial_s']:.1f} s ({wall / sol['serial_s']:.1f}x)")
        by_folder = defaultdict(list)
        for record in records:
            by_folder[record["folder"]].append(record)
        print("folder  pages  image tokens  output mean  p50  p90  p99  max"
              "  draft steps saved (4 / 8)  pages with no stored text")
        for folder, rows in sorted(by_folder.items()):
            outputs = [r["output_tokens"] for r in rows]
            saved = []
            for length in DRAFT_LENGTHS:
                plain = drafted = 0
                for r in rows:
                    text = stored[(r["pdf"], r["page"])].replace("\r\n", "\n")
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    plain += r["output_tokens"]
                    drafted += draft_steps(r["token_ids"], ids, length)
                saved.append(1 - drafted / max(plain, 1))
            empty = sum(1 for r in rows
                        if not stored[(r["pdf"], r["page"])].strip())
            print(f"{folder:7} {len(rows):5}  "
                  f"{sum(r['prompt_tokens'] for r in rows) / len(rows):12.0f}  "
                  f"{sum(outputs) / len(outputs):11.0f}  "
                  f"{percentile(outputs, 0.5):4}  {percentile(outputs, 0.9):4}  "
                  f"{percentile(outputs, 0.99):4}  {max(outputs):4}  "
                  f"{saved[0]:12.0%} / {saved[1]:.0%}  {empty:25}")
        if summary.get("profile"):
            profile = summary["profile"]
            print(f"profile: {profile['steps']} steps with {profile['running']} "
                  f"sequences, {profile['wall_s'] * 1e3 / profile['steps']:.1f} ms "
                  f"wall and {profile['gpu_s'] * 1e3 / profile['steps']:.1f} ms "
                  "GPU per step")
            for kernel in summary["kernels"][:20]:
                print(f"  {kernel['gpu_us'] / 1e3:9.1f} ms  {kernel['calls']:6}  "
                      f"{kernel['name'][:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
