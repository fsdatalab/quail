"""Compare every GPU kernel in the B=1024 and B=2048 profiler traces."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import ijson


GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
WRAPPER_NAME = "vllm::dynamic_flashinfer_deepgemm_blockscale_gemm"


def family_for(name: str) -> str:
    lowered = name.lower()
    rules = [
        ("deep_gemm::sm90_fp8_gemm", "DeepGEMM FP8 matrix multiplication"),
        ("per_token_group_quant_8bit", "FP8 activation quantization"),
        ("reshape_and_cache_flash", "KV cache write"),
        ("triton_red_fused_fused_add_rms_norm", "Fused add and RMS normalization"),
        ("triton_poi_fused_mul_silu_slice", "SiLU and gated multiply"),
        ("triton_red_fused_3", "Triton fused reduction 3"),
        ("nvjet_sm90", "Output projection matrix multiplication"),
        ("scaled_fp8_quant_kernel", "Static FP8 quantization"),
        ("triton_poi_fused_add_index_select_mul_rms_norm_split",
         "Fused add, index, and RMS normalization"),
        ("triton_red_fused_1", "Triton fused reduction 1"),
        ("triton_red_fused__to_copy_embedding_rms_norm",
         "Embedding RMS normalization"),
        ("flashattnfwdcombine", "FlashAttention combine"),
        ("flash::enable_sm90", "FlashAttention forward"),
        ("prepare_varlen_num_blocks", "FlashAttention metadata"),
        ("memcpy htod", "Host to device copy"),
        ("memcpy dtoh", "Device to host copy"),
        ("vectorized_elementwise_kernel", "CUDA vectorized elementwise"),
        ("reduce_kernel", "CUDA reduction"),
        ("index_elementwise_kernel", "CUDA index elementwise"),
        ("scatter_gather_elementwise_kernel", "CUDA gather"),
        ("_prepare_prefill_inputs_kernel", "Prepare prefill inputs"),
        ("_post_update_kernel", "Post step update"),
        ("_gather_block_tables_kernel", "Gather block tables"),
        ("_compute_slot_mappings_kernel", "Compute KV slot mappings"),
        ("_get_num_sampled_and_rejected_kernel", "Count sampled tokens"),
        ("_combine_sampled_and_draft_tokens_kernel", "Combine sampled tokens"),
        ("_apply_write_kernel", "Apply scheduler state write"),
        ("_gumbel_sample_kernel", "Gumbel sampling"),
        ("_bias_kernel", "Add output bias"),
    ]
    for needle, family in rules:
        if needle in lowered:
            return family
    return name


def scan_trace(path: Path) -> dict:
    exact = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
    wrapper = {"calls": 0, "total_cpu_us": 0.0}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as trace:
        for event in ijson.items(trace, "traceEvents.item"):
            duration = float(event.get("dur", 0))
            if duration <= 0:
                continue
            name = event.get("name", "")
            category = event.get("cat")
            if category in GPU_CATEGORIES:
                row = exact[name]
                row["calls"] += 1
                row["total_us"] += duration
            elif category == "cpu_op" and name == WRAPPER_NAME:
                wrapper["calls"] += 1
                wrapper["total_cpu_us"] += duration
    return {"exact": exact, "wrapper": wrapper}


def metrics(calls: int, total_us: float, total_kernel_us: float,
            window_s: float) -> dict:
    return {
        "calls": calls,
        "total_ms": round(total_us / 1000, 3),
        "mean_us_per_call": round(total_us / calls, 3) if calls else 0,
        "pct_gpu_kernel_time": round(100 * total_us / total_kernel_us, 4),
        "pct_profile_window": round(100 * total_us / (window_s * 1e6), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-1024", type=Path, required=True)
    parser.add_argument("--trace-2048", type=Path, required=True)
    parser.add_argument("--summary-1024", type=Path, required=True)
    parser.add_argument("--summary-2048", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        1024: json.loads(args.summary_1024.read_text()),
        2048: json.loads(args.summary_2048.read_text()),
    }
    scans = {
        1024: scan_trace(args.trace_1024),
        2048: scan_trace(args.trace_2048),
    }

    exact_names = sorted(set(scans[1024]["exact"]) | set(scans[2048]["exact"]))
    exact_rows = []
    family_totals = {
        1024: defaultdict(lambda: {"calls": 0, "total_us": 0.0, "names": set()}),
        2048: defaultdict(lambda: {"calls": 0, "total_us": 0.0, "names": set()}),
    }
    for name in exact_names:
        row = {"kernel_name": name, "family": family_for(name)}
        for batch_size in (1024, 2048):
            raw = scans[batch_size]["exact"].get(
                name, {"calls": 0, "total_us": 0.0})
            row[f"B{batch_size}"] = metrics(
                raw["calls"], raw["total_us"],
                summaries[batch_size]["total_kernel_us"],
                summaries[batch_size]["window_s"],
            )
            family = family_totals[batch_size][family_for(name)]
            family["calls"] += raw["calls"]
            family["total_us"] += raw["total_us"]
            if raw["calls"]:
                family["names"].add(name)
        exact_rows.append(row)

    families = sorted(
        set(family_totals[1024]) | set(family_totals[2048]),
        key=lambda family: -sum(
            family_totals[batch][family]["total_us"] for batch in (1024, 2048)
        ),
    )
    family_rows = []
    for family in families:
        row = {"family": family}
        for batch_size in (1024, 2048):
            raw = family_totals[batch_size][family]
            row[f"B{batch_size}"] = {
                **metrics(raw["calls"], raw["total_us"],
                          summaries[batch_size]["total_kernel_us"],
                          summaries[batch_size]["window_s"]),
                "exact_kernel_names": sorted(raw["names"]),
            }
        family_rows.append(row)

    wrappers = {}
    for batch_size in (1024, 2048):
        raw = scans[batch_size]["wrapper"]
        wrappers[f"B{batch_size}"] = {
            "calls": raw["calls"],
            "total_cpu_ms_inclusive": round(raw["total_cpu_us"] / 1000, 3),
            "mean_cpu_us_per_call_inclusive": round(
                raw["total_cpu_us"] / raw["calls"], 3),
            "pct_profile_window_inclusive": round(
                100 * raw["total_cpu_us"]
                / (summaries[batch_size]["window_s"] * 1e6), 4),
        }

    payload = {
        "notes": [
            "GPU percentages use summed CUDA kernel durations.",
            "The vLLM dynamic wrapper is a nested CPU operation, not a CUDA kernel.",
            "Wrapper CPU duration is inclusive and must not be added to GPU percentages.",
        ],
        "summaries": summaries,
        "dynamic_fp8_wrapper": wrappers,
        "kernel_families": family_rows,
        "exact_gpu_kernels": exact_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "family_count": len(family_rows),
        "exact_kernel_count": len(exact_rows),
        "top_families": family_rows[:12],
        "dynamic_fp8_wrapper": wrappers,
    }, indent=2))


if __name__ == "__main__":
    main()
