# calibrate: GPU rates for the Quail cost model

The `calibrate` package measures three rates of a GPU:

- the fp8 GEMM rate, with [`gemm.py`](gemm.py);
- the attention rate, with [`attention.py`](attention.py);
- the memory bandwidth, with [`attention.py`](attention.py).

A GEMM is a matrix multiply. The Quail cost model uses only the GEMM
rate. It uses the specification rates for attention and memory
bandwidth. The attention rates change with the call size, the library,
and the input data, so we do not use them yet.

The package uses vendor libraries (cuBLAS, FlashAttention 3, and
FlashInfer). It does not use the Quail kernels. Changes to the Quail
kernels do not change the measured rates.

[`to_spec.py`](to_spec.py) changes the results into the fields of
`quail/specs/<device>.py`. This package is not in the `quail` wheel.

In this document, a "measurement" is one execution of `gemm.py` or
`attention.py`.

## GEMM rate: gemm.py

[`gemm.py`](gemm.py) finds the highest fp8 GEMM rates of the GPU. The
cost model uses the MAMF as `sustained_fp8_flops`.

### Software that gemm.py uses

`gemm.py` uses
[mamf-finder](https://github.com/stas00/ml-engineering/blob/30ddd97054eb01035c0a5079a58f011df603b525/compute/accelerator/benchmarks/mamf-finder.py),
a script by Stas Bekman:

- `gemm.py` uses mamf-finder at commit `30ddd970` only.
- `gemm.py` downloads mamf-finder and examines its sha256. If the
  sha256 is incorrect, `gemm.py` stops.
- `gemm.py` does not change mamf-finder.

mamf-finder calls `torch._scaled_mm` with one fp8 scale for each
matrix. PyTorch sends this call to cuBLAS.

### MAMF and MSMF

mamf-finder gives two rates:

- **MAMF** (maximum achievable matmul FLOPS) is the highest rate in a
  short burst. At the start of a burst, the GPU is cool and its clock
  is at the maximum.
- **MSMF** (maximum sustainable matmul FLOPS) is the highest rate that
  the GPU holds for 2 s at full power.

At full power, the GPU decreases its clock to stay below its power
limit. For this reason, MSMF is lower than MAMF.

The cost model uses MAMF. The SoL estimate must be a lower bound on
the query time, so it uses the highest rate that a GEMM gets.

For example, for the H100 SXM:

| Rate | TFLOPS |
|---|---|
| Vendor specification, fp8 | 1,979 |
| Published MAMF | 1,550 |
| Published MSMF | 1,358 |

### Procedure that mamf-finder uses for one shape

A GEMM multiplies an M × K matrix by a K × N matrix. The shape of the
GEMM is M, N, and K. mamf-finder does these steps for each shape.

To find the MAMF of the shape:

1. Run 10 bursts of 20 GEMMs. Before each burst, do not use the GPU
   for 0.25 s.
2. Discard each GEMM with a clock below 0.97 times the maximum clock.
3. Find the 5 fastest GEMMs that remain.
4. Use the median rate of these 5 GEMMs.

To find the MSMF of the shape:

1. Run the GEMM until the clock does not decrease.
2. Measure the time of the GEMMs for 2 s. Divide the 2 s into 8
   equal parts.
3. Make sure that the result is steady. The result is steady if these
   two conditions are true:
   - The second half is not more than 1% slower than the first half.
   - The difference between the fastest part and the slowest part is
     not more than 10% of the median part.

The MAMF of the GPU is the highest MAMF of all the shapes. The MSMF
of the GPU is the highest MSMF of all the shapes.

### Shapes that gemm.py measures

`gemm.py` calls mamf-finder one or more times on the same GPU. Each
call has a name:

| Name | Shapes | Condition |
|---|---|---|
| `published_fp8` | The published MAMF shape and MSMF shape of the GPU | The GPU has published rates in [`targets.py`](targets.py) |
| `search` | The shapes that mamf-finder selects | The GPU has no published rates |
| `qkv`, `o`, `gate_up`, `down` | The four projections of one model; M is 16 to 131,072 tokens | You use `--model` |

Use the `published_fp8` result to compare the GPU with the published
rates. The cost model does not use the projection results.

### Measure the GEMM rate on a local GPU

Do this procedure on a computer that has the GPU.

1. Run this command:

   ```bash
   uv run python -m calibrate.gemm --device h100-sxm --root results/calibrate
   ```

2. To also measure the projections of a model, add
   `--model qwen3-4b-fp8`.

### Measure the GEMM rate on Modal

[`modal_app.py`](modal_app.py) does the same measurement on Modal GPUs.
It uses the NGC image `nvcr.io/nvidia/pytorch:26.04-py3`. The
published H100 SXM rates are from this image.

On Modal, you cannot select the physical GPU, and the GEMM rate is
different on different GPUs. For this reason, measure on more than one GPU.
`--gpus 5` starts 5 measurements at the same time. Each measurement is
on its own GPU.

1. Run these commands:

   ```bash
   mkdir -p results/benchmark
   log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
   uv run modal run --detach calibrate/modal_app.py::gemm \
       --device h100-sxm --gpus 5 --prediction "..." 2>&1 | tee "$log"
   ```

2. Find the results on the `quail-results` volume, in `calibrate/`.

### Output files of gemm.py

`--root` is the results directory. On Modal, the results directory is
`calibrate/` on the `quail-results` volume.

| File | Contents |
|---|---|
| `<root>/<device>/gemm/<stamp>/<name>.txt` | The log of one mamf-finder call |
| `<root>/<device>/gemm/<stamp>/summary.json` | The summary of the measurement, with the GPU UUID, written after each call |

### GEMM results on H100 SXM

These measurements used the `published_fp8` shapes and the same
software. Each row is a different physical H100.

| Results directory under `/results/` | MSMF, TFLOPS | MAMF, TFLOPS |
|---|---|---|
| `ablations/gemm_ceilings/20261012T222632Z/` | 1,352 | 1,543 |
| `calibrate/h100-sxm/gemm/20261013T221653Z/` | 1,429 | 1,629 |
| `calibrate/h100-sxm/gemm/20261013T234156Z/` | 1,315 | 1,500 |
| `calibrate/h100-sxm/gemm/20261014T173306Z-g0/` | 1,342 | 1,535 |
| `calibrate/h100-sxm/gemm/20261014T173306Z-g1/` | 1,335 | 1,541 |
| `calibrate/h100-sxm/gemm/20261014T173306Z-g2/` | 1,431 | 1,612 |
| `calibrate/h100-sxm/gemm/20261014T173306Z-g3/` | 1,349 | 1,535 |
| `calibrate/h100-sxm/gemm/20261014T173306Z-g4/` | 1,357 | 1,564 |

The published MSMF is 1,358 TFLOPS. Six of the eight GPUs are between
1,315 and 1,357 TFLOPS. Two GPUs are near 1,430 TFLOPS. The NVIDIA
driver version does not show the difference.

The MAMF is between 1,500 and 1,629 TFLOPS.
`quail/specs/h100_sxm.py` uses the highest MAMF, 1,629 TFLOPS. With
this rate, the SoL estimate is a lower bound on each GPU in the table.

## Attention rate and memory bandwidth: attention.py

[`attention.py`](attention.py) measures FlashAttention 3 and FlashInfer
in bf16, for one head layout. A head layout is three numbers: the
query heads, the KV heads, and the head dimension. `to_spec.py
--attention` prints one table of attention rates for each head layout.
The device specifications do not use these tables yet.

### Calls that attention.py measures

One call is the attention of one layer. Each call has the size of the
calls that Quail makes:

| Type | Call |
|---|---|
| `packed` | Causal attention on one chunk of tokens. The chunk is divided into documents of one length. There is one point for each length, from 128 to 32,768 tokens. |
| `cached` | Sequences whose query rows read KV that is in the cache. The query rows fill one chunk. The number of sequences is not more than the KV arena holds. There is one point for each (query rows, context) pair: 16 to 1,024 rows, 1,024 to 32,768 tokens of context. |
| `size` | One `packed` point and one `cached` point, at 1/4 and 1/2 of the call size. These points show how the rate changes with the call size. |
| `copy` | One 4 GB copy on the GPU, for the memory bandwidth |

The chunk and the KV arena come from the Quail planner, for the models
with the head layout ([`targets.py`](targets.py), `call_sizes`). If two
models have the same layout, `attention.py` uses the larger chunk and
the larger arena. A larger call does not have a lower rate, so the rate
stays an upper bound for both models.

| Head layout | Models | Chunk, tokens | KV arena, tokens |
|---|---|---|---|
| 16, 8, 128 | Qwen3 Reranker 0.6B, Decision-2.0-Kai 0.6B | 65,536 | 614,720 |
| 32, 8, 128 | Qwen3 4B, Qwen3 Reranker 4B | 110,376 | 387,952 |
| 64, 8, 128 | Qwen3 32B | 41,943 | 106,368 |

DiffusionGemma 26B-A4B has sliding-window layers. The cost model does
not use measured attention rates for it, so `attention.py` does not
measure its layout.

The inputs are random bf16 values. The time of a call is a function of
the shapes, not of the values.

### Procedure that attention.py uses for one point

[`timing.py`](timing.py) has this code.

1. Make the inputs.
2. Call each library one time. This plans the call and compiles the
   kernel, before the measurement starts.
3. Do steps 3.1 to 3.4 for each library. Alternate which library is
   first, from one point to the next.
   1. Run the call until the clock does not decrease: 3 readings in a
      row, 0.25 s apart, are not more than 5 MHz below the previous
      reading. Run the call for 1 s minimum and 20 s maximum.
   2. Run the call for 2 s. Record the time of each call with GPU
      events. Before each call, write zeros to a 256 MB buffer. This
      removes the data from the L2 cache. Do not record the time of
      this step.
   3. Examine the 2 s window as for MSMF. The window is steady if the
      second half is not more than 1% slower than the first half, and
      (fastest part − slowest part) ÷ median part of 8 parts is not
      more than 10%.
   4. Do steps 3.1 to 3.3 again until 3 windows are steady. Stop after
      6 windows.
4. The rate of the library is the median rate of its steady windows.
   If fewer than 3 windows are steady, the rate is not steady.

### Measure the attention rate on a local GPU

Do this procedure on a computer that has the GPU, FlashAttention 3, and
FlashInfer.

1. Run this command:

   ```bash
   uv run python -m calibrate.attention --device h100-sxm --heads 32,8,128 \
       --root results/calibrate
   ```

2. For a short test, add `--smoke`. The test measures one point of each
   type in one 0.5 s window. It also compares the two libraries with
   PyTorch.

### Measure the attention rate on Modal

`--heads all` measures each head layout in the table above. `--gpus 3`
measures each layout on 3 GPUs at the same time.

1. Run these commands:

   ```bash
   mkdir -p results/benchmark
   log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-attention.log"
   uv run modal run --detach calibrate/modal_app.py::attention \
       --device h100-sxm --heads all --gpus 3 --prediction "..." 2>&1 | tee "$log"
   ```

2. For a short test, add `--smoke`.

The results go to `<root>/<device>/attention/<stamp>/summary.json`.

## Make the device specification: to_spec.py

[`to_spec.py`](to_spec.py) reads the GEMM summaries and the attention
summaries of all the GPUs. It uses the highest rate of all the GPUs.
For this reason, the SoL estimate stays a lower bound on each GPU of
the type.

| Output | Value | In the device specification |
|---|---|---|
| `sustained_fp8_flops` | The highest MAMF of all the mamf-finder calls | Yes |
| `sustained_hbm_bw` | The highest steady bandwidth of the copy, with `--attention` | No |
| `attention_rates` | One table for each head layout, with `--attention`: at each point, the highest steady rate of the faster library | No |

`to_spec.py` writes the outputs that are not in the specification as
comments. If a point has no steady rate on any GPU, `to_spec.py
--attention` stops with an error. Measure that point again. `to_spec.py` does not use smoke
results or the `size` points. It writes the `size` points as comments.

1. If the summaries are on Modal, copy them from the volume:

   ```bash
   modal volume get quail-results calibrate/h100-sxm .
   ```

2. Run `to_spec.py` with the directory:

   ```bash
   uv run python -m calibrate.to_spec h100-sxm
   ```

3. Copy the output into `quail/specs/<device>.py`.

## Add a GPU

1. Add a `Target` to [`targets.py`](targets.py). The `Target` has:
   - the device name in `quail.specs`;
   - the Modal GPU type;
   - the FlashInfer architecture;
   - the published fp8 MAMF and MSMF, if they are available.
2. If the GPU has no published rates, set `published_fp8` to `None`.
   Then `gemm.py` uses the shapes that mamf-finder selects.
