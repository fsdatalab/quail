# calibrate: GPU rates for the Quail cost model

The `calibrate` package measures three rates of a GPU:

- the fp8 GEMM rate, with [`gemm.py`](gemm.py);
- the attention rate, with [`attention.py`](attention.py);
- the memory bandwidth, with [`attention.py`](attention.py).

A GEMM is a matrix multiply. The Quail cost model uses these three
rates to estimate the time of a query.

The package uses vendor libraries (cuBLAS, FlashAttention 3, and
FlashInfer). It does not use the Quail kernels. Changes to the Quail
kernels do not change the measured rates.

[`to_spec.py`](to_spec.py) changes the results into the fields of
`quail/specs/<device>.py`. This package is not in the `quail` wheel.

In this document, a "measurement" is one execution of `gemm.py` or
`attention.py`.

## GEMM rate: gemm.py

[`gemm.py`](gemm.py) finds the highest fp8 GEMM rate that the GPU
holds at full power. The cost model uses this rate as
`sustained_fp8_flops`.

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

The cost model uses MSMF. A query holds the GPU at full power for
seconds or minutes.

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

[`modal_app.py`](modal_app.py) does the same measurement on a Modal
GPU. It uses the NGC image `nvcr.io/nvidia/pytorch:26.04-py3`. The
published H100 SXM rates are from this image.

1. Run these commands:

   ```bash
   mkdir -p results/benchmark
   log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
   uv run modal run --detach calibrate/modal_app.py::gemm \
       --device h100-sxm --prediction "..." 2>&1 | tee "$log"
   ```

2. Find the results on the `quail-results` volume, in `calibrate/`.

### Output files of gemm.py

`--root` is the results directory. On Modal, the results directory is
`calibrate/` on the `quail-results` volume.

| File | Contents |
|---|---|
| `<root>/<device>/gemm/<stamp>/<name>.txt` | The log of one mamf-finder call |
| `<root>/<device>/gemm/<stamp>/summary.json` | The summary of the measurement, written after each call |
| `<root>/<device>/gemm.json` | A copy of the summary of the last measurement |

Each measurement replaces `gemm.json`. The `<stamp>` directories of
earlier measurements do not change.

### GEMM results on H100 SXM

On Modal, you cannot select the physical GPU. Each measurement can use
a different H100. These measurements used the `published_fp8` shapes
and the same software:

| Date | MSMF, TFLOPS | MAMF, TFLOPS | Results directory |
|---|---|---|---|
| 2026-10-12 | 1,352 | 1,543 | `/results/ablations/gemm_ceilings/20261012T222632Z/` |
| 2026-10-13 | 1,429 | 1,629 | `/results/calibrate/h100-sxm/gemm/20261013T221653Z/` |
| 2026-10-13 | 1,315 | 1,500 | `/results/calibrate/h100-sxm/gemm/20261013T234156Z/` |

The highest MSMF is 8.7% more than the lowest MSMF. In the published
results, the difference between GPUs is 2.7%.

`quail/specs/h100_sxm.py` uses 1,352 TFLOPS. The second GPU is faster
than this rate. On that GPU, the speed-of-light (SoL) estimate can be
longer than the measured query time. Then the estimate is not a lower
bound. Measurements on more GPUs are necessary to find the correct
rate.

## Attention rate and memory bandwidth: attention.py

[`attention.py`](attention.py) measures FlashAttention 3 and FlashInfer
in bf16, for one head layout. A head layout is three numbers: the
query heads, the KV heads, and the head dimension. The device
specification has one table of attention rates for each head layout.

`attention.py` measures three types of point:

| Type | Work |
|---|---|
| `packed` | Causal attention on documents of equal length |
| `cached` | Query rows that read cached KV in pages |
| `copy` | One 4 GB copy on the GPU, for the memory bandwidth |

`attention.py` measures each point for 2 s at full power. The 2 s window and its limits
are the same as for MSMF. [`timing.py`](timing.py) has this code.

### Measure the attention rate on a local GPU

Do this procedure on a computer that has the GPU, FlashAttention 3, and
FlashInfer.

1. Run this command:

   ```bash
   uv run python -m calibrate.attention --device h100-sxm --heads 32,8,128 \
       --root results/calibrate
   ```

2. For a short test, add `--smoke`. The test measures one point of each
   type for 0.5 s. It also compares the two libraries with PyTorch.

### Measure the attention rate on Modal

1. Run this command:

   ```bash
   uv run modal run --detach calibrate/modal_app.py::attention \
       --device h100-sxm --heads 32,8,128
   ```

2. For a short test, add `--smoke`.

## Make the device specification: to_spec.py

[`to_spec.py`](to_spec.py) reads the GEMM summary and the attention
summary. It writes the fields for `quail/specs/<device>.py`:

| Field | Value |
|---|---|
| `sustained_fp8_flops` | The highest MSMF of all the mamf-finder calls |
| `sustained_hbm_bw` | The bandwidth of the copy |
| `attention_rates` | The faster library at each point |

`to_spec.py` also writes each point that was not steady.

1. If the summaries are on Modal, copy them from the volume:

   ```bash
   modal volume get quail-results calibrate/h100-sxm/gemm.json .
   modal volume get quail-results calibrate/h100-sxm/attention_32_8_128.json .
   ```

2. Run `to_spec.py` with the two summaries:

   ```bash
   uv run python -m calibrate.to_spec gemm.json attention_32_8_128.json
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
