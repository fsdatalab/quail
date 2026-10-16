# calibrate: GPU GEMM rate for the Quail cost model

The `calibrate` package measures the fp8 GEMM rate of a GPU with
[`gemm.py`](gemm.py). A GEMM is a matrix multiply. The Quail cost model
uses this rate for the dense projections. It uses the specification
rates for attention and memory bandwidth.

The package uses a vendor library (cuBLAS). It does not use the Quail
kernels. Changes to the Quail kernels do not change the measured rate.

[`to_spec.py`](to_spec.py) changes the results into the
`sustained_fp8_flops` field of `quail/specs/<device>.py`. This package
is not in the `quail` wheel.

In this document, a "measurement" is one execution of `gemm.py`.

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

`gemm.py` calls mamf-finder one time. The call has a name:

| Name | Shapes | Condition |
|---|---|---|
| `published_fp8` | The published MAMF shape and MSMF shape of the GPU | The GPU has published rates in [`targets.py`](targets.py) |
| `search` | The shapes that mamf-finder selects | The GPU has no published rates |

Use the `published_fp8` result to compare the GPU with the published
rates.

### Measure the GEMM rate on a local GPU

Run this command on a computer that has the GPU:

```bash
uv run python -m calibrate.gemm --device h100-sxm --root results/calibrate
```

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

## Make the device specification: to_spec.py

[`to_spec.py`](to_spec.py) reads the GEMM summaries of all the GPUs.
It writes `sustained_fp8_flops` as the highest MAMF of all the
mamf-finder calls. For this reason, the SoL estimate stays a lower
bound on each GPU of the type.

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
   - the published fp8 MAMF and MSMF, if they are available.
2. If the GPU has no published rates, set `published_fp8` to `None`.
   Then `gemm.py` uses the shapes that mamf-finder selects.
