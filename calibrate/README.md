# calibrate

The `calibrate` package measures how fast a GPU does the work that
Quail's cost model prices. It measures three rates:

- the fp8 matrix multiply (GEMM) rate, in [`gemm.py`](gemm.py);
- the attention rate, in [`attention.py`](attention.py);
- the memory bandwidth, also in [`attention.py`](attention.py).

The package uses vendor libraries, not Quail's kernels. Thus the
measured rates do not change when Quail's code changes.

The results go into the measured fields of `quail/specs/<device>.py`.
[`to_spec.py`](to_spec.py) prints these fields from the saved results.

This package is not part of the `quail` wheel.

## The GEMM measurement

[`gemm.py`](gemm.py) measures the fastest fp8 GEMM rate that the GPU
can keep at full power. The cost model uses this rate as
`sustained_fp8_flops`.

### The tool

`gemm.py` runs
[mamf-finder](https://github.com/stas00/ml-engineering/blob/30ddd97054eb01035c0a5079a58f011df603b525/compute/accelerator/benchmarks/mamf-finder.py)
by Stas Bekman. `gemm.py` does not change mamf-finder.

- `gemm.py` uses one fixed version of mamf-finder, at commit
  `30ddd970`.
- `gemm.py` downloads the file and checks its sha256 before it runs it.
  If the sha256 is not correct, `gemm.py` stops.
- mamf-finder calls `torch._scaled_mm` with one fp8 scale for each
  tensor. PyTorch sends this call to cuBLAS.

### The two rates

mamf-finder gives two rates for each run:

- **MAMF** (maximum achievable matmul FLOPS) is the fastest rate in a
  short burst. The GPU starts each burst cool and at its top clock.
- **MSMF** (maximum sustainable matmul FLOPS) is the fastest rate that
  the GPU keeps for 2 s at full power. The GPU decreases its clock to
  stay at its power limit, so MSMF is lower than MAMF.

Quail uses MSMF. A query keeps the GPU at full power for seconds or
minutes, not for one short burst.

For example, on an H100 SXM, the vendor's fp8 specification is 1,979
TFLOPS. The published MAMF is 1,550 TFLOPS. The published MSMF is
1,358 TFLOPS.

### How mamf-finder measures one shape

A shape is the three sizes M, N, and K of one GEMM. For each shape,
mamf-finder does these steps:

1. For MAMF, it runs 10 bursts of 20 GEMMs. Before each burst, the GPU
   stays idle for 0.25 s.
2. It keeps only the GEMMs that ran at a clock of at least 0.97 times
   the top clock.
3. It takes the median of the 5 fastest GEMMs that it kept.
4. For MSMF, it runs the GEMM until the clock stops decreasing.
5. It then times the GEMM for 2 s, in 8 equal parts.
6. It accepts the 2 s window only if two conditions are true:
   - the second half is no more than 1% slower than the first half;
   - the parts are within 10% of each other.

The MAMF or MSMF of the GPU is the best rate of all the shapes.

### The runs

`gemm.py` does these runs on the same GPU:

| Run | Shapes | When |
|---|---|---|
| `published_fp8` | The published MAMF and MSMF shapes of the GPU | The GPU has published rates in [`targets.py`](targets.py) |
| `search` | The shape search of mamf-finder | The GPU has no published rates |
| `qkv`, `o`, `gate_up`, `down` | The four projections of one model, with M from 16 to 131,072 tokens | You give `--model` |

The `published_fp8` run lets you compare the GPU with the published
rates. The projection runs are optional. The cost model does not need
them.

### Run the measurement

To run on a computer with the GPU, use this command:

```bash
uv run python -m calibrate.gemm --device h100-sxm --root results/calibrate
```

To run on Modal, use [`modal_app.py`](modal_app.py). Write the output
to a log file:

```bash
mkdir -p results/benchmark
log="results/benchmark/$(date -u +%Y%m%dT%H%M%SZ)-calibrate-gemm.log"
uv run modal run --detach calibrate/modal_app.py::gemm \
    --device h100-sxm --prediction "..." 2>&1 | tee "$log"
```

The Modal run uses the NGC image `nvcr.io/nvidia/pytorch:26.04-py3`.
The published H100 SXM rates come from this image. The Modal run
writes to the `quail-results` volume, under `calibrate/`.

To also sweep the projections of a model, add `--model`, for example
`--model qwen3-4b-fp8`.

### The output

`--root` is the results directory. The run writes these files:

- `<root>/<device>/gemm/<stamp>/<run>.txt`: the log of each
  mamf-finder run;
- `<root>/<device>/gemm/<stamp>/summary.json`: the summary, updated
  after each run;
- `<root>/<device>/gemm.json`: a copy of the last summary.

Each new run replaces `gemm.json`. The logs of earlier runs stay in
their own `<stamp>` directories.

### Results on H100 SXM

Each Modal run can get a different physical H100. These runs used the
`published_fp8` shapes:

| Run | MSMF, TFLOPS | MAMF, TFLOPS | Results |
|---|---|---|---|
| 2026-10-12 | 1,352 | 1,543 | `/results/ablations/gemm_ceilings/20261012T222632Z/` |
| 2026-10-13, first | 1,429 | 1,629 | `/results/calibrate/h100-sxm/gemm/20261013T221653Z/` |
| 2026-10-13, second | 1,315 | 1,500 | `/results/calibrate/h100-sxm/gemm/20261013T234156Z/` |

The fastest MSMF is 8.7% higher than the slowest. The published rates
show a difference of 2.7% between GPUs. The software was the same in
all three runs.

`quail/specs/h100_sxm.py` uses 1,352 TFLOPS. This rate is lower than
the rate of the fastest GPU above. On that GPU, the speed of light
(SoL) estimate can be slower than the real query. More runs on more
GPUs will show the highest rate to use.

## The attention measurement

[`attention.py`](attention.py) times FlashAttention 3 and FlashInfer
in bf16, at one head layout. A head layout is the number of query
heads, the number of KV heads, and the head dimension. The device
specification keeps one table of attention rates for each head layout.

The measurement has three kinds of point:

- `packed`: causal attention on documents of equal length;
- `cached`: query rows that read cached KV in pages;
- `copy`: one 4 GB copy on the GPU, for the memory bandwidth.

The measurement times each point for 2 s at full power. It uses the
same 2 s window and the same limits as MSMF.
[`timing.py`](timing.py) contains this timing code.

To run on a computer with the GPU, FlashAttention 3, and FlashInfer,
use this command:

```bash
uv run python -m calibrate.attention --device h100-sxm --heads 32,8,128 \
    --root results/calibrate
```

To run on Modal, use this command:

```bash
uv run modal run --detach calibrate/modal_app.py::attention \
    --device h100-sxm --heads 32,8,128
```

Add `--smoke` for a short check. The smoke run measures one point of
each kind for 0.5 s. It also compares both libraries with PyTorch.

## Make the device specification

[`to_spec.py`](to_spec.py) reads the GEMM summary and the attention
summary. It prints the fields for `quail/specs/<device>.py`:

```bash
uv run python -m calibrate.to_spec \
    results/calibrate/h100-sxm/gemm.json \
    results/calibrate/h100-sxm/attention_32_8_128.json
```

- `sustained_fp8_flops` is the best MSMF of all the GEMM runs.
- `sustained_hbm_bw` is the bandwidth of the copy.
- The attention table uses the faster library at each point.

`to_spec.py` also prints each point that was not steady.

After a Modal run, first copy the summaries from the volume:

```bash
modal volume get quail-results calibrate/h100-sxm/gemm.json .
modal volume get quail-results calibrate/h100-sxm/attention_32_8_128.json .
```

## Add a GPU

Add a `Target` to [`targets.py`](targets.py). The `Target` gives:

- the device name in `quail.specs`;
- the Modal GPU type;
- the FlashInfer architecture;
- the published fp8 MAMF and MSMF, if they are available.

If the GPU has no published rates, `gemm.py` runs the shape search of
mamf-finder.
