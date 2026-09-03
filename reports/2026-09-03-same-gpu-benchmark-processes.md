# Same physical GPU benchmark confirmation

## Setup

The confirmation ran LEP-1 at scale factor 0.1 with Qwen3 4B fp8. One Modal
function received one H100. It ran Quail in one process group, followed by
stock vLLM and pipelined vLLM in a second process group. SGLang was excluded
because SGLang 0.5.18 and vLLM 0.26.0 require different exact versions of
`apache-tvm-ffi`.

The family function call was `fc-01M1JVQKES2W4PXXDZC3D904HG`. The saved data
is in the following files:

- `/results/benchmarks/quailb/family-runs/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json`
- `/results/benchmarks/quailb/runs/qb_20260903T052637Z_00a04205/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-quail.json`
- `/results/benchmarks/quailb/runs/qb_20260903T052637Z_237572db/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-stock_vllm.json`
- `/results/benchmarks/quailb/runs/qb_20260903T052637Z_b5c185c3/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families-pipelined_vllm.json`

## Prediction

All three configurations should finish. Both process groups should report one
matching physical GPU UUID. GPU memory should fall below 1 GiB after each
group stops. Stock vLLM should report a cold model load, while pipelined vLLM
should reuse that model and report a warm start.

## Result

The prediction was correct. All three configurations finished with no failed
query. Both process groups reported
`GPU-a4f6d03a-f439-f748-6bc2-2c4da514482c`. GPU memory use was 4 MiB after
each group stopped.

Stock vLLM loaded the model in 52.29 seconds. Pipelined vLLM reported a warm
start with 0 seconds of boot time, so the two vLLM configurations shared one
loaded model.

| Configuration | Query time (s) | Documents/s | GPU cost ($/query) |
| --- | ---: | ---: | ---: |
| Quail | 1.10 | 454.55 | 0.001207 |
| Stock vLLM, stage-major waves | 1.48 | 337.84 | 0.001624 |
| Pipelined vLLM | 1.40 | 357.14 | 0.001536 |

The query time and cost exclude model startup. Quail had a 24.91 second cold
start. Stock vLLM had a 52.29 second cold start. Pipelined vLLM reused the
loaded vLLM model.

Figure: plots/same_gpu_benchmark_processes.png

The previous manual cleanup path left 57.9 GB allocated after Quail. The new
runner does not depend on Python references becoming unreachable. It stops
the complete process group and checks GPU memory before starting vLLM.
