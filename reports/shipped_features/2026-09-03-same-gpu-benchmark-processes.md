# Same GPU benchmark processes

The QUAIL-B runner now runs Quail, stock vLLM, and pipelined vLLM on the same
physical H100 for each query family. One Modal function keeps the H100 while
two process groups run in order.

The first group runs Quail. The parent stops the complete group after it saves
the results. The second group loads vLLM once, then runs stock vLLM and
pipelined vLLM with the same model. Stopping the complete group also stops
vLLM's engine process.

The runner records the GPU UUID seen by each group. It stops the family run if
the groups do not see exactly one matching UUID. It also waits for GPU memory
use to fall below 1 GiB before starting the next group.

SGLang remains in a separate Modal container. vLLM 0.26.0 and SGLang 0.5.18
require different exact versions of `apache-tvm-ffi`, so they cannot be
installed together without overriding one backend's declared dependencies.

LEP-1 confirmed the new runner. Both groups used
`GPU-a4f6d03a-f439-f748-6bc2-2c4da514482c`, and each left 4 MiB in use after
exit. See `reports/2026-09-03-same-gpu-benchmark-processes.md`.
