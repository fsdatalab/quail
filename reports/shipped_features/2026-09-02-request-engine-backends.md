# vLLM and SGLang model backends

Stock vLLM, pipelined vLLM, and pipelined SGLang now implement the same model
backend interface as Quail. Each backend plans its own model node, starts its
own engine, clears its own KV between queries, and returns the same Arrow
answer relations.

The QUAIL-B runner now builds every query once through `quail.bench.quailb`.
It selects the method with `EngineConfig.backend`. It no longer rebuilds the
query in the stock vLLM runner.

The request backends report cached tokens, fresh tokens, KV regret, request
counts, query time, boot time, and KV capacity. Stock vLLM and pipelined vLLM
share one loaded vLLM model inside a query family container. Quail uses a
separate container so its loaded weights and KV do not remain when vLLM
starts. SGLang uses another Modal image and container so the vLLM and SGLang
Python packages do not change each other.

The CPU suite checks backend registration, physical planning, plan encoding,
filter answers, join answers, KV regret, and complete physical request
execution. The benchmark evaluator scores the Arrow answer relations from all
four backends.

The stock vLLM and SGLang paths were also checked through the public
`ModalComputeProvider.execute(QueryRequest)` interface. Both returned Arrow
query results without using a benchmark-specific execution path.

LEP-1 confirmed all four paths on one H100 per backend. The measured query
times were 1.27 seconds for Quail, 1.50 seconds for stock vLLM, 1.40 seconds
for pipelined vLLM, and 1.68 seconds for pipelined SGLang. See
`reports/2026-09-02-request-backend-confirmation.md`.
