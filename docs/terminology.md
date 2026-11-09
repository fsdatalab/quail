# Terminology

Use one term for each concept in every page. Do not use the terms in
the "Not" column.

| Term | Meaning | Not |
| --- | --- | --- |
| Quail | The system. The Python package is `quail`. | the engine, the framework |
| session | A `quail.Session`: a model, a device, and the tables registered on it. | context, connection |
| table | A table registered on a session, such as `reviews`. | relation, dataset |
| document | The text in one row that an AI function reads. | doc, record, item |
| AI function | `AI.IF`, `AI.SCORE`, `AI.CLASSIFY`, `AI.EXTRACT`, `AI.EXTRACT_SPAN`, `AI.MAP`, `AI.SAE_FEATURES`. | semantic operator, LLM function, UDF |
| AI call | One use of an AI function in a query. | invocation |
| prompt | The text a `PROMPT(...)` builds for the model, with `{0}`, `{1}` for documents. | template (except for the literal text) |
| answer | What the model returns for one document or pair: a label, a score, an extracted value, a map text, or features. | output, response, result (for one document) |
| label | The category `AI.CLASSIFY` returns. | class, category value |
| span | The `start` and `end` character positions of an extracted answer. | offset, range |
| filter | An `AI.IF` in `WHERE`, or a test of an answer. | predicate (in user pages) |
| join | An `AI.IF` in `ON` that reads two tables. | semantic join, pair filter |
| model region | The part of a query Quail runs on the GPU: AI calls and the operators that run with them. DataFusion runs the rest. | region block, GPU part |
| DataFusion | Apache DataFusion, which parses, plans, and runs SQL around model regions. | the SQL engine |
| optimizer rule | A rewrite of the logical plan, named R1 to R17. | pass, transformation |
| logical plan | DataFusion's plan of the query, with `AiNode`s for AI calls. | logical graph |
| physical plan | Quail's plan of one model region: nodes, order, and GPU settings. | execution plan (for Quail's) |
| planner | The code that chooses a physical plan. | optimizer (for physical choices) |
| executor | The code that runs a physical plan on the GPU. | runtime, worker |
| backend | `"quail"`, or a stock vLLM baseline. | engine |
| KV | The attention key-value cache. | KV cache, kv-cache |
| prefix sharing, pipelining, token-based admission, KV rewind | The execution mechanisms, by these names only. | |
| selectivity | The fraction of documents or pairs expected to pass a filter. | pass rate, hit rate |
| `HINT` | The SQL function that gives a filter's selectivity. | |
| GPU | A device; `gpus` is the count. | device count, workers |
