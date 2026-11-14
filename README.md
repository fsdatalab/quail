<p align="center">
  <img src="docs/assets/quail-logo.svg" alt="Quail logo" width="112">
</p>

<h1 align="center">Quail</h1>

<p align="center">
  <b>SQL with AI functions on your GPUs</b><br>
  from the <a href="https://fsdatalab.github.io/">Full Stack Data Lab</a> at CMU
</p>

<p align="center">
  <a href="https://pypi.org/project/quail-engine/"><img src="https://img.shields.io/pypi/v/quail-engine" alt="PyPI"></a>
  <a href="https://fsdatalab.github.io/quail"><img src="https://img.shields.io/badge/docs-fsdatalab.github.io%2Fquail-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Quail runs SQL with AI functions. Apache DataFusion parses, plans, and runs
SQL around model regions. Quail runs the AI calls and related operators
on the GPU in each model region.

[Documentation](https://fsdatalab.github.io/quail) |
[Quickstart](https://fsdatalab.github.io/quail/docs/user-guide/quickstart) |
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## Install

Use Python 3.12. Local AI calls require a supported CUDA GPU.
Install the `quail-engine` package; import it as `quail`.

E.g., install with uv:

```bash
uv pip install quail-engine
```

## Example

E.g., filter reviews on one H100 SXM:

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2"],
    "body": ["The ending was excellent.", "I liked the soundtrack."],
})
config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
with quail.Session(config=config) as session:
    session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    result = session.sql("""
        SELECT r.id
        FROM reviews r
        WHERE AI.IF(PROMPT('Does this review discuss the ending? {0}', r.body))
    """, dialect="bq").collect()
    print(result)
```

## Quail Server

Use [Quail Server](https://fsdatalab.github.io/quail/docs/user-guide/server)
to run queries on a remote GPU. Pass the server address as `endpoint`
when you create the session. The server keeps accepted queries running
after the client disconnects.

## Supported functions

| AI function | Answer |
| --- | --- |
| `AI.IF` | Boolean, for filters and joins |
| `AI.SCORE` | Numeric score |
| `AI.CLASSIFY` | Label |
| `AI.EXTRACT` | Text from a document |
| `AI.EXTRACT_SPAN` | Character positions of the extracted text |
| `AI.MAP` | Generated text |
| `AI.SAE_FEATURES` | Sparse feature vector |

See the [SQL reference](https://fsdatalab.github.io/quail/docs/user-guide/sql).

## Supported models and GPUs

See [Supported models and GPUs](https://fsdatalab.github.io/quail/docs/user-guide/models)
for model names and GPU requirements. AI function support depends on the model.

## Development

Follow the [repository setup and checks](https://fsdatalab.github.io/quail/docs/contributing).

## Contributing

[Report a bug](https://github.com/fsdatalab/quail/issues) or follow the
[contributing guide](https://fsdatalab.github.io/quail/docs/contributing)
to submit a change.
