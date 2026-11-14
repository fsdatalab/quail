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

Quail runs SQL with AI functions on your GPUs. An AI function asks a
model about the text in a row and returns the answer as a SQL value.
Apache DataFusion runs the SQL, and Quail runs the AI functions on the
GPU.

[Documentation](https://fsdatalab.github.io/quail) |
[Quickstart](https://fsdatalab.github.io/quail/docs/user-guide/quickstart) |
[QUAIL-B](https://github.com/fsdatalab/quail-bench)

## Install

Quail needs Python 3.12 and a
[supported CUDA GPU](https://fsdatalab.github.io/quail/docs/user-guide/models).

```bash
uv pip install quail-engine
```

The package is `quail-engine`, and you import it as `quail`. The code on
`main` can be ahead of the last release;
[install from source](https://fsdatalab.github.io/quail/docs/contributing)
to run it.

## Example

The query below keeps the reviews that discuss the ending, and labels
each one as praise or a complaint. It runs Qwen3 4B on one H100:

```python
import pyarrow as pa
import quail

reviews = pa.table({
    "id": ["r1", "r2", "r3"],
    "body": [
        "The ending was perfect and the cast was great.",
        "Too long, and the ending made no sense.",
        "Great music, but I want my money back.",
    ],
})
config = quail.EngineConfig(model="qwen3-4b-fp8", device="h100-sxm")
with quail.Session(config=config) as session:
    session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
    result = session.sql("""
        SELECT r.id, AI.CLASSIFY(r.body, ['praise', 'complaint']) AS kind
        FROM reviews r
        WHERE AI.IF(PROMPT('Does the review discuss the ending? {0}', r.body))
    """).collect()
    print(result)
```

`AI.IF` keeps a review when the model answers yes, and `AI.CLASSIFY`
returns one of the two labels. Quail reads each review once for both
functions, and classifies only the reviews that `AI.IF` keeps.

## AI functions

| Function | Returns |
| --- | --- |
| `AI.IF` | `TRUE` when the model answers yes; use it in a filter or a join |
| `AI.SCORE` | A number from 0 to 1; a higher number is a stronger yes |
| `AI.CLASSIFY` | The one label the model chooses from your list |
| `AI.EXTRACT` | Each question's answer, as text copied from the document |
| `AI.EXTRACT_SPAN` | The position of each `AI.EXTRACT` answer in the document |
| `AI.MAP` | The text the model writes for the document |
| `AI.SAE_FEATURES` | The model's strongest internal features for the prompt |

The [SQL reference](https://fsdatalab.github.io/quail/docs/user-guide/sql)
describes each function. The model decides which functions you can use;
[Supported models and GPUs](https://fsdatalab.github.io/quail/docs/user-guide/models)
lists the models and the GPU each one needs.

## Quail Server

[Quail Server](https://fsdatalab.github.io/quail/docs/user-guide/server)
runs your queries on a remote GPU. Start `quail-server` on the GPU
machine, and pass its address as `endpoint` when you create the session.
The server keeps a submitted query running after the client disconnects.

## Rust

A Rust program adds the `quail-engine` crate and calls `quail::Session`.
A program with its own DataFusion session adds Quail's AI functions to
the session.
[Use Quail from Rust](https://fsdatalab.github.io/quail/docs/contributing#use-quail-from-rust)
shows both.

## Contributing

[Report a bug](https://github.com/fsdatalab/quail/issues) on GitHub. To
change Quail, follow the
[developer guide](https://fsdatalab.github.io/quail/docs/contributing):
it says how to build from source and which checks to run.
