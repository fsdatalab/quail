# Writing

- Have a model of the user or person who asked the question. Start with
  what they want to know or do, and build on what they already know.
- Put the main subject and verb of every sentence at the beginning of
  the sentence. Use tenses consistent with your model of the person who
  asked the question.
- Don't use demonstrative pronouns (this, these, that, those).
- Make each sentence lead into the next sentence.
- Use only words and phrases found in the dictionary or common
  technical terms (simplified technical English). Define new or
  ambiguous terms inline when you first introduce them.
- Be as concise as possible.
- Use as much of the person's original wording as possible.
- Call the KV cache "KV".
- Say "run" or "configuration" for the runs of an experiment, never
  "arm".

## Code

- Comments state constraints the code cannot show, nothing else.
- Docstrings follow Google style: a one-line summary, a blank line,
  then optional Args, Returns, and Raises. Say what the code does, not
  why it was written or what it replaced.
- Use one name for each concept across the planner, the executor, the
  docs, and the metrics. `tools/check_names.py` enforces the chosen
  words; see Names below.
- Whatever opens a file, a process, or an engine owns it: enter it on
  an `ExitStack` or in `closing()`, and release it on the failure
  path as well as the normal one.
- Do not add a file for a one-time check or run; the result is the
  record. Delete a script, cell, tool, or example once no other file
  refers to it.

# Layout

- `quail/` is the Python package: the `Session` API, the GPU executor,
  the model specs, and the Modal images.
- `crates/` is the Rust workspace:
  - `quail-planner` plans one model region and does not use DataFusion.
  - `quail-datafusion` is the DataFusion extension: the AI functions,
    the optimizer rules, and model regions.
  - `quail` is the session and the executor. It starts one Python
    worker per GPU.
  - `quail-server` is Quail Server, over Arrow Flight.
  - `quail-python` is the Python module `_quail_native`.
  - `quail-dylib` is a shared library that only test programs link.
- `bench/` runs quail-b, `experiments/` holds GPU cells, and `docs/` is
  the docs site.

# Checks

Run the checks before pushing:

    uv run --no-sync ruff check quail bench calibrate tests experiments tools
    uv run --no-sync python tools/check_long_strings.py
    uv run --no-sync python tools/check_names.py
    uv run --no-sync vulture
    uv run --no-sync lint-imports
    uv run --no-sync pytest -q
    cargo fmt --check
    cargo clippy --all-targets --locked
    cargo test --locked

- CI runs the same checks on every push to `main`. On a pull request,
  CI runs when the pull request gets the label `run-ci` or is marked
  ready for review; remove the label and add it again after a new push.
- Ruff enforces line length 88, import order, naming, and Google style
  docstrings (`[tool.ruff]` in `pyproject.toml`).
- `lint-imports` checks that each Python package imports only packages
  below it, in the order `[tool.importlinter]` in `pyproject.toml`
  lists: `worker` on top of `api`, `backends`, `remote`, `execution`,
  `physical`, and so on down to `specs`. Put shared code in the lower
  package rather than importing upward.
- `tools/check_long_strings.py` flags string literals over 200
  characters. Long content is allowed when its name identifies content
  (a name containing PROMPT, TEMPLATE, SQL, QUERY, HTML, or TEXT), kept in a
  prompts module or folder, or plainly HTML or SQL. Docstrings and
  expected transcripts are exempt. Shorten long messages and log lines.
- `tools/check_names.py` reports two names for one concept or one
  name for two: duplicate function bodies, a parameter passed on
  under another keyword, identifiers that differ by a synonym, and a
  variable whose name describes a type it does not hold. `--inventory`
  lists what the code defines, for finding such names by hand.

## Building

- **A developer who edits no Rust needs only uv.** `tools/build.sh`
  downloads the wheel CI built for the checkout's Rust sources
  (`tools/native_key.py` prints their key) and installs `_quail_native`
  with no compiler. After it, run tools with `uv run --no-sync`: a
  plain `uv run` compiles the module.
- **A developer who edits Rust runs `tools/build.sh --compile`**, and
  runs it again after each Rust edit. Install rustup and uv first;
  Linux x86_64 also needs clang and lld. The first build compiles
  DataFusion, about 20 minutes on 4 cores; later builds reuse it.
- **Keep one Cargo home and one `target/`.** Do not delete caches to
  fix a rebuild. If DataFusion compiles again without cause, compare
  the compiler, the features, the rustflags, and the target directory
  before changing a setting; `CARGO_LOG=cargo::core::compiler::fingerprint=info`
  shows why Cargo rebuilt a crate.
- **Do not pass `--workspace` or a narrow `-p` to cargo.** Either one
  changes what Cargo builds. `--workspace` links the Rust standard
  library into `quail-dylib`, and every test program then fails to
  link with "cannot satisfy dependencies so `std` only shows up once";
  run `touch crates/quail-dylib/src/lib.rs` and build again to recover.
  A narrow `-p` can change a dependency's features and compile
  DataFusion again.
- **Test programs link DataFusion, Arrow, and Quail from `quail-dylib`**,
  so each is a few megabytes. A new test program names the crate with
  `use quail_dylib as _;` and lists it under `[dev-dependencies]`. An
  example a user copies does not name it.
- **The `quail-server` a user installs is a Python entry point** that
  calls `_quail_native.serve`, so the wheel holds DataFusion once.
- Cargo keeps the files of old builds in `target/`; `cargo sweep`
  (cargo-sweep) deletes them.

## Releases

- The version is in the workspace `Cargo.toml`, on three lines: under
  `[workspace.package]`, and on the `quail-planner` and
  `quail-datafusion` lines below it. Change all three together.
- `publish.yml` builds one wheel per platform. A change of the Rust
  sources on `main` uploads the wheels to the `native-wheels` release,
  which `tools/build.sh` and the Modal image read. A version tag also
  publishes the wheels to PyPI and the crates `quail-planner`,
  `quail-datafusion`, and `quail-engine` to crates.io.
- To release, merge the version change, then run
  `tools/release.sh <version>` on `main`. The script pushes the tag
  `v<version>`, and `publish.yml` refuses a tag that differs from the
  version.

## Tests

- An end-to-end test answers the model from `tests/helpers/oracle.py`:
  `Oracle` holds the planted truth, `oracle_executor` runs a request's
  graph on it, and `finish` runs a query to its result. The real
  executor runs the graph; `tests/helpers/planted_model.py` is the
  forward pass, which answers each prompt from the truth. Do not add
  another fake of the executor or the model.
- A test of a query is a transcript test. The test runs the query and
  compares a transcript, the text of what Quail did, with the expected
  transcript written in the test:

      def test_an_or_asks_only_where_the_column_is_not_true():
          assert_transcript(_transcript(
              f"SELECT r.id FROM reviews r WHERE r.long OR {KIND}"), """\
      --- SELECT r.id FROM reviews r WHERE r.long OR AI_FILTER(...) ---
      plan
        __quail_answer_1 = ai_filter('kind: {0}', r.body), guard: r.long IS NOT TRUE
      asked 3 of 6
        'kind: {0}': r1 FALSE, r2 TRUE, r4 TRUE
      rows 5
        r0
        ...
      """)

  - A transcript shows the plan's AI operators, each question the model
    answered, and the rows returned. "asked 3 of 6" compares Quail with
    the same query run with every pushdown off; the two runs must
    return the same rows.
  - Write a new test with `""` as the expected transcript. Run
    `QUAIL_UPDATE_TRANSCRIPTS=1 uv run --no-sync pytest <file>` to write
    the transcript into the test, then read the diff before you commit.
  - `tests/helpers/transcript.py` builds the transcript;
    `tests/planner/test_guards.py` is a worked example.
  - Andrew Lamb describes the method in
    [Transcript Testing](https://andrew.nerdnetworks.org/blog/2026/11/13/transcript-testing/);
    [DataFusion's transcript tests](https://github.com/apache/datafusion/pull/26897)
    are his example.
  - Assert directly, without a transcript, on what a transcript does
    not show: explain text, costs, error messages, and reports.
- A bug fix gets one small regression case, in the test file that
  already covers that path. Name a test as a sentence stating the
  behavior, such as `test_a_failed_boot_releases_the_gpu`.
- A test checks the behavior the user sees, such as the rows returned
  or the files left behind, not the order of internal calls.

# Names

- The project is Quail (QUery-Aware Inference Layer); the package is
  `quail`. On PyPI and crates.io the name is `quail-engine`; the Rust
  library is `quail`, and it re-exports `quail-datafusion` as
  `quail::extension` and `quail-planner` as `quail::planner`.
- quail-b, the benchmark, lives in
  https://github.com/fsdatalab/quail-bench and is installed here as
  `quail_b`, pinned to a commit in `pyproject.toml`. `bench/` is
  Quail's runner for it. Change queries or labels there first, then
  move the pin here.
- A model region is a group of AI calls that run together on the GPU.
- The execution mechanisms are "pipelining", "token-based admission",
  "KV rewind", and "prefix sharing".
- The attention paths are "unified" and "tree".
- Compare against "stock vLLM", and say which submission strategy it
  used: operator-at-a-time or pipelining.
- Every `Session` takes an `EngineConfig` that names `model` and
  `device`; neither has a default. `gpus` defaults to 1 and `backend`
  to `"quail"`.

## Names in code

One word per concept, following Spark, DuckDB, and DataFusion where
they have one. When a concept has no name yet, ask before inventing
one. `tools/check_names.py` enforces the chosen words:

- `document` and `documents`, not `doc` or `docs` (the public `docs()`
  API keeps its name).
- A count is `num_<things>`, such as `num_documents` or `num_pages`,
  never `n_<things>`, `<thing>_count`, or `number_of_<things>`.
- `gpus` is the GPU count everywhere: the planner, the executor, the
  settings, and the metrics. `workers` names worker handles only.
- `expected_rows` is the rows an operator expects to see, whatever it
  reads: documents, pairs, or tuples. `survivors` are the ids still in
  play after an operator.
- `index` not `idx`, `position` not `pos`, `count` not `cnt`,
  `tokens` not `toks`, `tokenizer` not `tok`, `seconds` not `secs`,
  `estimate` not `est`, `config` not `cfg`; `written_position` and
  `exec_index` for a node's places in the query and the run.
- `resident` is KV already read; `document_head` is the prompt tokens
  an operator shows before each document; `run` or `configuration`,
  never `arm`.
- Every document operator emits one `rows:<alias>` port carrying
  `Rows`, and a pair operator emits `rows`. Read ids from `Rows` with
  `document_ids()`; do not keep a separate ids port.
- A variable with a type's name holds that type: `rows`, `scored`,
  `labels`, and `answers` hold `Rows`; `table`, `batch`, and
  `relation` hold Arrow tables.

# Scope

- The AI functions are `AI.IF`, `AI.SCORE`, `AI.CLASSIFY`,
  `AI.EXTRACT`, `AI.EXTRACT_SPAN`, `AI.MAP`, and `AI.SAE_FEATURES`.
  DataFusion runs every other part of a query.
- `AI.MAP` writes open-ended text for each row, up to a limit on answer
  tokens. Speculation and forking are not supported.
- The models and devices are the files in `quail/specs/data/models`
  and `quail/specs/data/devices`. Each model copy runs on one GPU; no
  tensor-parallel weight sharding.

# What Quail builds on and learns from

Quail is built from these projects. Read their code and docs before
reinventing a piece, and hold Quail's code and docs to their
standard.

Built on:

- **vLLM**: model definitions, weight loading, and kernels; also the
  main baseline.
- **FlashAttention**: the attention kernels, over paged KV.
- **DeepGEMM**: the FP8 matrix multiplies.
- **Triton**: Quail's fused kernels (RMSNorm plus quantize, QK-norm
  plus RoPE, SiLU plus quantize, the attention merge).
- **Apache DataFusion** (with sqlparser): SQL parsing and binding,
  logical optimization, and every relational operator: scans, plain
  filters, plain joins, aggregates, sorts, and limits. Quail adds the
  AI operators and runs only those.
- **Apache Arrow**: tables, and the streams that carry rows between
  DataFusion and Quail.
- **Substrait**: the plan format quail-b queries are written in.
- **Hugging Face** (`transformers`, `huggingface_hub`, `datasets`):
  tokenizers, checkpoints, and benchmark data.
- **Modal**: every GPU run and the result volumes.

Learned from. Treat these as whole projects to study, not single
features: their APIs, their code layout, their tests, their docs, and
how they release and explain changes.

- **Apache Spark**: the model for Quail as a whole.
  - Catalyst, for logical and physical plans rewritten by rules
    (Quail's planner rules).
  - The SQL API, for Quail's `sql()`.
  - Spark Connect, for a thin client talking to a remote server
    (Quail Server and the remote `Session`).
  - `EXPLAIN` and the Spark UI, for showing a plan and what it cost
    (`explain()`, `explain(analyze=True)`).
- **Apache DataFusion**: also the model for how to extend an engine:
  clear extension points (Quail's extension registry, custom
  operators, and rules) and docs for each. Projects built on it, such
  as Comet, InfluxDB 3, and Lance, show how an extension marks the
  part of a plan it runs and leaves the rest to the host.
- **DuckDB**: an engine that is easy to embed and whose docs answer a
  question in one short page with a runnable example.
- **Apache Arrow**: columnar data and Arrow Flight for moving results
  between processes.
- **vLLM**: the serving engine Quail runs beside and compares
  against; its paper and docs explain one mechanism per section, such
  as paged KV and prefix caching, with a small example.
- **Modal**: docs that start with what a feature does, then a small
  example with full context.
- **Selinger et al. (1979)**: cost-based join ordering, which the
  planner follows.

# Experiments

- Every engine run goes through Modal; there is no local GPU.
- Do not create Modal app names. Attach GPU cells to
  `quail-milestone1` and the worker to `quail-engine`; caches and warm
  state belong to those apps.
- To check a change, run these 18 queries at scale factor 0.1, with
  Quail alone. Do not run the full suite; it is slow. Together they
  cover each AI operator alone, chained with the others, and followed
  by relational operators:
  - `IMDB-1`: one filter.
  - `IMDB-2`: one join.
  - `IMDB-4`: two filters, then a join.
  - `FEV-9`: four filters and a chain of three joins.
  - `IMDB-11`: classify.
  - `AGENT-5`: three classifies over the same rows.
  - `IMDB-15`: a classify, a join, then another classify.
  - `CUAD-1`: extract, one span per row.
  - `CUAD-4`: extract, several spans per row.
  - `CUAD-10`: three extracts over the same rows.
  - `CUAD-15`: a filter, an extract, then sort and limit.
  - `CUAD-11`: classify, then aggregate.
  - `SALES-5`: a relational filter, a score, then top 25.
  - `IMDB-24`: a filter, then three maps over the same rows.
  - `CUAD-20`: three extracts and a map.
  - `SALES-6`: a map whose answer a prefix test can stop early.
  - `SALES-7`: a relational filter, then the same map and test.
  - `SALES-8`: the same map with no test, the cost with no early stop.

  `bench.quailb_parallel` runs each query family on its own H100, so
  this set uses five H100s: IMDB, FEV, AGENT, CUAD, and SALES. If a
  change touches one operator, run only the queries that use it.

  For example:

      queries=IMDB-1,IMDB-2,IMDB-4,FEV-9,IMDB-11,AGENT-5,IMDB-15
      queries=$queries,CUAD-1,CUAD-4,CUAD-10,CUAD-15,CUAD-11,SALES-5
      queries=$queries,IMDB-24,CUAD-20,SALES-6,SALES-7,SALES-8
      uv run modal run --detach -m bench.quailb_parallel \
        --model qwen3-4b-fp8 --scale-factor 0.1 --query "$queries" \
        --no-include-baselines \
        2>&1 | tee "$run_log"

  Run more queries, or a larger scale factor, only when the change
  targets them or the user asks.
- Do not run stock vLLM or pipelined vLLM by default.
  `bench.quailb_parallel` runs them unless told not to, so pass
  `--no-include-baselines`. Run a baseline only
  for an experiment whose point is to compare Quail with it, or when
  the user asks. Reuse saved baseline results when the queries, scale
  factor, and model have not changed.
- Tee every Modal run to a file; the CLI drops old log lines.
- Do not save Modal return values to local JSON. Print the function
  call id (`fc-...`) into the tee file, and fetch the result later
  with `modal.FunctionCall.from_id("<id>").get()`. An Arrow Flight
  request has no function call id; print the Flight query id and the
  result volume path instead.
- Results live on the `quail-results` volume, summaries and per-item
  records alike. Do not commit them. Cite them by volume path, such as
  `/results/ablations/<file>.json`.
- State the prediction before the run, then report the result against
  it.
- Work from measured constants first. Run one confirming cell, not a
  sweep, unless the sweep is the point.
- Configure the baseline as well as Quail. If Quail gets a setting
  from the plan, the baseline gets the equivalent one. Report the
  setting with the result.
- For every benchmark query, report query time, throughput, and GPU
  cost:
  - Throughput is input tokens per second: the full length of every
    prompt the query evaluated, summed, counting a shared prefix every
    time whether or not its KV was reused, divided by query time.
    quail-b reports it as `input_tokens_per_second`. Give document and
    pair counts beside it as plain counts, not as throughput.
  - `$/query` is query time in hours times the number of GPUs times
    `DEVICES[device].usd_per_hour`. On H100 that is
    `H100_USD_PER_HOUR` ($3.9492, from https://modal.com/pricing).
  - Query time and `$/query` exclude model startup. Report startup
    separately if it matters.

# Reports and plots

- Reports and plots do not go on branches that target `main`, and a
  PR does not get its own report.
- They live on the `cursor/reports-dev-f955` branch, with the plot
  code, figures, and `engine-wiki.md`. Start report work there and
  target changes back to it.
- A plot script takes the directory of files pulled from the volume
  as its first argument, lists the `modal volume get` commands in its
  docstring, and computes percentages and ratios itself.
- Put one script per report in `reports/make_<slug>_plots.py` and its
  figures in `reports/plots/`. Delete a report's figures with the
  report. Run scripts from the repository root:
  `uv run --with matplotlib python reports/make_<slug>_plots.py $W`.
- Add a plot only when it shows the point better than a table.

## QUAIL-B plots

- One vector PDF per model, such as
  `reports/plots/classify_sf05_qwen3_4b.pdf`, with embedded fonts.
  Each is one page with every query of the run; no separate
  single-query figures. One script per report,
  `reports/make_<slug>_plots.py`, makes them all. Commit the PDFs only.
- The page has four panels, top to bottom. Each shows grouped bars per
  query, Quail first and then stock vLLM, in the same colors on every
  page; name stock vLLM's submission strategy in the legend.
  1. Latency in seconds, including planning and without model startup,
     on a log scale. Write stock vLLM's time over Quail's and the SoL
     seconds above each query.
  2. Throughput in input tokens per second (`input_tokens_per_second`),
     on a log scale.
  3. KV regret (`regret_tokens`), on a log scale: fresh tokens minus the
     fewest input tokens the requests need with unlimited KV (each
     document once, each question tail and anchor frame once per
     document, each partner suffix once after its anchor). quail-bench
     computes it (`quail_b.minimum`) from saved answer tables after the
     run. Say in the panel title when an engine's regret is approximate
     and that SoL's is 0.
  4. Accuracy: agreement with the saved reference labels, naming the
     reference model (label agreement for classification queries).
- Draw the speed-of-light (SoL) estimate as a horizontal line across
  each query's bars in the latency and throughput panels. Compute it on
  the CPU with `quail.speed_of_light_estimate`, passing the saved
  reference labels as the answer and label oracles
  (`bench.quailb.answer_oracle`), and save it to the results
  volume. Its throughput is the Quail run's input tokens over the SoL
  seconds. Label it as an estimate, give its KV capacity and survivor
  assumptions, and never give it an accuracy. Check the query
  definitions and corpus before reusing an estimate; recompute stale
  ones from saved inputs.
- Put the run ID, the SoL source and survivor assumption, and the
  meaning of each marker in a footer line. Mark a missing value with
  an x and a zero with a dash; never show a missing value as zero, and
  label a baseline that was not run as not run.
- Reuse saved results unless asked to rerun, and name the source run.
  Leave out measurements from an older query definition.
- When the layout changes, delete the old figures and scripts and
  update every reference.

## Other figures

- 300 DPI PNG, committed; no SVG.
- Use `reports/quail.mplstyle`
  (`plt.style.use(Path(__file__).parent / "quail.mplstyle")`) and the
  colors in `reports/plot_colors.py`.
- Every mark encodes data: no background fills, 3D, or decorative
  grid lines.
- Label data directly; use a legend only when labels would overlap.
- In profiling plots, keep the trace's own function names, such as
  `vllm.scheduler.schedule`.
- Every axis label is the unit alone, such as "microseconds per fresh
  token"; other context goes in the title. Every plot has a title,
  naming the query when it shows one.
- No footnotes or captions under the plot. Put notes such as "SoL is
  an estimate" or the reference model in the report or PR text, not
  in the figure.
- Put compared things side by side on one axis, and write the
  difference next to them.
- One color per category. Use a log scale only across more than one
  order of magnitude, and say so on the axis.
- Leave room between panels, keep labels the same across panels, and
  do not repeat the title in bar or tick labels.

# Issues and pull requests

- Include a figure when it shows the point better than text: a link
  to the saved run or to a plot on the report branch, pinned to a
  commit, for numbers; a mermaid diagram for a design or dataflow
  change.

## PR descriptions

Write for a reviewer who has not followed the conversation, and follow
the writing rules above. Keep the description short: a small change
needs a few lines under each heading. Use these headings, and leave one
out when it has nothing to say:

- **Relevant links:** the issue, the pull request it supersedes, and
  the saved runs.
- **What:** the changes, one line each.
- **Why:** the problem each change solves.
- **How:** the approach, with one concrete example, and where to start
  reading a large change.
- **Testing:** a checklist. Check what ran, with its numbers and the
  commit; leave unchecked what did not run, and say so.
- **Anything else?:** known gaps and what the change leaves for later.

Rules for the content:

- Describe the final change, not the history of making it.
- Give the source and configuration of each measurement, and separate
  measurements from estimates.
- Do not claim an advantage without evidence, and do not imply that
  passing tests show more than they cover.
- Put long tables behind a link to the saved run.
- Prefer a focused change that can be reviewed and merged promptly; a
  clear description does not make a large, mixed change small.

## Guidance for review

- [Why we need pull request descriptions, and how to craft them][pr-writing]
  explains why descriptions should preserve intent, decisions, risks,
  testing, and limits.
- Martin Fowler's [Pull Request][fowler-pr] discusses coherent
  contributions, small PRs, and timely review.
- Fowler's [Patterns for Managing Source Code Branches][fowler-branches]
  explains frequent integration and the cost of delaying it. These
  articles guide scope and review practice; they do not prescribe a
  description template.

[pr-writing]: https://wsbctechnicalblog.github.io/pull-request-descriptions-empowered-by-engineering-practices.html
[fowler-pr]: https://martinfowler.com/bliki/PullRequest.html
[fowler-branches]: https://martinfowler.com/articles/branching-patterns.html
