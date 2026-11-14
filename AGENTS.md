# Writing

The rules cover replies, docs, comments, commit messages, and pull
requests.

- Have a model of the reader. Start with what the reader wants to know
  or do, and build on what the reader already knows.
- Put the main subject and verb at the beginning of every sentence.
- Don't use demonstrative pronouns (this, these, that, those).
- Make each sentence lead into the next sentence.
- Use dictionary words and common technical terms. Define a new term
  inline when you first use it.
- Be as concise as possible.
- Use as much of the person's original wording as possible.
- Call the KV cache "KV". Say "run" or "configuration", never "arm".

# Code

- Write comments only for constraints the code cannot show.
- Write Google style docstrings: a one-line summary, then optional
  Args, Returns, and Raises. Say what the code does, not its history.
- Use one name for each concept; see Names.
- Release what you open. Enter a file, process, or engine on an
  `ExitStack` or in `closing()`, on the failure path too.
- Add no file for a one-time check or run. Delete a script, cell, tool,
  or example once no other file refers to it.

# Layout

- `quail/`: the Python package, with the `Session` API, the GPU
  executor, the model specs, and the Modal images.
- `crates/quail-planner`: plans one model region; no DataFusion.
- `crates/quail-datafusion`: the DataFusion extension, with the AI
  functions, the optimizer rules, and model regions.
- `crates/quail`: the session and the executor. It starts one Python
  worker per GPU. The package name is `quail-engine`.
- `crates/quail-server`: Quail Server, over Arrow Flight.
- `crates/quail-python`: the Python module `_quail_native`.
- `crates/quail-dylib`: a shared library that only test programs link.
- `bench/` runs quail-b, `experiments/` holds GPU cells, and `docs/` is
  the docs site.

# Checks

Run every check before you push:

    uv run --no-sync ruff check quail bench calibrate tests experiments tools
    uv run --no-sync python tools/check_long_strings.py
    uv run --no-sync python tools/check_names.py
    uv run --no-sync vulture
    uv run --no-sync lint-imports
    uv run --no-sync pytest -q
    cargo fmt --check
    cargo clippy --all-targets --locked
    cargo test --locked

- CI runs the checks on every push to `main`. On a pull request, CI
  runs when you add the label `run-ci`; remove and add the label after
  each push.
- `lint-imports` lets a Python package import only the packages below
  it in `[tool.importlinter]`. Put shared code in the lower package.
- `check_long_strings.py` flags a string over 200 characters unless its
  name says content: PROMPT, TEMPLATE, SQL, QUERY, HTML, or TEXT.
- `check_names.py` flags two names for one concept, and one name for
  two.

# Build

- Run `tools/build.sh` when you edit no Rust, and
  `tools/build.sh --compile` after each Rust edit.
- Run tools with `uv run --no-sync`.
- Never pass `--workspace` or `-p` to cargo, and never delete a cache
  to fix a rebuild.

# Release

- Change the version in the workspace `Cargo.toml` on three lines:
  `[workspace.package]`, `quail-planner`, and `quail-datafusion`.
- Merge the change, then run `tools/release.sh <version>` on `main`.
  The script pushes the tag `v<version>`.
- `publish.yml` then publishes the wheels to PyPI and three crates to
  crates.io. It refuses a tag that differs from the version.
- A change of the Rust sources on `main` uploads wheels to the
  `native-wheels` release, which `tools/build.sh` and Modal read.

# Tests

- Answer the model from `tests/helpers/oracle.py`. `Oracle` holds the
  planted truth, the real executor runs the query, and
  `tests/helpers/planted_model.py` is the forward pass. Add no other
  fake of the executor or the model.
- Test a query with a transcript: the text of what Quail did, written
  in the test.

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

  - A transcript shows the plan's AI operators, each question the
    model answered, and the rows. "asked 3 of 6" compares Quail with
    the same query run with every pushdown off; both runs must return
    the same rows.
  - Write a new test with `""` as the transcript. Run
    `QUAIL_UPDATE_TRANSCRIPTS=1 uv run --no-sync pytest <file>`, then
    read the diff.
  - Read `tests/planner/test_guards.py` for a worked example, and
    Andrew Lamb's
    [Transcript Testing](https://andrew.nerdnetworks.org/blog/2026/11/13/transcript-testing/)
    for the method.
  - Assert directly on what a transcript does not show: explain text,
    costs, error messages, and reports.
- Give a bug fix one small regression case, in the test file that
  covers the path.
- Name a test as a sentence, such as
  `test_a_failed_boot_releases_the_gpu`.
- Check what the user sees, such as the rows returned, not the order
  of internal calls.

# Names

- The project is Quail (QUery-Aware Inference Layer). The Python
  package is `quail`; PyPI and crates.io call it `quail-engine`.
- quail-b, the benchmark, lives in
  https://github.com/fsdatalab/quail-bench and is pinned in
  `pyproject.toml`. Change a query or label there first, then move the
  pin.
- A model region is a group of AI calls that run together on the GPU.
- The execution mechanisms are "pipelining", "token-based admission",
  "KV rewind", and "prefix sharing". The attention paths are "unified"
  and "tree".
- The baseline is "stock vLLM"; name its submission strategy,
  operator-at-a-time or pipelining.
- Every `Session` takes an `EngineConfig` with a `model` and a
  `device`; neither has a default.

Use one word per concept in code, following Spark, DuckDB, and
DataFusion. Ask before you invent a name. `tools/check_names.py`
enforces the words:

- `document`, not `doc`. A count is `num_<things>`.
- `gpus` is the GPU count everywhere; `workers` names worker handles.
- `expected_rows` is the rows an operator expects to see; `survivors`
  are the ids still in play after an operator.
- `index`, `position`, `count`, `tokens`, `tokenizer`, `seconds`,
  `estimate`, and `config`, never a short form.
- `resident` is KV already read; `document_head` is the prompt tokens
  an operator shows before each document.
- A document operator emits one `rows:<alias>` port carrying `Rows`,
  and a pair operator emits `rows`. Read ids with `document_ids()`.
- A variable with a type's name holds the type: `rows`, `scored`,
  `labels`, and `answers` hold `Rows`; `table`, `batch`, and `relation`
  hold Arrow tables.

# Scope

- The AI functions are `AI.IF`, `AI.SCORE`, `AI.CLASSIFY`,
  `AI.EXTRACT`, `AI.EXTRACT_SPAN`, `AI.MAP`, and `AI.SAE_FEATURES`.
  DataFusion runs every other part of a query.
- `AI.MAP` writes open-ended text, up to a limit on answer tokens.
  Speculation and forking are out of scope.
- The models and devices are the files in `quail/specs/data`. Each
  model copy runs on one GPU; weights are never split across GPUs.

# Other projects

Read a project's code and docs before you write a piece it already has.

- Quail is built on vLLM, FlashAttention, DeepGEMM, Triton, Apache
  DataFusion, Apache Arrow, Substrait, Hugging Face, and Modal.
- Quail learns from Spark (rule-based plans, `sql()`, Spark Connect,
  `EXPLAIN`), DataFusion (extension points), DuckDB (short docs with a
  runnable example), vLLM (one mechanism per section), and Selinger et
  al. (cost-based join order).

# Experiments

- Run every engine on Modal. Attach GPU cells to the app
  `quail-milestone1` and the worker to `quail-engine`; create no app.
- Check a change with 18 queries at scale factor 0.1, Quail alone:
  `IMDB-1`, `IMDB-2`, `IMDB-4`, `IMDB-11`, `IMDB-15`, `IMDB-24`,
  `FEV-9`, `AGENT-5`, `CUAD-1`, `CUAD-4`, `CUAD-10`, `CUAD-11`,
  `CUAD-15`, `CUAD-20`, `SALES-5`, `SALES-6`, `SALES-7`, and `SALES-8`.
  Run only the queries that use an operator when a change touches one
  operator. Run more only when the user asks.

      uv run modal run --detach -m bench.quailb_parallel \
        --model qwen3-4b-fp8 --scale-factor 0.1 --query "$queries" \
        --no-include-baselines 2>&1 | tee "$run_log"

- Pass `--no-include-baselines`. Run stock vLLM only to compare Quail
  with it, and reuse a saved baseline when nothing changed.
- State the prediction before the run, then report the result against
  the prediction.
- Run one confirming cell, not a sweep, unless the sweep is the point.
- Give the baseline every setting Quail gets, and report the setting.
- Tee every run to a file. Print the function call id (`fc-...`) and
  fetch the result with `modal.FunctionCall.from_id`. Save no result
  to local JSON.
- Keep results on the `quail-results` volume, never in git. Cite a
  result by its volume path.
- Report three numbers per query, all without model startup:
  - query time;
  - input tokens per second (`input_tokens_per_second`): every prompt's
    full length, a shared prefix counted each time, over query time;
  - `$/query`: query time in hours, times GPUs, times the device's
    `usd_per_hour` ($3.9492 on H100).

# Reports and plots

- Keep reports, plot code, and figures on the branch
  `cursor/reports-dev-f955`, never on a branch that targets `main`.
- Write one script per report, `reports/make_<slug>_plots.py`. The
  script takes the folder of volume files as its first argument, lists
  the `modal volume get` commands in its docstring, and computes every
  percentage itself.
- Add a plot only when it shows the point better than a table.
- Reuse saved results, name the source run, and leave out results from
  an older query definition.

A quail-b plot is one vector PDF per model, one page, every query:

- Draw four panels of grouped bars, Quail first, then stock vLLM with
  its submission strategy in the legend:
  1. latency in seconds, log scale, with stock vLLM's time over
     Quail's above each query;
  2. throughput in input tokens per second, log scale;
  3. KV regret (`regret_tokens`), log scale;
  4. accuracy against the saved reference labels.
- Draw the speed-of-light (SoL) estimate as a line across the latency
  and throughput bars. Compute it with `quail.speed_of_light_estimate`,
  label it as an estimate, and give it no accuracy.
- Put the run ID, the SoL assumptions, and each marker's meaning in a
  footer. Mark a missing value with an x and a zero with a dash.

Every other figure is a 300 DPI PNG:

- Use `reports/quail.mplstyle` and the colors in
  `reports/plot_colors.py`, one color per category.
- Make every mark encode data. Label data directly.
- Write the unit alone on each axis, and the rest in the title.
- Put compared things side by side on one axis, with the difference
  written beside them.
- Use a log scale only across more than one order of magnitude.
- Put notes in the report text, never under the plot.

# Pull requests

Write for a reviewer who has not followed the conversation. Keep the
description short, and leave out a heading with nothing to say:

- **Relevant links:** the issue, the superseded pull request, and the
  saved runs.
- **What:** the changes, one line each.
- **Why:** the problem each change solves.
- **How:** the approach, with one concrete example, and where to start
  reading.
- **Testing:** a checklist. Check what ran, with numbers and the
  commit. Leave unchecked what did not run.
- **Anything else?:** known gaps.

- Describe the final change, not its history.
- Give the source and configuration of each measurement. Separate a
  measurement from an estimate.
- Claim no advantage without evidence.
- Add a mermaid diagram for a design change, and a link to the saved
  run for numbers.
- Prefer a small, focused change.
