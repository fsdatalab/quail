# How to communicate in this project

Write every response simply, plainly, and clearly, like two human
engineers talking to each other at a whiteboard. No jargon.

- Use bullet points for any response longer than a few sentences.
- No analogies or metaphors, ever. Say the literal thing: "the
  experiment is running on Modal", not "the flight is in the air".
- No invented shorthand or dramatic phrasing ("banked", "landed",
  "armed", "healthy") when a plain verb exists: saved, finished,
  set up, running.
- Use everyday words. If a technical term is unavoidable, define it in
  one sentence the first time it appears, then use it consistently.
- Short sentences. One idea per sentence. Lead with the answer, then
  give the reasoning.
- Use numbers, and say in plain words what each number is compared
  with: "52.3 seconds, compared with the 47 seconds the hardware needs
  for the unavoidable work alone."
- When something failed or is uncertain, say so directly and say what
  would settle it.
- Code comments state constraints the code cannot show; nothing else.
- Docstrings follow Google style: one-line summary, blank line, then
  optional Args/Returns/Raises sections. Keep them short — say what
  the code does, not why it was written or what it replaced.
- Call the KV cache "KV". Do not rename it with analogies like "notes".
- Never say "arm" or "arms" for the runs of an experiment. Say
  "run", "configuration", or name the method being run.

For Claude Code users: install the `plain-writing` skill from
`docwriter-org/plain-writing-skill` for automated enforcement of
these rules.

# Naming in this project

- The project is Quail (QUery-Aware Inference Layer). The package is
  `quail`. Nothing is called DocEngine any more.
- The three mechanisms are "pipelining", "token-based admission", and
  "KV rewind". Say those names.
- "Chain mode" is the internal name for KV rewind (one living request
  per document). Either is fine in code; prefer "KV rewind" in prose.
- The comparison is against "stock vLLM", and say which submission
  strategy it used: separate requests per stage, or stage-major waves.
- `de1|` in request ids is a wire-format version tag, not a product
  name. Leave it alone.

# Scope

Filter queries only, Qwen3 4B fp8 or Qwen3 32B fp8, one H100 per model
(one model copy per GPU - no tensor-parallel weight sharding across
GPUs). Open-ended maps, classification, speculation, and forking were
removed on purpose. Do not reintroduce them without being asked; if a
change needs one of them, say so instead of quietly adding it back.

# Experiments

- Every engine run goes through Modal; there is no local GPU.
- Never create new Modal app names; caches and warm state ride on
 the app. New GPU cells attach to an existing app
 ("quail-milestone1" for cells, "quail-engine" for the worker).
- Tee every Modal run to a file. The CLI drops old log lines.
- Do not write Modal return values to local JSON files. Print the
  function call id (the `fc-...` Modal assigns to one invocation)
  and keep that id in the tee file. When you need the result, pull
  it with `modal.FunctionCall.from_id("<id>").get()`.
- Experiment data lives on the `quail-results` Modal volume, summaries
  as well as per-item records. Do not commit it. Reports cite it by
  volume path: `/results/ablations/<file>.json`.
- A plot script reads its numbers from the volume. Take the workdir
  holding the pulled files as its first argument, and put the
  `modal volume get` commands in its docstring so the figure can be
  rebuilt from the report alone. Derive percentages, ratios and other
  computed values in the script rather than storing them.
- State the prediction before the run, then report what happened
  against it.
- Compute from measured constants first; run one confirming cell, not
  a sweep, unless the sweep is the point.
- A baseline must be configured as well as the thing it is compared
  with. If our side gets a plan-derived setting, the baseline gets the
  analytically equivalent one. Report the setting alongside the
  result.
- Report query time, throughput, and GPU cost for every benchmark
  query. Use the following definitions consistently:
  - For a filter-only query, `documents/second` is the number of input
    document rows divided by query runtime in seconds.
  - For a query with joins, `document pairs/second` is the number of
    evaluated document pairs, summed across all join stages, divided
    by query runtime in seconds.
  - `$/query` is query runtime in hours multiplied by the number of
    GPUs and the H100! hourly price. Use
    `quail.bench.evaluate.H100_USD_PER_HOUR`, which is currently
    $3.9492 from https://modal.com/pricing.
  - The primary `$/query` number excludes model startup, just as the
    primary query time does. If startup cost is useful, report it as a
    separate clearly labeled number.

# Reports

All experiment and feature reports live under `reports/`.

- Every PR that includes an experiment must produce a report in
  `reports/`. Name the file `YYYY-MM-DD-<short-slug>.md`.
  The report states the setup, the prediction, the measured result,
  and what the numbers mean. Cite the data by its `quail-results`
  volume path.
- When a report is superseded or its numbers are no longer current,
  delete the report, its plot script, and its PNGs from
  `reports/plots/`. Before starting a new task, scan `reports/`
  for outdated reports, orphaned plot scripts, and PNGs not
  referenced by any current report, and delete them all.
- When a PR ships a new feature (a code change that lands on main),
  add a short description in `reports/shipped_features/`.
  Name the file `YYYY-MM-DD-<short-slug>.md`. It should say what
  changed, why, and the before/after numbers if applicable.
- `reports/engine-wiki.md` is a living reference doc, not a
  per-PR report. Update it in place when the engine's design changes.

# Issues and PR descriptions

Include a figure whenever one carries the point better than text:

- For measured numbers, embed the report's committed plot. Link the
  image by its raw GitHub URL pinned to a commit
  (`.../raw/<sha>/reports/plots/<name>.png`) so it keeps
  rendering as the branch moves. Do not make new plots just for an
  issue or PR body; reuse the report's.
- For a design, plan, or dataflow change, include a mermaid diagram
  of the structure (GitHub renders ```mermaid blocks).

# Plots

Every report with measured results should include at least one plot.

## Where plotting code lives

- One script per report (or per group of related reports), named
  `make_<slug>_plots.py`, in `reports/`.
- Output PNGs go to `reports/plots/`. Delete a report's PNGs when
  the report is deleted.
- Reference plots in the report by relative path:
  `"Figure: plots/<name>.png"`.
- Each script should be runnable from the repository root, given a
  workdir holding the files pulled off the volume:
  `uv run --with matplotlib python reports/make_<slug>_plots.py $W`.
- The PNGs are committed; the numbers behind them are not. That is
  what lets a PR body embed a figure by raw GitHub URL.

## Style

Use `reports/quail.mplstyle` in every plotting script:

    plt.style.use(Path(__file__).parent / "quail.mplstyle")

Import colors from `reports/plot_colors.py`:

    from plot_colors import BLUE, GRAY, GREEN, RED, DARK, ORANGE, TEAL

### Principles

These follow Tufte's data-ink ratio and Heer's encoding
effectiveness research:

- Every mark encodes data. No background fills, no 3D effects, no
  decorative gridlines.
- Label data directly on or next to bars/points. Use a legend only
  when direct labels would overlap or repeat.
- State the unit on every axis. Omit the plot title when the report
  heading already says what the plot shows.
- When comparing two things, put them next to each other on the same
  axis so the reader's eye measures the gap, not their memory.
- Annotate the delta (speedup, difference) inline near the data it
  describes.
- Use color to distinguish categories, not to decorate. Two
  categories need two colors, not five.
- Use a log scale only when the data spans more than one order of
  magnitude. Say so in the axis label.
- Use 300 DPI PNG files. No SVG.
- Make the canvas large enough that text and data marks remain sharp when
  viewed on GitHub.
