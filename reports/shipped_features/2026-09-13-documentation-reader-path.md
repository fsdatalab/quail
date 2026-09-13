# Documentation reader path

## What changed

- The repository README now explains Quail before showing setup and benchmark
  commands. It includes one query, the four execution mechanisms, the
  supported scope, and a task-based documentation index.
- The documentation home page now follows the same order. It removes the
  unsupported product roadmap and the single-query benchmark table.
- The quickstart now uses the maintained `demos/quickstart.py` example. It is
  154 lines, compared with 313 lines before this change.
- Installation choices are stated before CUDA and cache details.
- Repository and source links now use `fsdatalab/quail` and the `main`
  branch. The old links used the repository's former
  `fsdatalab/quail-exploration` name and a deleted `blogs` branch.
- The site uses IBM Plex Sans and IBM Plex Mono. The home page uses a compact
  technical layout that works in the existing light and dark themes.

## Why

The old entry pages repeated the same long product description. They mixed
current behavior, unsupported future features, architecture details, and old
measured output before a reader could run one query. The quickstart also
depended on copied output from a separate eight-document example instead of
the maintained demo.

The new path is: understand the supported query, run it, inspect its result,
then choose a task-specific guide.

## Prediction and checks

Before the checks, the prediction is that the docs site will build without
broken MDX or TypeScript, all CPU project checks will pass unchanged, and a
fresh reader will be able to identify Quail's scope, requirements, first
command, and next guide from the rewritten entry pages.

- `pnpm build`: passed. Next.js generated 93 static pages.
- `pnpm types:check`: passed.
- Ruff, the long-string check, and Vulture: passed.
- `pytest`: not run. The environment cannot read the private
  `fsdatalab/quail-bench` dependency, so `uv sync --frozen` could not install
  the test environment. GitHub also reports that repository as unavailable
  to this environment.

No Modal run is needed. This change does not alter model execution, so it has
no Modal function call id or `quail-results` volume path.
