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
- Call the KV cache "KV". Do not rename it with analogies like "notes".
- Never say "arm" or "arms" for the runs of an experiment. Say
  "run", "configuration", or name the method being run.

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
- State the prediction before the run, then report what happened
  against it.
- Compute from measured constants first; run one confirming cell, not
  a sweep, unless the sweep is the point.
- A baseline must be configured as well as the thing it is compared
  with. If our side gets a plan-derived setting, the baseline gets the
  analytically equivalent one. Report the setting alongside the
  result.
