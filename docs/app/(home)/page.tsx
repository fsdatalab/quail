import Link from 'next/link';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';

const example = `import quail

session = quail.Session()
session.register("reviews", quail.DocumentProvider.from_table(reviews, id_col="id"))
session.register("aspects", quail.DocumentProvider.from_table(aspects, id_col="id"))

pairs = session.sql("""
    SELECT r.id, a.aspect
    FROM reviews r
    JOIN aspects a
      ON AI_FILTER(PROMPT('Does the review in DOCUMENT {0} discuss the movie aspect in DOCUMENT {1}?',
                          r.body, a.aspect),
                   {'selectivity': 0.15})
    WHERE AI_FILTER(PROMPT('Judge strictly from the review above whether it mentions at least one positive aspect of the movie.

{0}

Instruction: answer TRUE if the review mentions at least one positive aspect of the movie, FALSE otherwise.', r.body),
                    {'selectivity': 0.6})
""").run()

pairs.collect()      # 10 (r.id, a.aspect) pairs from 8 IMDB reviews, in 1.53 s
pairs.report         # query time, tokens, per stage selectivity, KV reuse`;

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

function Feature({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-fd-border bg-fd-card p-4">
      <h3 className="font-medium mb-1">{title}</h3>
      <p className="text-sm text-fd-muted-foreground">{children}</p>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="mx-auto w-full max-w-4xl px-6 py-16 flex flex-col gap-16">
      <header className="flex flex-col gap-5">
        <h1 className="text-4xl font-bold tracking-tight">Quail</h1>
        <p className="text-xl text-fd-muted-foreground leading-relaxed">
          Quail is a query engine for AI functions in SQL. It runs the
          language model calls that Snowflake&apos;s <code>AI_FILTER</code>{' '}
          and BigQuery&apos;s <code>AI.IF</code> describe, on your own
          GPUs, as one planned query instead of one request per row.
        </p>
        <p className="text-fd-muted-foreground">
          The planner sees the whole query before the model sees a
          single document. Each document&apos;s KV is computed once and
          reused for every question about it, every forward pass is
          full, and a document moves to its next question the moment it
          passes the current one. Quail runs Qwen3 4B fp8 and Qwen3 32B
          fp8 with one model copy per H100, through Modal, reads Arrow,
          Parquet, and Hugging Face datasets, and returns Arrow tables.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/docs/user-guide/quickstart"
            className="rounded-md bg-fd-primary px-4 py-2 text-sm font-medium text-fd-primary-foreground"
          >
            Quickstart
          </Link>
          <Link
            href="/docs"
            className="rounded-md border border-fd-border px-4 py-2 text-sm font-medium"
          >
            Documentation
          </Link>
          <Link
            href="/docs/architecture"
            className="rounded-md border border-fd-border px-4 py-2 text-sm font-medium"
          >
            Architecture
          </Link>
        </div>
      </header>

      <Section title="What a query looks like">
        <DynamicCodeBlock lang="python" code={example} />
        <p className="text-fd-muted-foreground">
          This is the join from the{' '}
          <Link href="/docs/user-guide/quickstart" className="underline">
            quickstart
          </Link>
          , with the QUAIL-B benchmark&apos;s own predicates over real
          IMDB reviews. The model answers each question with one token, TRUE or FALSE.
          Quail constrains the decode step to those two tokens, so no
          open-ended generation runs. The relational part of a query,
          grouping, ordering, arithmetic, stays in the database the
          document ids came from.
        </p>
      </Section>

      <Section title="AI functions">
        <p className="text-fd-muted-foreground">
          Cloud warehouses now ship SQL functions that call a language
          model per row. Snowflake Cortex has <code>AI_FILTER</code>,{' '}
          <code>AI_CLASSIFY</code>, <code>AI_EXTRACT</code>,{' '}
          <code>AI_AGG</code>, <code>AI_SUMMARIZE_AGG</code>,{' '}
          <code>AI_SENTIMENT</code>, <code>AI_EMBED</code>, and{' '}
          <code>AI_COMPLETE</code>. BigQuery has <code>AI.IF</code>,{' '}
          <code>AI.CLASSIFY</code>, <code>AI.SCORE</code>,{' '}
          <code>AI.GENERATE_BOOL</code>, <code>AI.GENERATE_TABLE</code>,{' '}
          <code>AI.AGG</code>, and <code>AI.EMBED</code>. Each one is a
          question the model answers about a row or a pair of rows, and
          each is billed and scheduled as an independent request.
        </p>
        <p className="text-fd-muted-foreground">
          Quail treats those functions as query operators with a plan,
          a cost model, and a scheduler that shares work across rows.
          Today it supports AI-powered filters and joins: Snowflake&apos;s{' '}
          <code>AI_FILTER</code> over one table or across two or more,
          and BigQuery&apos;s <code>AI.IF</code> in the same positions.
          Operators that produce a label, a number, an extracted field,
          or a summary, such as classify, extract, score, and summarize,
          are the next step and are not built yet.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-fd-border text-left">
                <th className="py-2 pr-4 font-medium">Operator</th>
                <th className="py-2 pr-4 font-medium">Snowflake</th>
                <th className="py-2 pr-4 font-medium">BigQuery</th>
                <th className="py-2 font-medium">Quail</th>
              </tr>
            </thead>
            <tbody className="text-fd-muted-foreground">
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Filter a table</td>
                <td className="py-2 pr-4"><code>AI_FILTER</code></td>
                <td className="py-2 pr-4"><code>AI.IF</code></td>
                <td className="py-2">supported</td>
              </tr>
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Join tables on a question</td>
                <td className="py-2 pr-4"><code>AI_FILTER</code> in <code>ON</code></td>
                <td className="py-2 pr-4"><code>AI.IF</code> over a cross join</td>
                <td className="py-2">supported</td>
              </tr>
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Classify, score</td>
                <td className="py-2 pr-4"><code>AI_CLASSIFY</code>, <code>AI_SENTIMENT</code></td>
                <td className="py-2 pr-4"><code>AI.CLASSIFY</code>, <code>AI.SCORE</code></td>
                <td className="py-2">planned</td>
              </tr>
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Extract, generate</td>
                <td className="py-2 pr-4"><code>AI_EXTRACT</code>, <code>AI_COMPLETE</code></td>
                <td className="py-2 pr-4"><code>AI.GENERATE</code>, <code>AI.GENERATE_TABLE</code></td>
                <td className="py-2">planned</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 text-fd-foreground">Aggregate, summarize</td>
                <td className="py-2 pr-4"><code>AI_AGG</code>, <code>AI_SUMMARIZE_AGG</code></td>
                <td className="py-2 pr-4"><code>AI.AGG</code></td>
                <td className="py-2">planned</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Out-of-the-box capabilities">
        <div className="grid gap-4 sm:grid-cols-2">
          <Feature title="Pipelining">
            A document that passes stage 1 goes into the next forward
            pass with its stage 2 question. It does not wait for the rest
            of stage 1.
          </Feature>
          <Feature title="Token-based admission">
            Each forward pass is filled by token count and KV pages, not
            by request count, so the GPU does full work every step.
          </Feature>
          <Feature title="KV rewind">
            A document&apos;s KV is computed once and stays on the GPU
            while every later question about it runs. Only the question
            tokens are new work.
          </Feature>
          <Feature title="Packed joins">
            One anchor document&apos;s KV is shared by every partner in
            a forward pass, with gating and dedup between stages.
          </Feature>
          <Feature title="Two front ends, one plan">
            Snowflake <code>AI_FILTER</code>, BigQuery <code>AI.IF</code>,
            and a Python builder compile to the same logical plan.
          </Feature>
          <Feature title="Arrow in, Arrow out">
            Table providers stream Arrow batches; results are Arrow
            tables assembled by Acero joins over exact answer relations.
          </Feature>
          <Feature title="Multi-GPU">
            1, 2, 4, or 8 H100s in one container, one model copy each,
            with the coordinator sharding documents and anchors.
          </Feature>
          <Feature title="Comparable baselines">
            Stock vLLM, pipelined vLLM, and pipelined SGLang run the
            same queries behind the same backend interface.
          </Feature>
        </div>
      </Section>

      <Section title="Measured">
        <p className="text-fd-muted-foreground">
          LEP-1 from the QUAIL-B benchmark, one H100, Qwen3 4B fp8, all
          three configurations on the same physical GPU. Query time
          excludes model startup.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-fd-border text-left">
                <th className="py-2 pr-4 font-medium">Configuration</th>
                <th className="py-2 pr-4 font-medium text-right">Query time (s)</th>
                <th className="py-2 pr-4 font-medium text-right">Documents/s</th>
                <th className="py-2 font-medium text-right">$/query</th>
              </tr>
            </thead>
            <tbody className="text-fd-muted-foreground">
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Quail</td>
                <td className="py-2 pr-4 text-right">1.10</td>
                <td className="py-2 pr-4 text-right">454.55</td>
                <td className="py-2 text-right">0.001207</td>
              </tr>
              <tr className="border-b border-fd-border">
                <td className="py-2 pr-4 text-fd-foreground">Stock vLLM, operator-at-a-time</td>
                <td className="py-2 pr-4 text-right">1.48</td>
                <td className="py-2 pr-4 text-right">337.84</td>
                <td className="py-2 text-right">0.001624</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 text-fd-foreground">Pipelined vLLM</td>
                <td className="py-2 pr-4 text-right">1.40</td>
                <td className="py-2 pr-4 text-right">357.14</td>
                <td className="py-2 text-right">0.001536</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-fd-muted-foreground">
          Data: <code>/results/benchmarks/quailb/family-runs/20260903T052637Z-quailb-sf0.1-lf1-qwen3-4b-fp8-families/manifest.json</code> on
          the <code>quail-results</code> volume. See{' '}
          <Link href="/docs/user-guide/benchmark" className="underline">
            Running QUAIL-B
          </Link>
          .
        </p>
      </Section>

      <Section title="Architecture and customization">
        <p className="text-fd-muted-foreground">
          Quail is built like a small database engine. SQL or the builder
          produces a logical plan. Rules rewrite it. A model backend
          proposes a typed physical graph with time estimates, and a
          generic runner executes it node by node. Every built in piece
          registers through the same registry an extension would use:
        </p>
        <ul className="list-disc pl-6 text-fd-muted-foreground space-y-1">
          <li>table providers for your own storage, on the client or opened by the worker</li>
          <li>logical nodes and optimizer rules</li>
          <li>physical nodes with codecs and runtimes, physical planners, and physical rules</li>
          <li>model backends that own their scheduler, KV layout, and model calls</li>
          <li>compute providers that decide where a query runs</li>
          <li>execution observers that record per node metrics</li>
        </ul>
        <p className="text-fd-muted-foreground">
          The{' '}
          <Link href="/docs/architecture" className="underline">
            architecture section
          </Link>{' '}
          explains the layers, and{' '}
          <Link href="/docs/extending" className="underline">
            Extending Quail
          </Link>{' '}
          has a working example for each interface, taken from the test
          suite.
        </p>
      </Section>

      <Section title="Getting started">
        <ul className="list-disc pl-6 text-fd-muted-foreground space-y-1">
          <li>
            <Link href="/docs/user-guide" className="underline">
              Install
            </Link>{' '}
            the package and connect Modal.
          </li>
          <li>
            Follow the{' '}
            <Link href="/docs/user-guide/quickstart" className="underline">
              quickstart
            </Link>
            : eight real IMDB reviews, a filter, a join, and the plans
            and reports they produce.
          </li>
          <li>
            Read the{' '}
            <Link href="/docs/user-guide/sql" className="underline">
              SQL reference
            </Link>{' '}
            for what the language accepts and refuses.
          </li>
          <li>
            Run{' '}
            <Link href="/docs/user-guide/benchmark" className="underline">
              QUAIL-B
            </Link>{' '}
            to compare against stock vLLM on your own GPU budget.
          </li>
        </ul>
      </Section>
    </main>
  );
}
