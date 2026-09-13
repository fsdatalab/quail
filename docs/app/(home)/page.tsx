import Link from 'next/link';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';

const example = `SELECT c.comment_id, f.field
FROM comments c
JOIN fields f
  ON AI_FILTER(PROMPT(
       'Does the description in DOCUMENT {1} apply to '
       'the comment in DOCUMENT {0}?',
       c.text, f.statement
     ))
WHERE AI_FILTER(PROMPT(
  'Is the comment in DOCUMENT {0} hateful, threatening, or abusive?',
  c.text
))`;

function Section({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="grid gap-6 border-t border-fd-border pt-10 md:grid-cols-[14rem_1fr]">
      <div>
        {eyebrow ? (
          <p className="mb-2 font-mono text-xs font-medium uppercase tracking-[0.18em] text-fd-muted-foreground">
            {eyebrow}
          </p>
        ) : null}
        <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function Mechanism({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[2rem_1fr] gap-3 border-t border-fd-border py-5 first:border-t-0 first:pt-0">
      <span className="font-mono text-xs text-fd-muted-foreground">{number}</span>
      <div>
        <h3 className="mb-1 font-medium">{title}</h3>
        <p className="text-sm leading-6 text-fd-muted-foreground">{children}</p>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-20 px-6 py-16 md:py-24">
      <header className="relative overflow-hidden border-y border-fd-border py-12 md:py-20">
        <div className="pointer-events-none absolute inset-y-0 right-0 hidden w-2/5 border-l border-fd-border bg-[linear-gradient(to_right,var(--color-fd-border)_1px,transparent_1px),linear-gradient(to_bottom,var(--color-fd-border)_1px,transparent_1px)] bg-[size:32px_32px] opacity-35 md:block" />
        <div className="relative max-w-3xl">
          <p className="mb-5 font-mono text-xs font-medium uppercase tracking-[0.2em] text-fd-muted-foreground">
            Quail
          </p>
          <h1 className="text-5xl font-semibold tracking-[-0.04em] md:text-7xl">
            A declarative, extensible
            <span className="block text-fd-muted-foreground">
              query engine for AI SQL.
            </span>
          </h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-fd-muted-foreground">
            Use LLM-powered operators to filter documents, join collections,
            and return structured results.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/docs/user-guide/quickstart"
              className="rounded-md bg-fd-primary px-4 py-2.5 text-sm font-medium text-fd-primary-foreground"
            >
              Run the quickstart
            </Link>
            <Link
              href="/docs"
              className="rounded-md border border-fd-border bg-fd-background px-4 py-2.5 text-sm font-medium"
            >
              Read the docs
            </Link>
          </div>
        </div>
      </header>

      <Section eyebrow="01 / Analysis" title="Describe the query">
        <div className="min-w-0 space-y-4">
          <p className="max-w-3xl leading-7 text-fd-muted-foreground">
            Consider the 448,000 comments in the Jigsaw Civil Comments
            dataset. We first ask which comments are toxic. For each comment
            that passes, we ask 31 more questions about toxicity type,
            identity references, and moderator decisions.
          </p>
          <p className="max-w-3xl leading-7 text-fd-muted-foreground">
            In Quail, the model predicates are operators in one relational
            query. You do not write request loops, parse model responses, or
            connect intermediate ids by hand.
          </p>
        </div>
      </Section>

      <Section eyebrow="02 / AI SQL" title="Write the filter and join">
        <div className="min-w-0 space-y-4">
          <DynamicCodeBlock lang="sql" code={example} />
          <p className="max-w-3xl leading-7 text-fd-muted-foreground">
            The model answers each predicate with one constrained token:{' '}
            <code>TRUE</code> or <code>FALSE</code>. Quail returns the matching
            comment and field pairs as an Arrow table.
          </p>
          <p className="max-w-3xl text-sm leading-6 text-fd-muted-foreground">
            The complete query is in{' '}
            <code>demos/civil_comments_join.py</code>.
          </p>
        </div>
      </Section>

      <Section eyebrow="03 / Execution" title="Share work across stages">
        <div>
          <Mechanism number="01" title="Pipelining">
            A comment that passes the toxicity filter can enter the join
            immediately. It does not wait for every other comment.
          </Mechanism>
          <Mechanism number="02" title="Token-based admission">
            The scheduler fills each forward pass by token count and
            available KV, rather than by request count.
          </Mechanism>
          <Mechanism number="03" title="KV rewind">
            When capacity allows, a document&apos;s KV stays on the GPU for
            its later predicates. Quail can then avoid computing the prefix
            again.
          </Mechanism>
          <Mechanism number="04" title="Packed joins">
            One anchor document shares its KV across many join partners in
            the same forward pass.
          </Mechanism>
          <p className="border-t border-fd-border pt-5 text-sm leading-6 text-fd-muted-foreground">
            Pipelining and admission reduce waiting and unused batch
            capacity. KV rewind and packed joins reduce fresh input-token
            computation when the needed KV remains available.
          </p>
        </div>
      </Section>

      <Section eyebrow="04 / Evaluation" title="Measured on 33 queries">
        <div className="space-y-4">
          <p className="max-w-3xl leading-7 text-fd-muted-foreground">
            In the September 12, 2026 QUAIL-B sf=0.1 run with Qwen3 4B
            fp8 and one H100 per configuration, Quail had lower query time
            than stock vLLM on 31 of 33 queries. Stock vLLM used
            operator-at-a-time execution. Total query time was 1,542.13
            seconds, compared with 3,249.04 seconds for stock vLLM. Total
            GPU cost was 52.5% lower.
          </p>
          <p className="max-w-3xl leading-7 text-fd-muted-foreground">
            On IMDB-10, Quail computed 5.48 million fresh input tokens,
            compared with 7.71 million for stock vLLM. Query time was 48.58
            seconds, compared with 85.89 seconds. Each backend&apos;s answers
            can change the rows that reach later stages, so the token
            difference is not execution-only. Query time excludes model
            startup.
          </p>
          <p className="max-w-3xl text-sm leading-6 text-fd-muted-foreground">
            <Link href="/docs/user-guide/benchmark" className="underline">
              Run QUAIL-B
            </Link>{' '}
            to measure accuracy, output precision and recall, fresh input
            tokens, throughput, and GPU cost.
          </p>
        </div>
      </Section>

      <Section eyebrow="05 / Operators" title="LLM-powered operators">
        <div className="grid gap-px overflow-hidden rounded-lg border border-fd-border bg-fd-border sm:grid-cols-2">
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Filters and joins</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              True-or-false filters, AI joins, <code>EXISTS</code>, and{' '}
              <code>NOT EXISTS</code>, through Snowflake-style SQL,
              BigQuery-style SQL, or Python.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Structured outputs</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              <code>AI.CLASSIFY</code>, <code>AI.EXTRACT</code>, and{' '}
              <code>AI.MAP</code> will add labels, typed fields, and typed
              values to the current true-or-false operators.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Models and data</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              Qwen3 4B fp8 or Qwen3 32B fp8 over Arrow, Parquet, and Hugging
              Face tables. Results are Arrow tables.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Hardware</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              One H100 per model copy. A query can use 1, 2, 4, or 8 H100s
              in one Modal container.
            </p>
          </div>
        </div>
      </Section>

      <Section eyebrow="06 / Next" title="Start with a real query">
        <div className="grid gap-4 sm:grid-cols-2">
          {[
            {
              href: '/docs/user-guide/quickstart',
              title: 'Quickstart',
              text: 'Run IMDB-1 on eight published reviews and inspect the result.',
            },
            {
              href: '/docs/user-guide/sql',
              title: 'SQL reference',
              text: 'See every accepted filter, join, and prompt form.',
            },
            {
              href: '/docs/architecture',
              title: 'Architecture',
              text: 'Follow one query from SQL to the GPU and back.',
            },
            {
              href: '/docs/user-guide/benchmark',
              title: 'QUAIL-B',
              text: 'Compare Quail with stock vLLM on the same queries.',
            },
          ].map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="group rounded-lg border border-fd-border p-5 transition-colors hover:bg-fd-accent"
            >
              <h3 className="font-medium">
                {item.title}{' '}
                <span className="inline-block transition-transform group-hover:translate-x-1">
                  →
                </span>
              </h3>
              <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
                {item.text}
              </p>
            </Link>
          ))}
        </div>
      </Section>
    </main>
  );
}
