import Link from 'next/link';
import { DynamicCodeBlock } from 'fumadocs-ui/components/dynamic-codeblock';

const example = `import quail

with quail.Session() as session:
    session.register(
        "reviews",
        quail.DocumentProvider.from_table(reviews, id_col="id"),
    )
    result = session.sql("""
        SELECT r.id
        FROM reviews r
        WHERE AI_FILTER(PROMPT(
            'Does the review in DOCUMENT {0} praise the movie?',
            r.body
        ))
    """).run()
    rows = result.collect()
    metrics = result.report`;

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
            Query-aware inference
          </p>
          <h1 className="text-5xl font-semibold tracking-[-0.04em] md:text-7xl">
            Ask models about data.
            <span className="block text-fd-muted-foreground">
              Plan the work first.
            </span>
          </h1>
          <p className="mt-7 max-w-2xl text-lg leading-8 text-fd-muted-foreground">
            Quail runs language-model filters and joins over document
            collections. Write SQL or Python. Quail schedules the complete
            query so model calls can share work on your GPUs.
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

      <Section eyebrow="01 / Query" title="One query, one planned job">
        <div className="min-w-0 space-y-4">
          <DynamicCodeBlock lang="python" code={example} />
          <p className="max-w-3xl text-sm leading-6 text-fd-muted-foreground">
            Register an Arrow table, write an <code>AI_FILTER</code>, and
            collect an Arrow result. The same query can be built with the
            Python API. The model returns one constrained token for each
            predicate: <code>TRUE</code> or <code>FALSE</code>.
          </p>
        </div>
      </Section>

      <Section eyebrow="02 / Execution" title="Share work across stages">
        <div>
          <Mechanism number="01" title="Pipelining">
            A document starts its next predicate as soon as it passes the
            current one. It does not wait for the rest of the stage.
          </Mechanism>
          <Mechanism number="02" title="Token-based admission">
            The scheduler fills each forward pass by token count and
            available KV, rather than by request count.
          </Mechanism>
          <Mechanism number="03" title="KV rewind">
            A document&apos;s KV stays on the GPU for its later predicates.
            Quail computes the document prefix once.
          </Mechanism>
          <Mechanism number="04" title="Packed joins">
            One anchor document shares its KV across many join partners in
            the same forward pass.
          </Mechanism>
        </div>
      </Section>

      <Section eyebrow="03 / Scope" title="Focused by design">
        <div className="grid gap-px overflow-hidden rounded-lg border border-fd-border bg-fd-border sm:grid-cols-2">
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Queries</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              True-or-false filters, AI joins, <code>EXISTS</code>, and{' '}
              <code>NOT EXISTS</code>, through Snowflake-style SQL,
              BigQuery-style SQL, or Python.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Data</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              Arrow tables, Parquet files, Arrow datasets, and Hugging Face
              datasets in. Arrow tables out.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Models</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              Qwen3 4B fp8 or Qwen3 32B fp8. Each GPU holds one model copy.
            </p>
          </div>
          <div className="bg-fd-background p-5">
            <h3 className="font-medium">Hardware</h3>
            <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">
              A local CUDA GPU or 1, 2, 4, or 8 H100s in one Modal
              container. Quail does not split one model across GPUs.
            </p>
          </div>
        </div>
      </Section>

      <Section eyebrow="04 / Next" title="Start with a real query">
        <div className="grid gap-4 sm:grid-cols-2">
          {[
            {
              href: '/docs/user-guide/quickstart',
              title: 'Quickstart',
              text: 'Run QUAIL-B IMDB-1 on 100 reviews and inspect the result.',
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
