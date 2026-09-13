import type { ReactNode } from 'react';

type Metric = {
  label: string;
  value: string;
  detail?: string;
};

export function QueryOutput({
  title,
  subtitle,
  metrics,
  children,
}: {
  title: string;
  subtitle: string;
  metrics: Metric[];
  children?: ReactNode;
}) {
  return (
    <section className="my-8 overflow-hidden rounded-xl border border-fd-border bg-fd-card shadow-sm">
      <header className="flex flex-col gap-2 border-b border-fd-border bg-black px-5 py-4 text-white sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-mono text-[0.7rem] font-medium uppercase tracking-[0.2em] text-[#ff8294]">
            Measured output
          </p>
          <h3 className="mt-1 text-base font-semibold">{title}</h3>
        </div>
        <p className="max-w-sm text-sm text-[#e0e0e0] sm:text-right">
          {subtitle}
        </p>
      </header>
      <div className="grid gap-px bg-fd-border sm:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <div key={metric.label} className="bg-fd-background px-5 py-4">
            <p className="text-xs font-medium uppercase tracking-wide text-fd-muted-foreground">
              {metric.label}
            </p>
            <p className="mt-2 font-mono text-2xl font-medium tracking-tight text-fd-foreground">
              {metric.value}
            </p>
            {metric.detail ? (
              <p className="mt-1 text-xs text-fd-muted-foreground">
                {metric.detail}
              </p>
            ) : null}
          </div>
        ))}
      </div>
      {children ? (
        <div className="border-t border-fd-border px-5 py-4 [&>*:last-child]:mb-0 [&>*:first-child]:mt-0">
          {children}
        </div>
      ) : null}
    </section>
  );
}
