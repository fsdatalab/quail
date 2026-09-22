import type { ReactNode } from 'react';

export function ApiClass() {
  return (
    <span className="ms-2 inline-flex translate-y-[-0.1em] items-center rounded-md border border-fd-border bg-fd-muted px-2 py-0.5 font-sans text-xs font-medium text-fd-muted-foreground">
      Class
    </span>
  );
}

export function ApiMethod({ children }: { children: ReactNode }) {
  return (
    <div className="my-4 flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-lg border border-fd-border bg-fd-card px-4 py-3">
      <span className="font-sans text-xs font-medium text-fd-muted-foreground">
        Method
      </span>
      <code className="text-sm">{children}</code>
    </div>
  );
}
