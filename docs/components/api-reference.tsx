import type { ReactNode } from 'react';

export function ApiClass() {
  return (
    <div className="mb-[-0.75rem] mt-8 flex">
      <span className="inline-flex items-center rounded-md border border-fd-border bg-fd-muted px-2 py-0.5 font-sans text-xs font-medium text-fd-muted-foreground">
        Class
      </span>
    </div>
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
