'use client';

import { useEffect, useId, useState } from 'react';

// Mermaid renders in the browser only; the page tree is prerendered
// without it and the diagram appears after hydration.
export function Mermaid({ chart }: { chart: string }) {
  const id = useId().replace(/:/g, '');
  const [svg, setSvg] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const mermaid = (await import('mermaid')).default;
      const dark = document.documentElement.classList.contains('dark');
      mermaid.initialize({
        startOnLoad: false,
        theme: dark ? 'dark' : 'neutral',
        securityLevel: 'strict',
      });
      const { svg } = await mermaid.render(`mermaid-${id}`, chart);
      if (!cancelled) setSvg(svg);
    })();
    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  if (!svg) {
    return (
      <pre className="text-sm opacity-60">
        <code>{chart}</code>
      </pre>
    );
  }
  return (
    <div
      className="my-6 flex justify-center overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
