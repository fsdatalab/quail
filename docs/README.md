# Quail docs

The documentation site for Quail. It is a [Fumadocs](https://fumadocs.dev)
site on Next.js. The pages are MDX files under `content/docs/`.

```bash
cd docs
pnpm install
pnpm dev      # http://localhost:3000
pnpm build    # static build check
```

Page order inside each folder comes from that folder's `meta.json`.
The planning figure on the Developers page is an inline SVG,
`components/planning-figure.tsx`.
