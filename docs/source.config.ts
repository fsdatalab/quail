import { defineDocs, defineConfig } from 'fumadocs-mdx/config';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import remarkMath from 'remark-math';
import remarkKatexInline from './lib/remark-katex-inline.mjs';

export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export default defineConfig({
  mdxOptions: {
    rehypeCodeOptions: {
      addLanguageClass: true,
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
    },
    remarkPlugins: [remarkMath, remarkKatexInline],
  },
});
