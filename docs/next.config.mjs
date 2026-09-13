import { createMDX } from 'fumadocs-mdx/next';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const withMDX = createMDX({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
};

export default withMDX(config);
