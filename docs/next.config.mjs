import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  basePath: '/quail',
  // a static host has no image optimizer, so pages link each image file
  images: { unoptimized: true },
};

export default withMDX(config);
