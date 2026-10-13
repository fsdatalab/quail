import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getPageImageUrl, source } from '@/lib/source';
import { notFound } from 'next/navigation';
import { ImageResponse } from 'next/og';
import { appName } from '@/lib/shared';

export const revalidate = false;

// Matches the Full Stack Data Lab card at fsdatalab.github.io/assets/og-image.png.
const red = '#C41230';
const fontDir = join(process.cwd(), 'assets/fonts');

// Bars from fsdatalab.github.io/assets/logo.png at 1x, without its red tile.
function Logo() {
  return (
    <svg width={128} height={138} viewBox="0 0 128 138">
      <rect x="0" y="0" width="128" height="30" rx="4" fill="white" />
      <rect x="0" y="36" width="64" height="30" rx="4" fill="white" />
      <rect x="0" y="72" width="32" height="30" rx="4" fill="white" />
      <rect x="0" y="108" width="32" height="30" rx="4" fill="white" />
    </svg>
  );
}

export async function GET(_req: Request, { params }: RouteContext<'/og/docs/[...slug]'>) {
  const { slug } = await params;
  const page = source.getPage(slug.slice(0, -1));
  if (!page) notFound();

  const [lora, inter] = await Promise.all([
    readFile(join(fontDir, 'Lora-Bold.ttf')),
    readFile(join(fontDir, 'Inter-Regular.ttf')),
  ]);
  const title = page.slugs.length === 0 ? appName : page.data.title;
  const kicker = page.slugs.length === 0 ? 'Documentation' : 'Quail docs';
  const description =
    page.data.description ??
    (page.slugs.length === 0 ? 'A query engine for AI functions in SQL.' : undefined);

  return new ImageResponse(
    (
      <div
        style={{
          display: 'flex',
          width: '100%',
          height: '100%',
          background: red,
          color: 'white',
          padding: '0 96px',
          alignItems: 'center',
          gap: 64,
          fontFamily: 'Inter',
        }}
      >
        <Logo />
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <div style={{ fontSize: 30, opacity: 0.85 }}>{kicker}</div>
          <div
            style={{
              fontFamily: 'Lora',
              fontSize: title.length > 28 ? 64 : 80,
              lineHeight: 1.1,
              marginTop: 12,
            }}
          >
            {title}
          </div>
          {description ? (
            <div style={{ fontSize: 30, lineHeight: 1.35, marginTop: 20, opacity: 0.92 }}>
              {description}
            </div>
          ) : null}
          <div style={{ fontSize: 26, marginTop: 40, opacity: 0.8 }}>
            Full Stack Data Lab, Carnegie Mellon University
          </div>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
      fonts: [
        { name: 'Lora', data: lora, weight: 700, style: 'normal' },
        { name: 'Inter', data: inter, weight: 400, style: 'normal' },
      ],
    },
  );
}

export function generateStaticParams() {
  return source.getPages().map((page) => ({
    lang: page.locale,
    slug: getPageImageUrl(page).segments,
  }));
}
