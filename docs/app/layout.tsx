import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Inter, Lora, IBM_Plex_Mono } from 'next/font/google';

const sans = Inter({
  subsets: ['latin'],
  variable: '--font-quail-sans',
  weight: ['400', '500', '600'],
});

const serif = Lora({
  subsets: ['latin'],
  variable: '--font-quail-serif',
  weight: ['600', '700'],
});

const mono = IBM_Plex_Mono({
  subsets: ['latin'],
  variable: '--font-quail-mono',
  weight: ['400', '500'],
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${serif.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
