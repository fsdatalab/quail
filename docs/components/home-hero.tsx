import Image from 'next/image';
import Link from 'next/link';
import quailLogo from '@/assets/quail-logo.svg';
import { gitConfig } from '@/lib/shared';

/** Shows the logo, the lab, the tagline, and the install command on the home page. */
export function HomeHero() {
  return (
    <header className="quail-hero">
      <Image src={quailLogo} alt="Quail logo" width={88} height={88} priority />
      <p className="quail-hero-lab">
        <a href="https://fsdatalab.github.io/">Full Stack Data Lab</a>, Carnegie
        Mellon University
      </p>
      <h1 className="quail-hero-title">Quail</h1>
      <p className="quail-hero-tagline">SQL with AI functions on your GPUs</p>
      <div className="quail-hero-actions">
        <Link className="quail-hero-primary" href="/docs/user-guide/quickstart">
          Quickstart
        </Link>
        <a
          className="quail-hero-secondary"
          href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
        >
          GitHub
        </a>
        <code className="quail-hero-install">uv pip install quail-engine</code>
      </div>
    </header>
  );
}
