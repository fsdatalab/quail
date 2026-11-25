import Image from 'next/image';
import Link from 'next/link';
import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  ArrowRight,
  Cpu,
  FileSearch,
  FileText,
  Filter,
  Gauge,
  Images,
  ListTree,
  PenLine,
  Rocket,
  Server,
  Tags,
  Terminal,
  TextCursor,
  Wrench,
} from 'lucide-react';
import quailLogo from '@/assets/quail-logo.svg';
import { gitConfig } from '@/lib/shared';

const reviews = [
  { id: 'r1', body: 'The ending was perfect and the cast was great.', kept: true },
  { id: 'r2', body: 'Too long, and the ending made no sense.', kept: true },
  { id: 'r3', body: 'Great music, but I want my money back.', kept: false },
];

/** Shows the lab, the logo, a summary, and an example query on the home page. */
export function HomeHero() {
  return (
    <header className="quail-hero not-prose">
      <div className="quail-hero-text">
        <p className="quail-hero-lab">
          <a href="https://fsdatalab.github.io/">Full Stack Data Lab</a> at
          Carnegie Mellon University
        </p>
        <h1 className="quail-hero-title">
          <Image src={quailLogo} alt="" width={56} height={56} priority />
          Quail
        </h1>
        <p className="quail-hero-lead">
          A fast engine for AI functions in SQL.
        </p>
        <div className="quail-hero-actions">
          <Link className="quail-hero-primary" href="/docs/user-guide/quickstart">
            Get started <ArrowRight size={16} aria-hidden />
          </Link>
          <a
            className="quail-hero-secondary"
            href={`https://github.com/${gitConfig.user}/${gitConfig.repo}`}
          >
            GitHub
          </a>
        </div>
      </div>
      <figure className="quail-hero-query">
        <figcaption>
          <span className="quail-hero-dots" aria-hidden>
            <span />
            <span />
            <span />
          </span>
          query.sql
        </figcaption>
        <pre>
          <span className="k">SELECT</span> r.id <span className="k">FROM</span> reviews r{'\n'}
          <span className="k">WHERE</span> <span className="f">AI.IF</span>(<span className="f">PROMPT</span>({'\n'}
          {'  '}<span className="s">&apos;Does the review discuss the ending? {'{0}'}&apos;</span>,{'\n'}
          {'  '}r.body))
        </pre>
        <table>
          <thead>
            <tr>
              <th>id</th>
              <th>body</th>
              <th>AI.IF</th>
            </tr>
          </thead>
          <tbody>
            {reviews.map(({ id, body, kept }) => (
              <tr key={id} className={kept ? undefined : 'quail-hero-dropped'}>
                <td>{id}</td>
                <td>{body}</td>
                <td>
                  <span className={kept ? 'quail-hero-true' : 'quail-hero-false'}>
                    {kept ? 'true' : 'false'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </figure>
    </header>
  );
}

type Guide = { title: string; href: string; icon: LucideIcon; description: string };

const guides: Guide[] = [
  {
    title: 'Quickstart',
    href: '/docs/user-guide/quickstart',
    icon: Rocket,
    description: 'Run your first query.',
  },
  {
    title: 'Running queries',
    href: '/docs/user-guide/sessions',
    icon: Terminal,
    description: 'Choose a model, register tables, and run SQL.',
  },
  {
    title: 'SQL reference',
    href: '/docs/user-guide/sql',
    icon: ListTree,
    description: 'Every AI function and option.',
  },
  {
    title: 'Multimodal data',
    href: '/docs/user-guide/multimodal-data',
    icon: Images,
    description: 'Query images and PDFs.',
  },
  {
    title: 'Models and GPUs',
    href: '/docs/user-guide/models',
    icon: Cpu,
    description: 'Supported models and GPUs, and Modal.',
  },
  {
    title: 'Quail Server',
    href: '/docs/user-guide/server',
    icon: Server,
    description: 'Run queries on a remote GPU machine.',
  },
];

/** Shows the main guides as a grid of linked cells. */
export function HomeGuides() {
  return (
    <nav className="quail-guides not-prose" aria-label="Guides">
      {guides.map(({ title, href, icon: Icon, description }) => (
        <Link key={href} href={href} className="quail-guide">
          <Icon size={20} aria-hidden />
          <span className="quail-guide-title">{title}</span>
          <span className="quail-guide-description">{description}</span>
        </Link>
      ))}
    </nav>
  );
}

const functionsPage = '/docs/user-guide/sql/ai-functions';

const functions: Guide[] = [
  { title: 'AI.IF', href: '#aiifprompt-options', icon: Filter, description: 'Yes or no' },
  { title: 'AI.SCORE', href: '#aiscoreprompt-options', icon: Gauge, description: 'A score from 0 to 1' },
  { title: 'AI.CLASSIFY', href: '#aiclassifyinput-categories-options', icon: Tags, description: 'One label from a list' },
  { title: 'AI.EXTRACT', href: '#aiextractinput-questions', icon: FileSearch, description: 'Answers to questions' },
  { title: 'AI.EXTRACT_SPAN', href: '#aiextract_spaninput-questions', icon: TextCursor, description: 'Answers copied from the text' },
  { title: 'AI.MAP', href: '#aimapinput-instruction-max_tokens', icon: PenLine, description: 'Open-ended text' },
  { title: 'AI.PARSE_DOCUMENT', href: '#aiparse_documentdocument', icon: FileText, description: 'The text of a PDF' },
  { title: 'AI.SAE_FEATURES', href: '#aisae_featuresprompt-options', icon: Wrench, description: 'Sparse autoencoder features' },
];

/** Lists the AI functions with links to their reference entries. */
export function HomeFunctions() {
  return (
    <div className="quail-functions not-prose">
      {functions.map(({ title, href, icon: Icon, description }) => (
        <Link key={href} href={`${functionsPage}${href}`} className="quail-function">
          <span className="quail-function-icon">
            <Icon size={16} aria-hidden />
          </span>
          <span>
            <code>{title}</code>
            <span className="quail-function-description">{description}</span>
          </span>
        </Link>
      ))}
    </div>
  );
}

/** Lays out a home page section with the heading on the left and the content on the right. */
export function HomeSection({
  title,
  description,
  children,
}: {
  title: string;
  description: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="quail-home-section">
      <div className="quail-home-section-intro not-prose">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      <div>{children}</div>
    </section>
  );
}
