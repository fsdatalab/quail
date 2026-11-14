import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import Image from 'next/image';
import quailLogo from '@/assets/quail-logo.svg';
import { appName, gitConfig } from './shared';

export function baseOptions(): BaseLayoutProps {
  return {
    themeSwitch: {
      mode: 'light-dark',
      className: 'quail-theme-switch',
    },
    nav: {
      title: (
        <span className="quail-brand">
          <Image src={quailLogo} alt="" width={32} height={32} />
          <span className="quail-brand-name">{appName}</span>
          <span className="quail-brand-label">Docs</span>
        </span>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
