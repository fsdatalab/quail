import { CodeBlock, Pre } from 'fumadocs-ui/components/codeblock';
import { isValidElement, type ComponentProps } from 'react';

const languageNames: Record<string, string> = {
  bash: 'Terminal',
  sh: 'Terminal',
  shell: 'Terminal',
  python: 'Python',
  sql: 'SQL',
  json: 'JSON',
  yaml: 'YAML',
  toml: 'TOML',
  rust: 'Rust',
  text: 'Output',
};

export function CodeExample(props: ComponentProps<typeof CodeBlock>) {
  const language = isValidElement<{ className?: string }>(props.children)
    ? props.children.props.className
        ?.split(' ')
        .find((name) => name.startsWith('language-'))
        ?.slice('language-'.length)
    : undefined;

  return (
    <CodeBlock
      {...props}
      title={props.title ?? (language ? languageNames[language] ?? language : undefined)}
    >
      <Pre>{props.children}</Pre>
    </CodeBlock>
  );
}
