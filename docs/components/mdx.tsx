import defaultMdxComponents from 'fumadocs-ui/mdx';
import { Callout } from 'fumadocs-ui/components/callout';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { TypeTable } from 'fumadocs-ui/components/type-table';
import type { MDXComponents } from 'mdx/types';
import { ApiClass, ApiMethod } from './api-reference';
import { DeriveFigure, PlanStagesFigure, SeveralRegionsFigure } from './planning-details';
import { PlanningFigure } from './planning-figure';
import { QueryOutput } from './query-output';
import { CodeExample } from './code-example';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    pre: CodeExample,
    ApiClass,
    ApiMethod,
    Callout,
    DeriveFigure,
    PlanStagesFigure,
    PlanningFigure,
    QueryOutput,
    SeveralRegionsFigure,
    Step,
    Steps,
    Tab,
    Tabs,
    TypeTable,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
